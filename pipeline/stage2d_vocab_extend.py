"""
Stage 2d — Vocabulary Extension + LoRA Fine-tune
==================================================
Answers the second structural bug: Stage 3's LLM compression always used
Airavata's own tokenizer ("tokenizer": "model_default"), never the DevAware
tokenizer, because you cannot swap a pretrained LM's tokenizer for an
unrelated one and get meaningful next-token probabilities -- the LM head
was trained against its OWN vocabulary's embeddings. A different 32K-vocab
tokenizer produces token ids the model has never seen a gradient for.

There are two honest ways to close this gap. This file implements the real
one (fine-tuning); the other (cheaper) option is a scope decision, not code:
report the DevAware tokenizer as an intrinsic tokenizer-efficiency metric
only (Stage 2b's tokens/char, vowel-split-%, etc.), and state explicitly in
the paper that Stage 3 LLM-compression uses the model's native tokenizer.
That's defensible and requires no further engineering -- just say so in the
methodology section instead of silently defaulting to it. Reach for this
file only if you actually want the three-condition table with a working
"your tokenizer" column and have the GPU time to spend on it.

What this does, instead of a full vocabulary replacement (which would mean
retraining Airavata's embedding + LM head from scratch -- not something a
Kaggle session budget supports):

1. VOCABULARY EXTENSION, not replacement. From the DevAware SentencePiece
   model, pull out only the pieces that are genuinely novel: multi-akshara
   merges that Airavata's own tokenizer cannot already produce as a single
   token. Add those as new tokens on top of Airavata's existing vocabulary
   (tokenizer.add_tokens), so old text still tokenizes exactly as before and
   only genuinely new merged units are affected.

2. SMART EMBEDDING INIT. Each new token's embedding is initialized as the
   mean of Airavata's own embeddings for the sub-tokens it used to need to
   spell that same string. This is the standard trick for vocabulary
   extension (used e.g. in Chinese-LLaMA-style vocab-extension work) and
   gives the new tokens a much better starting point than random init, which
   directly reduces the fine-tuning budget needed to make them useful.

3. LORA + FULL EMBEDDING FINE-TUNE. The transformer blocks get LoRA
   adapters (cheap, a few million trainable params). The embedding matrix
   and LM head are fine-tuned in full -- they're the piece that actually
   needs to learn what the new tokens mean, and full fine-tuning there is
   still small relative to the frozen backbone.

4. A short continued-pretraining pass (causal LM loss) on the language's
   own corpus, using the extended tokenizer, adapts the model to the new
   vocabulary.

The result is a real, loadable (model, tokenizer) pair that Stage 3 can use
for a genuine "your tokenizer" condition -- see the LLMCompressor patch in
stage3_compress.py.

Usage:
    python -m pipeline.stage2d_vocab_extend --lang hindi
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from pipeline.config import (
    SPLIT_DIR, TOKENIZER_DIR, FINETUNE_DIR, LANGUAGES,
    INDIC_LLM_MODEL, VOCAB_EXTEND_MIN_AKSHARAS, VOCAB_EXTEND_MAX_NEW_TOKENS,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    FINETUNE_BLOCK_SIZE, FINETUNE_BATCH_SIZE, FINETUNE_GRAD_ACCUM_STEPS,
    FINETUNE_LR, FINETUNE_STEPS, FINETUNE_WARMUP_STEPS, FINETUNE_SAVE_EVERY,
    FINETUNE_LOG_EVERY, FINETUNE_MAX_TRAIN_CHARS, ensure_dirs,
)
from pipeline.devaware_tokenizer import DevAwareTokenizer
from pipeline.stage2b_devanagari_tokenizer import segment_into_aksharas


# ─── Step 1: find genuinely novel merged pieces ─────────────────────────────

def find_novel_merged_pieces(dev_tok: DevAwareTokenizer, base_tokenizer,
                              min_aksharas: int = VOCAB_EXTEND_MIN_AKSHARAS,
                              max_new: int = VOCAB_EXTEND_MAX_NEW_TOKENS) -> list:
    """
    Scan the DevAware SentencePiece vocab and keep only pieces that:
      (a) decode to >= min_aksharas real aksharas (i.e. are actual
          multi-akshara merges, not just a single protected akshara), and
      (b) are NOT already a single token in the base (Airavata) tokenizer.

    Returns a list of Devanagari strings (the new tokens to add), longest
    akshara-span first, capped at max_new.
    """
    candidates = []
    for piece_id in range(dev_tok.vocab_size()):
        piece = dev_tok.sp.IdToPiece(piece_id)
        if piece in ("<unk>", "<s>", "</s>", "<pad>"):
            continue
        decoded = dev_tok.piece_to_devanagari(piece).strip(" ")
        if not decoded:
            continue

        n_aksharas = sum(
            1 for seg in segment_into_aksharas(decoded)
            if seg != "▁"
        )
        if n_aksharas < min_aksharas:
            continue

        # Already representable as one token in the base tokenizer? Then
        # there's nothing new to add.
        base_ids = base_tokenizer.encode(decoded, add_special_tokens=False)
        if len(base_ids) <= 1:
            continue

        candidates.append((n_aksharas, decoded))

    # Longest merges first: they're the ones carrying the most information
    # about script-aware structure, and the most expensive for the base
    # tokenizer to spell out piece-by-piece today.
    candidates.sort(key=lambda x: (-x[0], x[1]))
    seen = set()
    new_tokens = []
    for _, decoded in candidates:
        if decoded in seen:
            continue
        seen.add(decoded)
        new_tokens.append(decoded)
        if len(new_tokens) >= max_new:
            break

    return new_tokens


# ─── Step 2: extend tokenizer + smart-init embeddings ───────────────────────
#
# IMPORTANT ordering constraint: sub-token decompositions of each new_token
# string must be captured with base_tokenizer.encode() BEFORE calling
# add_tokens() -- once a string is added as its own token, encode() returns
# that new single id instead of the old multi-token decomposition needed for
# smart init. build_vocab_extended_model() below does this in the correct
# order; there's no standalone helper here to avoid that footgun.


# ─── Step 3: LoRA wrap + trainable embedding/head ───────────────────────────

def apply_lora_and_unfreeze_embeddings(model):
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)

    # peft freezes everything except LoRA params by default. The whole
    # point here is that the new tokens' meaning lives in the embedding /
    # LM head, so those need to be trainable too, not just the adapters.
    for name, param in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")
    return model


# ─── Step 4: continued-pretraining data + loop ───────────────────────────────

class BlockDataset(Dataset):
    """Tokenizes a text file with the extended tokenizer and chunks it into
    fixed-length blocks for causal LM training."""

    def __init__(self, text: str, tokenizer, block_size: int):
        ids = tokenizer.encode(text, add_special_tokens=False)
        n_blocks = len(ids) // block_size
        ids = ids[: n_blocks * block_size]
        self.examples = [
            ids[i:i + block_size] for i in range(0, len(ids), block_size)
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.examples[idx], dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def finetune(model, tokenizer, train_text: str, device: str, save_dir: Path):
    model.train()
    dataset = BlockDataset(train_text, tokenizer, FINETUNE_BLOCK_SIZE)
    print(f"  Fine-tune corpus: {len(train_text):,} chars -> "
          f"{len(dataset):,} blocks of {FINETUNE_BLOCK_SIZE} tokens")

    if len(dataset) == 0:
        print("  ⚠ Not enough data to build even one training block -- skipping fine-tune.")
        return model

    loader = DataLoader(dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=FINETUNE_LR
    )
    total_steps = min(FINETUNE_STEPS, len(loader) * 50)  # don't schedule past what data supports many epochs of
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, step / max(1, FINETUNE_WARMUP_STEPS))
        * max(0.0, (total_steps - step) / max(1, total_steps)),
    )

    step = 0
    t0 = time.time()
    optimizer.zero_grad()
    data_iter = iter(loader)
    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / FINETUNE_GRAD_ACCUM_STEPS
        loss.backward()

        if (step + 1) % FINETUNE_GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % FINETUNE_LOG_EVERY == 0:
            elapsed = time.time() - t0
            print(f"    step {step:5d}/{total_steps}  "
                  f"loss={outputs.loss.item():.4f}  "
                  f"ppl={math.exp(min(outputs.loss.item(), 20)):.2f}  "
                  f"({elapsed:.0f}s elapsed)", flush=True)

        if step > 0 and step % FINETUNE_SAVE_EVERY == 0:
            _save_checkpoint(model, tokenizer, save_dir, step)

        step += 1

    _save_checkpoint(model, tokenizer, save_dir, step, final=True)
    model.eval()
    return model


def _save_checkpoint(model, tokenizer, save_dir: Path, step: int, final: bool = False):
    out_dir = save_dir / ("final" if final else f"step_{step}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)   # saves LoRA adapter + resized embed/head
    tokenizer.save_pretrained(out_dir)
    print(f"  ✓ Checkpoint saved: {out_dir}")


# ─── Orchestration ────────────────────────────────────────────────────────

def build_vocab_extended_model(lang: str, device: str = None):
    """
    Full Stage 2d pipeline for one language:
      DevAware SPM -> novel merged pieces -> extend Airavata's tokenizer
      -> smart-init new embeddings -> LoRA-wrap -> fine-tune -> save.

    Returns (model, tokenizer) ready to hand to LLMCompressor.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ensure_dirs()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tok_dir = TOKENIZER_DIR / lang
    dev_tok = DevAwareTokenizer(
        spm_model_path=tok_dir / f"devaware_bpe_{lang}.model",
        pua_map_path=tok_dir / f"akshara_pua_map_{lang}.json",
    )

    print(f"  Loading base tokenizer + model: {INDIC_LLM_MODEL}")
    base_tokenizer = AutoTokenizer.from_pretrained(INDIC_LLM_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        INDIC_LLM_MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)

    print(f"  Finding novel multi-akshara merges (base vocab size={len(base_tokenizer)})...")
    new_tokens = find_novel_merged_pieces(dev_tok, base_tokenizer)
    print(f"  ✓ Found {len(new_tokens)} novel merged tokens to add "
          f"(min_aksharas={VOCAB_EXTEND_MIN_AKSHARAS}, "
          f"cap={VOCAB_EXTEND_MAX_NEW_TOKENS})")

    if not new_tokens:
        print("  ⚠ No novel merged tokens found -- DevAware tokenizer isn't "
              "adding anything the base tokenizer can't already do. Nothing "
              "to fine-tune; the 'your tokenizer' condition would be "
              "identical to 'model_default'. Check Stage 2b results first.")
        return model, base_tokenizer

    # Capture sub-token decompositions BEFORE mutating the tokenizer with
    # add_tokens(), since encode() on a newly-added token returns its own
    # id afterwards, not the old decomposition.
    old_decompositions = {
        tok: base_tokenizer.encode(tok, add_special_tokens=False)
        for tok in new_tokens
    }

    old_vocab_size = len(base_tokenizer)
    base_tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(base_tokenizer))

    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    tied = (output_emb is not None
            and output_emb.weight.data_ptr() == input_emb.weight.data_ptr())

    with torch.no_grad():
        for tok, sub_ids in old_decompositions.items():
            new_id = base_tokenizer.convert_tokens_to_ids(tok)
            sub_ids = [sid for sid in sub_ids if sid < old_vocab_size]
            if new_id is None or new_id < old_vocab_size or not sub_ids:
                continue
            sub_ids_t = torch.tensor(sub_ids, device=input_emb.weight.device)
            input_emb.weight[new_id] = input_emb.weight[sub_ids_t].mean(dim=0)
            if output_emb is not None and not tied:
                output_emb.weight[new_id] = output_emb.weight[sub_ids_t].mean(dim=0)

    print(f"  ✓ Vocab extended to {len(base_tokenizer)} tokens, "
          f"new rows smart-initialized.")

    model = apply_lora_and_unfreeze_embeddings(model)

    # Load fine-tune corpus (capped for compute budget -- see config).
    train_file = SPLIT_DIR / lang / "train.txt"
    with open(train_file, "r", encoding="utf-8") as f:
        train_text = f.read(FINETUNE_MAX_TRAIN_CHARS)

    save_dir = FINETUNE_DIR / lang
    model = finetune(model, base_tokenizer, train_text, device, save_dir)

    return model, base_tokenizer


def load_finetuned_devaware_model(lang: str, device: str = None, merge_lora: bool = True):
    """
    Load a previously fine-tuned (Stage 2d) model + tokenizer for use as
    Stage 3c's 'your tokenizer' condition. Raises FileNotFoundError if
    Stage 2d hasn't been run for this language yet.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = FINETUNE_DIR / lang / "final"
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"No Stage 2d checkpoint found at {ckpt_dir}. "
            f"Run: python -m pipeline.stage2d_vocab_extend --lang {lang}"
        )

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        INDIC_LLM_MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, ckpt_dir)

    if merge_lora:
        model = model.merge_and_unload()  # fold LoRA weights into base for faster inference

    model = model.to(device).eval()
    return model, tokenizer


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2d: extend Airavata's vocab with DevAware merges + LoRA fine-tune"
    )
    parser.add_argument("--lang", choices=LANGUAGES + ["all"], default="all")
    args = parser.parse_args()

    langs = LANGUAGES if args.lang == "all" else [args.lang]

    for lang in langs:
        print(f"\n{'='*60}")
        print(f"  STAGE 2d: VOCAB EXTENSION + FINE-TUNE — {lang.upper()}")
        print(f"{'='*60}")
        build_vocab_extended_model(lang)

    print("\n✓ Stage 2d (Vocabulary Extension + Fine-tune) complete.")


if __name__ == "__main__":
    main()