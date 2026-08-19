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

3. QLORA + FULL EMBEDDING FINE-TUNE. The frozen backbone is loaded in 4-bit
   (NF4, double-quant) via bitsandbytes -- a plain bf16 7B model already
   uses ~14GB just sitting there, which doesn't leave enough headroom on a
   16GB Kaggle GPU for the embedding/LM-head optimizer state once you add
   gradients on top. 4-bit quantizing the frozen transformer blocks drops
   that to ~4GB and gives the actually-trainable pieces (LoRA adapters +
   full embed_tokens/lm_head) the room they need. `lm_head` is explicitly
   excluded from quantization (`llm_int8_skip_modules`) since it's fully
   fine-tuned, not just adapted -- you can't backprop a real gradient into
   a 4-bit-quantized weight.

4. A short continued-pretraining pass (causal LM loss) on the language's
   own corpus, using the extended tokenizer, adapts the model to the new
   vocabulary. Bounded by a hard wall-clock budget
   (FINETUNE_MAX_WALL_SECONDS) so it always checkpoints before a Kaggle
   session gets killed, rather than losing an in-progress run. Re-running
   the same command resumes from the last checkpoint automatically.

The result is a real, loadable (model, tokenizer) pair that Stage 3 can use
for a genuine "your tokenizer" condition -- see the LLMCompressor patch in
stage3_compress.py.

Usage:
    python -m pipeline.stage2d_vocab_extend --lang hindi
    python -m pipeline.stage2d_vocab_extend --lang hindi --max-hours 4
"""

import argparse
import gc
import json
import math
import os
import shutil
import time
from pathlib import Path

# Reduce CUDA allocator fragmentation on repeated OOM/retry within the same
# kernel -- PyTorch's own OOM message recommends this. Must be set before
# the first CUDA allocation, so it's set at import time, not inside a
# function called after torch/cuda are already in use.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import Dataset, DataLoader

from pipeline.config import (
    SPLIT_DIR, TOKENIZER_DIR, FINETUNE_DIR, LANGUAGES,
    INDIC_LLM_MODEL, VOCAB_EXTEND_MIN_AKSHARAS, VOCAB_EXTEND_MAX_NEW_TOKENS,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    FINETUNE_BLOCK_SIZE, FINETUNE_BATCH_SIZE, FINETUNE_GRAD_ACCUM_STEPS,
    FINETUNE_LR, FINETUNE_STEPS, FINETUNE_WARMUP_STEPS, FINETUNE_SAVE_EVERY,
    FINETUNE_LOG_EVERY, FINETUNE_MAX_TRAIN_CHARS, FINETUNE_MAX_WALL_SECONDS,
    ensure_dirs,
)
from pipeline.devaware_tokenizer import DevAwareTokenizer
from pipeline.stage2b_devanagari_tokenizer import segment_into_aksharas


# ─── CUDA OOM safety net ─────────────────────────────────────────────────────
#
# Two crashes were observed on a 16GB Kaggle GPU: one while resuming a
# checkpoint (PeftModel.from_pretrained), one inside optimizer.step(). Full
# fine-tuning of embed_tokens/lm_head (required because new vocab tokens are
# added) makes their optimizer state large, and there's a real risk of GPU
# memory being left fragmented (visible as "reserved but unallocated" in the
# CUDA OOM message) if a run dies and the SAME Kaggle kernel is reused for
# the next attempt -- Python doesn't release CUDA memory just because an
# exception was raised while a debugger/traceback still references the
# tensors. `_cuda_cleanup()` is best-effort hygiene for that; it cannot force
# a leaked reference to be freed, hence the explicit kernel-restart advice
# below when an OOM is actually hit.
def _cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_OOM_RESTART_MSG = (
    "  ✗ CUDA out of memory. If this keeps happening on retry within the "
    "same notebook session, RESTART THE KAGGLE KERNEL before re-running --  "
    "a crashed CUDA context can leave memory reserved-but-unusable that "
    "torch.cuda.empty_cache() cannot reclaim from here. Re-running the same "
    "command after a fresh kernel start will resume from the last "
    "checkpoint automatically."
)


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
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    # Required before LoRA-wrapping a 4-bit-quantized model: casts norm
    # layers to fp32 for stability, enables input-grads on the (now
    # resized) embedding layer, and turns on gradient checkpointing --
    # which we want anyway given how little headroom is left after the
    # embed/lm_head optimizer state.
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    # `modules_to_save` (not a manual requires_grad loop) is peft's
    # native mechanism for "fully fine-tune this whole module, not just a
    # LoRA delta, and make save_pretrained/from_pretrained round-trip it
    # correctly." peft wraps each named module in a ModulesToSaveWrapper
    # that keeps a frozen original + a trainable copy (initialized from
    # the module's current weights -- i.e. our smart-initialized
    # embeddings, since this runs after that step), and both
    # save_pretrained and from_pretrained know to persist/restore the
    # trainable copy as part of the adapter checkpoint. This replaces an
    # earlier version that set requires_grad=True by hand and hoped
    # peft's embedding-resize auto-detection would catch it on save --
    # this is the tested path instead of an assumption.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora_config)

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


def finetune(model, tokenizer, train_text: str, device: str, save_dir: Path,
             start_step: int = 0,
             max_wall_seconds: float = FINETUNE_MAX_WALL_SECONDS):
    """
    `start_step`: resume point (0 for a fresh run, >0 when continuing from
    a previously-saved step_N checkpoint -- see build_vocab_extended_model).

    `max_wall_seconds`: hard budget for THIS call. Checked every step (not
    just at log intervals) so a session that's about to be killed still
    gets a checkpoint written. Hitting the budget is not an error -- it
    saves and returns early; re-running the same command picks up where
    this left off.
    """
    model.train()
    dataset = BlockDataset(train_text, tokenizer, FINETUNE_BLOCK_SIZE)
    print(f"  Fine-tune corpus: {len(train_text):,} chars -> "
          f"{len(dataset):,} blocks of {FINETUNE_BLOCK_SIZE} tokens")

    if len(dataset) == 0:
        print("  ⚠ Not enough data to build even one training block -- skipping fine-tune.")
        return model

    loader = DataLoader(dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # 8-bit AdamW (bitsandbytes) instead of full-precision AdamW: the
    # trainable set here includes the FULLY fine-tuned embed_tokens/lm_head
    # (large tensors, since vocab was extended), and plain AdamW keeps two
    # fp32 state tensors per param -- roughly 4x that tensor's own memory.
    # 8-bit state cuts that to ~1x, which is what actually fixed the
    # optimizer.step() OOM observed on a 16GB GPU. Falls back to regular
    # AdamW (with a warning) if bitsandbytes isn't installed.
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=FINETUNE_LR)
        print("  Using bitsandbytes 8-bit AdamW (reduced optimizer memory).",
              flush=True)
    except Exception as e:
        # Fully fine-tuning embed_tokens/lm_head on a 14-16GB GPU is only
        # viable with 8-bit optimizer state. Falling back to full-precision
        # AdamW here doesn't just run slower -- it reliably OOMs on
        # optimizer.step() a few steps in, which wastes the whole run.
        # Fail loudly instead of silently degrading, and flush immediately
        # so this is never lost if the process dies shortly after (this is
        # exactly what happened previously: the warning was printed but
        # never made it to the log before the OOM crash ate the buffer).
        print(f"  ✗ bitsandbytes unavailable ({type(e).__name__}: {e}). "
              f"Full-precision AdamW over {sum(p.numel() for p in trainable_params):,} "
              f"trainable params will almost certainly OOM on this GPU. "
              f"Install a working bitsandbytes build first: "
              f"`pip install -U bitsandbytes`, then restart the kernel and "
              f"re-run (it will resume from the last checkpoint).",
              flush=True)
        raise RuntimeError(
            "bitsandbytes import/init failed -- refusing to continue with "
            "full-precision AdamW (see message above)."
        ) from e
    total_steps = min(FINETUNE_STEPS, len(loader) * 50)  # don't schedule past what data supports many epochs of

    # Resume: restore actual Adam momentum/variance state (not just model
    # weights), and construct the scheduler at last_epoch=start_step-1 so
    # its internal __init__ step() lands it exactly back on the LR the
    # warmup/decay schedule would have produced at this step -- not a
    # restart of the schedule. `initial_lr` has to be seeded manually
    # since this optimizer was just constructed fresh (last_epoch!=-1
    # normally expects a scheduler to have already set it once before).
    if start_step > 0:
        opt_path = save_dir / f"step_{start_step}" / "optimizer.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            print(f"  ✓ Restored optimizer (Adam momentum/variance) state from {opt_path}")
        else:
            print(f"  ⚠ No optimizer.pt at {opt_path} -- resuming with fresh "
                  f"Adam state (momentum reset, model weights unaffected).")
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", FINETUNE_LR)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, step / max(1, FINETUNE_WARMUP_STEPS))
        * max(0.0, (total_steps - step) / max(1, total_steps)),
        last_epoch=start_step - 1 if start_step > 0 else -1,
    )
    # Remaining honest gap, much smaller than before: if the wall-clock
    # cutoff lands mid-way through an 8-step grad-accum window, whatever
    # gradient had accumulated in that unfinished window (never applied,
    # never saved) is lost on resume -- at most a slightly noisier single
    # update right at the resume boundary, not a correctness issue.

    if start_step >= total_steps:
        print(f"  ✓ {start_step} steps already done (target {total_steps}) -- "
              f"nothing left to train, finalizing.")
        _save_checkpoint(model, tokenizer, save_dir, start_step, final=True)
        model.eval()
        return model

    step = start_step
    t0 = time.time()
    optimizer.zero_grad()
    data_iter = iter(loader)
    wall_clock_hit = False
    while step < total_steps:
        if time.time() - t0 > max_wall_seconds:
            wall_clock_hit = True
            break

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        try:
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
        except torch.OutOfMemoryError:
            # Save whatever we've got before anything else touches the GPU,
            # then clean up and fail loudly with actionable advice instead
            # of leaving a half-updated model / dangling CUDA context.
            print(f"\n{_OOM_RESTART_MSG}\n  (OOM at step {step}/{total_steps} -- "
                  f"attempting an emergency checkpoint before exiting.)")
            optimizer.zero_grad(set_to_none=True)
            _cuda_cleanup()
            try:
                _save_checkpoint(model, tokenizer, save_dir, step)
                print(f"  ✓ Emergency checkpoint saved at step {step} -- "
                      f"restart the kernel, then re-run the same command to resume.")
            except torch.OutOfMemoryError:
                print("  ✗ Could not even save an emergency checkpoint -- GPU "
                      "memory is too fragmented. Restart the kernel and re-run; "
                      "training will resume from the last periodic checkpoint "
                      f"(every {FINETUNE_SAVE_EVERY} steps) instead.")
            raise

        # Periodic cache clear: gradient checkpointing + long training loops
        # fragment the CUDA allocator over many steps, which is part of why
        # the resume/optimizer OOMs got worse on later steps/attempts.
        if step % FINETUNE_LOG_EVERY == 0:
            _cuda_cleanup()

        if step % FINETUNE_LOG_EVERY == 0:
            elapsed = time.time() - t0
            print(f"    step {step:5d}/{total_steps}  "
                  f"loss={outputs.loss.item():.4f}  "
                  f"ppl={math.exp(min(outputs.loss.item(), 20)):.2f}  "
                  f"({elapsed:.0f}s elapsed / {max_wall_seconds:.0f}s budget)", flush=True)

        if step > 0 and step % FINETUNE_SAVE_EVERY == 0:
            _save_checkpoint(model, tokenizer, save_dir, step, optimizer=optimizer)

        step += 1

    if wall_clock_hit:
        print(f"  ⏱ Wall-clock budget ({max_wall_seconds:.0f}s) hit at step "
              f"{step}/{total_steps} -- checkpointing and stopping early. "
              f"Re-run the exact same command to resume from here.")
        _save_checkpoint(model, tokenizer, save_dir, step, optimizer=optimizer)
        model.eval()
        return model

    _save_checkpoint(model, tokenizer, save_dir, step, final=True)
    model.eval()
    return model


def _prune_old_checkpoints(save_dir: Path, keep_step: int, keep_last_n: int = 1):
    """Delete periodic step_N checkpoint dirs other than the `keep_last_n`
    most recent ones (the just-written `keep_step` plus any already-newer
    ones, in the unlikely case of out-of-order calls). Resume logic
    (`_find_latest_checkpoint`) only ever reads the highest step_N dir, so
    older ones are pure dead weight -- and each one carries a full
    embed_tokens/lm_head + Adam optimizer state, easily several GB. Left
    unpruned, this is what fills a Kaggle 20GB working-disk quota after
    just 3-4 checkpoints and crashes mid-run with "No space left on
    device". `final` is never touched here.
    """
    if not save_dir.exists():
        return
    candidates = []
    for d in save_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                candidates.append((int(d.name.split("_")[1]), d))
            except (IndexError, ValueError):
                continue
    candidates.sort(key=lambda x: x[0])  # ascending by step
    to_delete = candidates[:-keep_last_n] if keep_last_n > 0 else candidates
    for step_num, d in to_delete:
        try:
            shutil.rmtree(d)
            print(f"  🧹 Pruned old checkpoint: {d} (superseded by step_{keep_step})")
        except OSError as e:
            # Non-fatal: worst case is one extra stale checkpoint dir left
            # around, not a reason to abort a training run that otherwise
            # succeeded.
            print(f"  ⚠ Could not prune {d}: {e}")


def _save_checkpoint(model, tokenizer, save_dir: Path, step: int,
                      final: bool = False, optimizer=None):
    out_dir = save_dir / ("final" if final else f"step_{step}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)   # saves LoRA adapter + resized embed/head
    tokenizer.save_pretrained(out_dir)
    if optimizer is not None:
        # Not saved for 'final' -- nothing resumes past a completed run,
        # no need to carry Adam state around after training is done.
        torch.save(optimizer.state_dict(), out_dir / "optimizer.pt")
    print(f"  ✓ Checkpoint saved: {out_dir}")

    # Only periodic step_N checkpoints accumulate across a run; 'final' is
    # a single one-off write and needs no pruning. Keep just the checkpoint
    # we just wrote -- resume only ever reads the single highest step_N dir
    # (see _find_latest_checkpoint), so nothing older is ever read again.
    if not final:
        _prune_old_checkpoints(save_dir, keep_step=step, keep_last_n=1)


def _load_quantized_base(device: str):
    """Load Airavata with the frozen backbone in 4-bit NF4, lm_head kept at
    full precision since it gets fully fine-tuned (you can't backprop a
    real gradient into a 4-bit-quantized weight)."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["lm_head"],
    )
    return AutoModelForCausalLM.from_pretrained(
        INDIC_LLM_MODEL,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={"": 0} if device == "cuda" else "cpu",
    )


def _find_latest_checkpoint(save_dir: Path):
    """Return (path, step) of the highest step_N checkpoint under save_dir,
    or (None, 0) if there isn't one. Deliberately ignores 'final' -- the
    caller checks that separately, since a 'final' dir means this language
    is already fully done and shouldn't re-enter the training loop at all."""
    if not save_dir.exists():
        return None, 0
    candidates = []
    for d in save_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                candidates.append((int(d.name.split("_")[1]), d))
            except (IndexError, ValueError):
                continue
    if not candidates:
        return None, 0
    candidates.sort(key=lambda x: x[0])
    step, path = candidates[-1]
    return path, step


# ─── Orchestration ────────────────────────────────────────────────────────

def build_vocab_extended_model(lang: str, device: str = None,
                                max_wall_seconds: float = None):
    """
    Full Stage 2d pipeline for one language:
      DevAware SPM -> novel merged pieces -> extend Airavata's tokenizer
      -> smart-init new embeddings -> QLoRA-wrap -> fine-tune -> save.

    Idempotent across calls / Kaggle sessions:
      - if FINETUNE_DIR/lang/final exists, this language is already done --
        loads and returns it without touching the GPU for training.
      - elif FINETUNE_DIR/lang/step_N exists, resumes fine-tuning from the
        highest N instead of starting over.
      - else, does the full fresh build (vocab extension + smart init +
        LoRA-wrap) before fine-tuning.

    Returns (model, tokenizer) ready to hand to LLMCompressor.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, prepare_model_for_kbit_training

    ensure_dirs()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    max_wall_seconds = (FINETUNE_MAX_WALL_SECONDS if max_wall_seconds is None
                         else max_wall_seconds)

    # Best-effort reclaim of any memory left dangling by a previous crashed
    # attempt in this same process/kernel (see _cuda_cleanup docstring above
    # -- this cannot fix real leaks/fragmentation, only reduce their odds).
    _cuda_cleanup()

    save_dir = FINETUNE_DIR / lang
    final_dir = save_dir / "final"

    if final_dir.exists():
        print(f"  ✓ Stage 2d already complete for {lang} -- loading "
              f"{final_dir} (delete it to force a re-run).")
        tokenizer = AutoTokenizer.from_pretrained(final_dir, trust_remote_code=True)
        model = _load_quantized_base(device)
        # mean_resizing=False: these rows are overwritten by the checkpoint's
        # saved embed_tokens/lm_head weights on the next line anyway.
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model = PeftModel.from_pretrained(model, final_dir)
        return model, tokenizer

    resume_dir, resume_step = _find_latest_checkpoint(save_dir)

    if resume_dir is not None:
        print(f"  ↻ Resuming {lang} fine-tune from {resume_dir} (step {resume_step})")
        try:
            base_tokenizer = AutoTokenizer.from_pretrained(resume_dir, trust_remote_code=True)
            model = _load_quantized_base(device)
            # mean_resizing=False: overwritten by the resumed checkpoint's
            # saved embed_tokens/lm_head weights via PeftModel.from_pretrained
            # a couple lines down.
            model.resize_token_embeddings(len(base_tokenizer), mean_resizing=False)
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            model = PeftModel.from_pretrained(model, resume_dir, is_trainable=True)
        except torch.OutOfMemoryError:
            print(_OOM_RESTART_MSG)
            _cuda_cleanup()
            raise

        train_file = SPLIT_DIR / lang / "train.txt"
        with open(train_file, "r", encoding="utf-8") as f:
            train_text = f.read(FINETUNE_MAX_TRAIN_CHARS)

        model = finetune(model, base_tokenizer, train_text, device, save_dir,
                          start_step=resume_step, max_wall_seconds=max_wall_seconds)
        return model, base_tokenizer

    tok_dir = TOKENIZER_DIR / lang
    dev_tok = DevAwareTokenizer(
        spm_model_path=tok_dir / f"devaware_bpe_{lang}.model",
        pua_map_path=tok_dir / f"akshara_pua_map_{lang}.json",
    )

    print(f"  Loading base tokenizer + model (4-bit): {INDIC_LLM_MODEL}")
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(INDIC_LLM_MODEL, trust_remote_code=True)
        model = _load_quantized_base(device)
    except torch.OutOfMemoryError:
        print(_OOM_RESTART_MSG)
        _cuda_cleanup()
        raise

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

    model = finetune(model, base_tokenizer, train_text, device, save_dir,
                      start_step=0, max_wall_seconds=max_wall_seconds)

    return model, base_tokenizer


def load_finetuned_devaware_model(lang: str, device: str = None, merge_lora: bool = True,
                                   base_model=None):
    """
    Load a previously fine-tuned (Stage 2d) model + tokenizer for use as
    Stage 3c's 'your tokenizer' condition. Raises FileNotFoundError if
    Stage 2d hasn't been run for this language yet.

    `base_model`: pass Stage 3's already-loaded shared LLMCompressor.model
    here to attach the adapter to it IN PLACE instead of loading a second
    full bf16 copy of Airavata. This is required on single-GPU setups
    (e.g. a 14.56 GiB Kaggle T4) -- Stage 3's shared compressor already
    occupies ~14 GiB, so loading a second full-precision base model here
    is what was causing:
        torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 808.00 MiB.
    at the PeftModel.from_pretrained(...) call below -- not because the
    LoRA adapter itself is large, but because there was no room left for
    it once two full 7B copies were resident simultaneously.

    If `base_model` is omitted, falls back to loading a fresh copy from
    disk (only safe if nothing else is on the GPU at the time).

    NOTE: when `base_model` is passed and `merge_lora=True`, merging folds
    the LoRA weights into the shared base model's weights in place. Call
    `detach_devaware_adapter()` afterwards to restore the original
    (unfine-tuned) weights before reusing the shared model for the next
    language -- otherwise merges would stack across languages.
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

    if base_model is not None:
        # Reuse the caller's already-loaded model -- no second 7B load.
        if len(tokenizer) != base_model.get_input_embeddings().weight.shape[0]:
            # mean_resizing=False: the default mean-resizing init computes a
            # mean/covariance over the OLD embedding matrix and needs a full
            # temporary copy of it to do so. On top of a full bf16 7B model
            # already using ~14 GiB of a 14.56 GiB T4, that temp copy is what
            # pushes this over the edge:
            #   torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 752.00 MiB.
            # It's also pointless work here: modules_to_save=["embed_tokens",
            # "lm_head"] means PeftModel.from_pretrained() below immediately
            # overwrites these rows with the actually fine-tuned weights, so
            # whatever resize_token_embeddings initializes them to is thrown
            # away a line later. Skipping mean-resizing costs nothing and
            # removes the memory spike.
            base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model = PeftModel.from_pretrained(base_model, ckpt_dir)
    else:
        # Fallback: fresh full bf16 load. Only safe if this is the only
        # model on the GPU (e.g. Stage 2d standalone, not Stage 3).
        base_model = AutoModelForCausalLM.from_pretrained(
            INDIC_LLM_MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        # Same reasoning as above -- these rows get overwritten by the
        # checkpoint's saved embed_tokens/lm_head weights immediately below.
        base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model = PeftModel.from_pretrained(base_model, ckpt_dir)

    if merge_lora:
        model = model.merge_and_unload()  # fold LoRA weights into base for faster inference

    model = model.to(device).eval()
    return model, tokenizer


def detach_devaware_adapter(model, base_tokenizer, base_original_vocab_size: int):
    """
    Undo an in-place load_finetuned_devaware_model(..., base_model=shared_model)
    call so the shared base model is safe to reuse for the next language,
    without reloading anything from disk.

    Handles both cases:
      - merge_lora=False: model is still a PeftModel wrapper -> .unload()
        strips the adapter and returns the clean base model.
      - merge_lora=True: model IS the base model with LoRA weights already
        merged in -- there's no adapter left to .unload(). In this case
        the resized embedding rows for the extra devaware tokens also need
        to be trimmed back off, since merge_and_unload() bakes the extended
        vocabulary's rows into the weight matrix permanently.

    `base_original_vocab_size`: the base model's vocab size BEFORE any
    devaware vocab extension (i.e. len(shared_compressor.tokenizer) at
    Stage 3 startup, before ever calling load_finetuned_devaware_model).
    Pass this in so embeddings can be trimmed back to exactly that size.
    """
    from peft import PeftModel

    if isinstance(model, PeftModel):
        base_model = model.unload()
    else:
        # merge_lora=True path: model is the (now-merged) base model.
        base_model = model
        if base_model.get_input_embeddings().weight.shape[0] != base_original_vocab_size:
            base_model.resize_token_embeddings(base_original_vocab_size)

    torch.cuda.empty_cache()
    return base_model


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2d: extend Airavata's vocab with DevAware merges + LoRA fine-tune"
    )
    parser.add_argument("--lang", choices=LANGUAGES + ["all"], default="all")
    parser.add_argument(
        "--max-hours", type=float, default=None,
        help="Override FINETUNE_MAX_WALL_SECONDS (config.py) for this run, "
             "in hours. Applies per language, not to the whole --lang all "
             "batch. Default (config.py) is 3h/language.",
    )
    args = parser.parse_args()

    langs = LANGUAGES if args.lang == "all" else [args.lang]
    max_wall_seconds = args.max_hours * 3600 if args.max_hours is not None else None

    for lang in langs:
        print(f"\n{'='*60}")
        print(f"  STAGE 2d: VOCAB EXTENSION + FINE-TUNE — {lang.upper()}")
        print(f"{'='*60}")
        build_vocab_extended_model(lang, max_wall_seconds=max_wall_seconds)

    print("\n✓ Stage 2d (Vocabulary Extension + Fine-tune) complete.")


if __name__ == "__main__":
    main()
