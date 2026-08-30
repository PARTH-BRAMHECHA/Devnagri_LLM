"""
Stage 6 — Downstream Task Evaluation
=====================================
BPC/compression ratio (Stage 3/4) is an INTRINSIC metric: it shows the
DevAware tokenizer makes the LLM's probability model of the text better,
but says nothing about whether that improvement is usable for anything.
This stage adds one downstream task so a BPC win can be checked against
real task performance instead of standing alone.

Task: XNLI (3-way natural language inference), Hindi subset -- the
standard, widely-used labeled Hindi classification benchmark on the HF
Hub, so there is no new data-collection step.

Method: a FROZEN-representation linear probe (sklearn LogisticRegression
on mean-pooled last-hidden-state features), not end-to-end fine-tuning.
This is deliberate: the question is whether the two tokenizer conditions'
existing representations (after Stage 2d) differ in usefulness, not
whether more gradient steps can close a gap. Both conditions reuse the
SAME fine-tuned model weights -- only the tokenizer changes -- so a probe
accuracy difference isolates a tokenizer effect, mirroring how Stage 3
isolates the tokenizer effect from the fine-tuning effect for BPC.

Usage:
    python -m pipeline.stage6_downstream --lang hindi
    python -m pipeline.stage6_downstream --lang hindi --seed 2   # a second Stage 2d seed
    python -m pipeline.stage6_downstream --lang hindi --n-train 2000 --n-eval 500  # faster/smaller
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pipeline.config import (
    RESULTS_DIR, LANGUAGES, INDIC_LLM_MODEL, SEED, RESULTS_FILE_SUFFIX, ensure_dirs
)
from pipeline.error_diagnostics import run_with_diagnostics

# XNLI only ships a Hindi subset -- Marathi/Sanskrit have no equivalent
# labeled NLI set on the Hub as of writing. Extend this map if/when one
# becomes available rather than silently reusing Hindi's.
XNLI_LANG_CODE = {"hindi": "hi"}

DOWNSTREAM_TRAIN_CAP = 4000  # examples used to fit the probe (speed knob)
DOWNSTREAM_EVAL_CAP = 1000   # examples used to score it
MAX_SEQ_LEN = 128


def _results_filename(suffix: str = "") -> str:
    """Mirrors stage3_compress._results_filename -- keeps multi-seed runs
    from overwriting each other's downstream_results.json."""
    suffix = suffix or RESULTS_FILE_SUFFIX
    return f"downstream_results_{suffix}.json" if suffix else "downstream_results.json"


@torch.no_grad()
def _embed(model, tokenizer, premise: str, hypothesis: str, device: str) -> np.ndarray:
    """Mean-pool the final hidden state over premise+hypothesis, encoded as
    a single sequence (standard NLI-as-one-sequence framing)."""
    sep = tokenizer.sep_token or tokenizer.eos_token or "[SEP]"
    text = f"{premise} {sep} {hypothesis}"
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                     max_length=MAX_SEQ_LEN).to(device)
    out = model(**enc, output_hidden_states=True)
    last_hidden = out.hidden_states[-1][0]  # (seq_len, hidden)
    return last_hidden.mean(dim=0).float().cpu().numpy()


def _build_features(model, tokenizer, examples, device: str, desc: str = ""):
    feats, labels = [], []
    for ex in tqdm(examples, desc=desc):
        feats.append(_embed(model, tokenizer, ex["premise"], ex["hypothesis"], device))
        labels.append(ex["label"])
    return np.stack(feats), np.array(labels)


def run_downstream(lang: str, seed: int = None, device: str = None,
                    n_train: int = DOWNSTREAM_TRAIN_CAP, n_eval: int = DOWNSTREAM_EVAL_CAP):
    if lang not in XNLI_LANG_CODE:
        print(f"  ⚠ No downstream task wired up for '{lang}' yet -- XNLI only "
              f"covers {list(XNLI_LANG_CODE)}. Skipping.")
        return None

    # Imported lazily so `python -m pipeline.stage6_downstream --help` and
    # other stages don't pay for these (heavier, GPU-loading) imports.
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoTokenizer
    from pipeline.stage2d_vocab_extend import load_finetuned_devaware_model

    ensure_dirs()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lang_code = XNLI_LANG_CODE[lang]
    run_seed = seed if seed is not None else SEED

    print(f"\n{'='*70}")
    print(f"  STAGE 6: DOWNSTREAM TASK EVAL ({lang}, XNLI-{lang_code})")
    print(f"{'='*70}")

    ds = load_dataset("facebook/xnli", lang_code)
    train_examples = ds["train"].shuffle(seed=run_seed).select(
        range(min(n_train, len(ds["train"])))
    )
    eval_examples = ds["validation"].shuffle(seed=run_seed).select(
        range(min(n_eval, len(ds["validation"])))
    )
    print(f"  Train probe on {len(train_examples)} examples, "
          f"eval on {len(eval_examples)}.")

    print("  Loading fine-tuned model (shared across both tokenizer conditions "
          "-- only the tokenizer differs between the two probe runs)...")
    model, devaware_tokenizer = load_finetuned_devaware_model(
        lang, device=device, merge_lora=True, seed=seed
    )
    default_tokenizer = AutoTokenizer.from_pretrained(INDIC_LLM_MODEL, trust_remote_code=True)

    results = {}
    for cond_name, tok in [
        ("devaware_tokenizer", devaware_tokenizer),
        ("default_tokenizer", default_tokenizer),
    ]:
        print(f"\n  --- Condition: {cond_name} ---")
        t0 = time.time()
        X_train, y_train = _build_features(
            model, tok, train_examples, device, desc=f"  embed train ({cond_name})"
        )
        X_eval, y_eval = _build_features(
            model, tok, eval_examples, device, desc=f"  embed eval ({cond_name})"
        )

        probe = LogisticRegression(max_iter=2000)
        probe.fit(X_train, y_train)
        preds = probe.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="macro")

        results[cond_name] = {
            "task": f"xnli-{lang_code}",
            "n_train": len(train_examples),
            "n_eval": len(eval_examples),
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1), 4),
            "elapsed_s": round(time.time() - t0, 1),
        }
        print(f"    accuracy={acc:.4f}  macro_f1={f1:.4f}")

    dev_acc = results["devaware_tokenizer"]["accuracy"]
    def_acc = results["default_tokenizer"]["accuracy"]
    results["_meta"] = {
        "model": f"{INDIC_LLM_MODEL} (vocab-extended, LoRA fine-tuned)",
        "seed": run_seed,
        "probe": "sklearn LogisticRegression on mean-pooled frozen last-hidden-state",
        "accuracy_delta_devaware_minus_default": round(dev_acc - def_acc, 4),
        "note": (
            "Same fine-tuned model weights for both conditions -- only the "
            "tokenizer changes. A single run at one seed; treat like the BPC "
            "numbers in compression_results.json and re-run at 2-3 seeds "
            "(config.SEED workflow) before reporting this delta as a stable "
            "effect rather than noise."
        ),
    }

    out_path = RESULTS_DIR / lang / _results_filename()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Downstream results saved: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 6: downstream task eval (XNLI probe)")
    parser.add_argument("--lang", choices=LANGUAGES + ["all"], default="hindi")
    parser.add_argument("--seed", type=int, default=None,
                         help="Load a specific Stage 2d seed's checkpoint "
                              "(default: config.SEED)")
    parser.add_argument("--n-train", type=int, default=DOWNSTREAM_TRAIN_CAP)
    parser.add_argument("--n-eval", type=int, default=DOWNSTREAM_EVAL_CAP)
    args = parser.parse_args()

    langs = LANGUAGES if args.lang == "all" else [args.lang]
    for lang in langs:
        try:
            run_with_diagnostics(
                run_downstream, lang, seed=args.seed, n_train=args.n_train, n_eval=args.n_eval
            )
        except FileNotFoundError as e:
            print(f"  ⚠ Skipping {lang}: {e}")


if __name__ == "__main__":
    main()
