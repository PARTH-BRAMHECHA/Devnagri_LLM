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
on last-token-pooled hidden-state features), not end-to-end fine-tuning.
This is deliberate: the question is whether the two tokenizer conditions'
existing representations (after Stage 2d) differ in usefulness, not
whether more gradient steps can close a gap. Both conditions reuse the
SAME fine-tuned model weights -- only the tokenizer changes -- so a probe
accuracy difference isolates a tokenizer effect, mirroring how Stage 3
isolates the tokenizer effect from the fine-tuning effect for BPC.

Two run modes, same file, same module path:
  --seed  (int, default: config.SEED)   Single run, exactly the original
      behaviour. Fast smoke-test. Saved to downstream_results.json.
  --seeds (one or more ints)            Multi-seed run: for EACH seed,
      reloads that seed's Stage 2d checkpoint, embeds train/eval once,
      picks C on a held-out slice of the TRAIN set (never touches eval),
      then aggregates accuracy/macro-F1 across seeds with a bootstrap
      95% CI and an explicit CI-overlap flag. Saved to
      downstream_results_multiseed.json. Use this before trusting any
      DevAware-vs-default delta -- a single seed's delta is noise-sized.

Usage:
    python -m pipeline.stage6_downstream --lang hindi
    python -m pipeline.stage6_downstream --lang hindi --seed 2
    python -m pipeline.stage6_downstream --lang hindi --n-train 2000 --n-eval 500
    python -m pipeline.stage6_downstream --lang hindi --seeds 0 1 2
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
C_VALUES = [0.01, 0.1, 1.0, 10.0]   # sweep used only in --seeds mode
VAL_FRACTION = 0.2                  # held-out slice of TRAIN, used only to pick C


def _results_filename(suffix: str = "") -> str:
    """Mirrors stage3_compress._results_filename -- keeps multi-seed runs
    from overwriting each other's downstream_results.json."""
    suffix = suffix or RESULTS_FILE_SUFFIX
    return f"downstream_results_{suffix}.json" if suffix else "downstream_results.json"


@torch.no_grad()
def _embed(model, tokenizer, premise: str, hypothesis: str, device: str,
           pooling: str = "last") -> np.ndarray:
    """Encode premise+hypothesis as a single sequence and pool the final
    hidden state into one vector.

    `pooling`:
      - "last" (default): the last non-padded token's hidden state. This
        is the standard choice for a DECODER-ONLY (causal) model -- only
        the final token's hidden state has attended to the full sequence;
        every earlier token's state was computed under a causal mask that
        blocks it from seeing anything after it, so those earlier states
        carry an incomplete (and for a premise-then-hypothesis sequence,
        premise-ONLY) view of the pair being classified.
      - "mean": mean-pool over all tokens (the original implementation).
        This is a bidirectional-encoder (BERT-style) convention that
        implicitly assumes every token's representation already reflects
        the whole sequence -- not true here, and averaging in a large
        block of premise-only states is closer to diluting the signal
        with noise than aggregating complementary views of it. Kept as an
        option only to A/B against "last", not as a recommended default.
    """
    sep = tokenizer.sep_token or tokenizer.eos_token or "[SEP]"
    text = f"{premise} {sep} {hypothesis}"
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                     max_length=MAX_SEQ_LEN).to(device)
    out = model(**enc, output_hidden_states=True)
    last_hidden = out.hidden_states[-1][0]  # (seq_len, hidden)
    if pooling == "last":
        return last_hidden[-1].float().cpu().numpy()
    return last_hidden.mean(dim=0).float().cpu().numpy()


def _build_features(model, tokenizer, examples, device: str, desc: str = "",
                     pooling: str = "last"):
    feats, labels = [], []
    for ex in tqdm(examples, desc=desc):
        feats.append(_embed(model, tokenizer, ex["premise"], ex["hypothesis"], device, pooling))
        labels.append(ex["label"])
    return np.stack(feats), np.array(labels)


def _fit_and_score(X_tr, y_tr, X_te, y_te, C: float = 1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    probe = LogisticRegression(max_iter=2000, C=C)
    probe.fit(X_tr, y_tr)
    preds = probe.predict(X_te)
    return accuracy_score(y_te, preds), f1_score(y_te, preds, average="macro")


def _pick_best_C(X_train, y_train, C_values):
    """Split off a validation slice from TRAIN (never touches eval) to
    choose C, so C-selection can't leak into the reported eval score."""
    n_val = max(1, int(len(X_train) * VAL_FRACTION))
    X_tr, y_tr = X_train[n_val:], y_train[n_val:]
    X_val, y_val = X_train[:n_val], y_train[:n_val]
    scores = {}
    for C in C_values:
        acc, _ = _fit_and_score(X_tr, y_tr, X_val, y_val, C)
        scores[C] = acc
    best_C = max(scores, key=scores.get)
    return best_C, scores


def bootstrap_ci(values, n_boot=2000, alpha=0.05, rng_seed=0):
    if len(values) < 2:
        v = float(values[0])
        return v, v
    rng = np.random.default_rng(rng_seed)
    values = np.asarray(values, dtype=float)
    boots = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _load_xnli_split(lang_code, seed, n_train, n_eval):
    from datasets import load_dataset
    ds = load_dataset("facebook/xnli", lang_code)
    train_examples = ds["train"].shuffle(seed=seed).select(range(min(n_train, len(ds["train"]))))
    eval_examples = ds["validation"].shuffle(seed=seed).select(range(min(n_eval, len(ds["validation"]))))
    return train_examples, eval_examples


# ---------------------------------------------------------------------------
# Single-seed run (original behaviour, unchanged output format/filename)
# ---------------------------------------------------------------------------

def run_downstream(lang: str, seed: int = None, device: str = None,
                    n_train: int = DOWNSTREAM_TRAIN_CAP, n_eval: int = DOWNSTREAM_EVAL_CAP,
                    pooling: str = "last"):
    if lang not in XNLI_LANG_CODE:
        print(f"  ⚠ No downstream task wired up for '{lang}' yet -- XNLI only "
              f"covers {list(XNLI_LANG_CODE)}. Skipping.")
        return None

    from transformers import AutoTokenizer
    from pipeline.stage2d_vocab_extend import load_finetuned_devaware_model

    ensure_dirs()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lang_code = XNLI_LANG_CODE[lang]
    run_seed = seed if seed is not None else SEED

    print(f"\n{'='*70}")
    print(f"  STAGE 6: DOWNSTREAM TASK EVAL ({lang}, XNLI-{lang_code}, pooling={pooling})")
    print(f"{'='*70}")

    train_examples, eval_examples = _load_xnli_split(lang_code, run_seed, n_train, n_eval)
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
            model, tok, train_examples, device, desc=f"  embed train ({cond_name})",
            pooling=pooling
        )
        X_eval, y_eval = _build_features(
            model, tok, eval_examples, device, desc=f"  embed eval ({cond_name})",
            pooling=pooling
        )

        acc, f1 = _fit_and_score(X_train, y_train, X_eval, y_eval, C=1.0)

        results[cond_name] = {
            "task": f"xnli-{lang_code}",
            "pooling": pooling,
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
        "pooling": pooling,
        "probe": f"sklearn LogisticRegression on {pooling}-pooled frozen last-hidden-state",
        "accuracy_delta_devaware_minus_default": round(dev_acc - def_acc, 4),
        "note": (
            "Same fine-tuned model weights for both conditions -- only the "
            "tokenizer changes. A single run at one seed; treat like the BPC "
            "numbers in compression_results.json and re-run with --seeds "
            "0 1 2 (this same script) before reporting this delta as a "
            "stable effect rather than noise. If accuracy is still near the "
            "33% chance baseline for 3-way XNLI even with pooling='last', "
            "that's evidence the frozen-representation probe isn't "
            "capturing the task -- not that the two tokenizer conditions "
            "are equivalent."
        ),
    }

    out_path = RESULTS_DIR / lang / _results_filename()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Downstream results saved: {out_path}")
    return results


# ---------------------------------------------------------------------------
# Multi-seed run (new): reruns per seed, adds a cheap C-sweep on cached
# features, aggregates with a bootstrap CI, and flags CI overlap.
# ---------------------------------------------------------------------------

def _run_one_seed(lang, lang_code, seed, device, n_train, n_eval, pooling):
    from transformers import AutoTokenizer
    from pipeline.stage2d_vocab_extend import load_finetuned_devaware_model

    train_examples, eval_examples = _load_xnli_split(lang_code, seed, n_train, n_eval)

    print(f"\n  [seed={seed}] loading fine-tuned checkpoint...")
    model, devaware_tokenizer = load_finetuned_devaware_model(
        lang, device=device, merge_lora=True, seed=seed
    )
    default_tokenizer = AutoTokenizer.from_pretrained(INDIC_LLM_MODEL, trust_remote_code=True)

    seed_results = {}
    for cond_name, tok in [
        ("devaware_tokenizer", devaware_tokenizer),
        ("default_tokenizer", default_tokenizer),
    ]:
        t0 = time.time()
        X_train, y_train = _build_features(
            model, tok, train_examples, device, desc=f"  [seed={seed}] embed train ({cond_name})",
            pooling=pooling,
        )
        X_eval, y_eval = _build_features(
            model, tok, eval_examples, device, desc=f"  [seed={seed}] embed eval ({cond_name})",
            pooling=pooling,
        )

        best_C, sweep_scores = _pick_best_C(X_train, y_train, C_VALUES)
        acc, f1 = _fit_and_score(X_train, y_train, X_eval, y_eval, best_C)

        seed_results[cond_name] = {
            "seed": seed,
            "best_C": best_C,
            "C_sweep_val_acc": sweep_scores,
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1), 4),
            "elapsed_s": round(time.time() - t0, 1),
        }
        print(f"    [seed={seed}] {cond_name}: best_C={best_C} "
              f"acc={acc:.4f} macro_f1={f1:.4f}")

    del model
    torch.cuda.empty_cache()
    return seed_results


def run_downstream_multiseed(lang: str, seeds, device: str = None,
                              n_train: int = DOWNSTREAM_TRAIN_CAP,
                              n_eval: int = DOWNSTREAM_EVAL_CAP,
                              pooling: str = "last"):
    if lang not in XNLI_LANG_CODE:
        print(f"  ⚠ No downstream task wired up for '{lang}' yet. Skipping.")
        return None

    ensure_dirs()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lang_code = XNLI_LANG_CODE[lang]

    print(f"\n{'='*70}")
    print(f"  STAGE 6 (multi-seed): {lang}, XNLI-{lang_code}, "
          f"pooling={pooling}, seeds={seeds}")
    print(f"{'='*70}")

    per_seed = {"devaware_tokenizer": [], "default_tokenizer": []}
    for seed in seeds:
        seed_result = _run_one_seed(lang, lang_code, seed, device, n_train, n_eval, pooling)
        for cond in per_seed:
            per_seed[cond].append(seed_result[cond])

    results = {}
    for cond in per_seed:
        accs = [r["accuracy"] for r in per_seed[cond]]
        f1s = [r["macro_f1"] for r in per_seed[cond]]
        results[cond] = {
            "pooling": pooling,
            "n_train": n_train,
            "n_eval": n_eval,
            "per_seed": per_seed[cond],
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "accuracy_ci95": bootstrap_ci(accs),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
            "macro_f1_ci95": bootstrap_ci(f1s),
        }

    dev_ci = results["devaware_tokenizer"]["accuracy_ci95"]
    def_ci = results["default_tokenizer"]["accuracy_ci95"]
    ci_overlap = not (dev_ci[1] < def_ci[0] or def_ci[1] < dev_ci[0])
    delta = results["devaware_tokenizer"]["accuracy_mean"] - results["default_tokenizer"]["accuracy_mean"]

    results["_meta"] = {
        "seeds": list(seeds),
        "accuracy_delta_devaware_minus_default": round(float(delta), 4),
        "ci95_overlap": ci_overlap,
        "note": (
            "CIs computed via bootstrap over per-seed accuracies. If "
            "ci95_overlap is True, the delta is not distinguishable from "
            "noise at this seed count -- do not report a winner. If "
            "accuracy_mean is still near the 33% chance baseline, that's "
            "still a probe-capacity/task-fit problem, not a tokenizer effect."
        ),
    }

    out_path = RESULTS_DIR / lang / "downstream_results_multiseed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ Saved: {out_path}")
    print(f"  delta={delta:+.4f}  CI overlap={ci_overlap}")
    print(f"  DevAware: {results['devaware_tokenizer']['accuracy_mean']:.4f} "
          f"CI{dev_ci}  |  Default: {results['default_tokenizer']['accuracy_mean']:.4f} CI{def_ci}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 6: downstream task eval (XNLI probe)")
    parser.add_argument("--lang", choices=LANGUAGES + ["all"], default="hindi")
    parser.add_argument("--seed", type=int, default=None,
                         help="Single-run mode: load a specific Stage 2d seed's "
                              "checkpoint (default: config.SEED). Mutually "
                              "exclusive with --seeds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Multi-seed mode: run each seed, aggregate with "
                              "a bootstrap 95%% CI and a C-sweep. Overrides "
                              "--seed if both are given.")
    parser.add_argument("--n-train", type=int, default=DOWNSTREAM_TRAIN_CAP)
    parser.add_argument("--n-eval", type=int, default=DOWNSTREAM_EVAL_CAP)
    parser.add_argument("--pooling", choices=["last", "mean"], default="last",
                         help="How to pool the final hidden state into one "
                              "probe feature vector. 'last' (default) is the "
                              "standard choice for a decoder-only/causal "
                              "model -- only the last token has attended to "
                              "the full premise+hypothesis sequence. 'mean' "
                              "is the original (BERT-style) implementation, "
                              "kept for A/B comparison.")
    args = parser.parse_args()

    langs = LANGUAGES if args.lang == "all" else [args.lang]
    for lang in langs:
        try:
            if args.seeds:
                run_with_diagnostics(
                    run_downstream_multiseed, lang, seeds=args.seeds,
                    n_train=args.n_train, n_eval=args.n_eval, pooling=args.pooling
                )
            else:
                run_with_diagnostics(
                    run_downstream, lang, seed=args.seed, n_train=args.n_train,
                    n_eval=args.n_eval, pooling=args.pooling
                )
        except FileNotFoundError as e:
            print(f"  ⚠ Skipping {lang}: {e}")


if __name__ == "__main__":
    main()
