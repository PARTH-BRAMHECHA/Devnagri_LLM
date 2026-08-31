"""
Multi-seed Stage 6 downstream eval.

Reuses the real embedding/model-loading code from pipeline.stage6_downstream
so this isn't a re-implementation -- it just loops Stage 6 over several
(Stage 2d) seeds, adds a cheap C-sweep on cached features (no re-embedding),
and reports a bootstrap CI + a CI-overlap flag so a DevAware-vs-default
delta can't be over-read from a single noisy run.

Drop this in as pipeline/stage6_multiseed.py.

Usage:
    python -m pipeline.stage6_multiseed --lang hindi --seeds 0 1 2
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from pipeline.config import RESULTS_DIR, LANGUAGES, INDIC_LLM_MODEL, ensure_dirs
from pipeline.stage2d_vocab_extend import load_finetuned_devaware_model
from pipeline.stage6_downstream import (
    XNLI_LANG_CODE, DOWNSTREAM_TRAIN_CAP, DOWNSTREAM_EVAL_CAP, _build_features,
)

C_VALUES = [0.01, 0.1, 1.0, 10.0]
VAL_FRACTION = 0.2  # held-out slice of the train set, used only to pick C


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


def _fit_and_score(X_tr, y_tr, X_te, y_te, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    probe = LogisticRegression(max_iter=2000, C=C)
    probe.fit(X_tr, y_tr)
    preds = probe.predict(X_te)
    return accuracy_score(y_te, preds), f1_score(y_te, preds, average="macro")


def _pick_best_C(X_train, y_train, C_values):
    """Split off a validation slice from TRAIN (never touches eval) to
    choose C, avoiding leakage into the reported eval score."""
    n_val = max(1, int(len(X_train) * VAL_FRACTION))
    X_tr, y_tr = X_train[n_val:], y_train[n_val:]
    X_val, y_val = X_train[:n_val], y_train[:n_val]
    scores = {}
    for C in C_values:
        acc, _ = _fit_and_score(X_tr, y_tr, X_val, y_val, C)
        scores[C] = acc
    best_C = max(scores, key=scores.get)
    return best_C, scores


def run_one_seed(lang, lang_code, seed, device, n_train, n_eval, pooling="last"):
    from datasets import load_dataset

    ds = load_dataset("facebook/xnli", lang_code)
    train_examples = ds["train"].shuffle(seed=seed).select(range(min(n_train, len(ds["train"]))))
    eval_examples = ds["validation"].shuffle(seed=seed).select(range(min(n_eval, len(ds["validation"]))))

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


def run_multiseed(lang, seeds, device=None, n_train=DOWNSTREAM_TRAIN_CAP,
                   n_eval=DOWNSTREAM_EVAL_CAP, pooling="last"):
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
        seed_result = run_one_seed(lang, lang_code, seed, device, n_train, n_eval, pooling)
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
        "seeds": seeds,
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
    parser = argparse.ArgumentParser(description="Stage 6, multi-seed: downstream XNLI probe with CI")
    parser.add_argument("--lang", choices=LANGUAGES, default="hindi")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-train", type=int, default=DOWNSTREAM_TRAIN_CAP)
    parser.add_argument("--n-eval", type=int, default=DOWNSTREAM_EVAL_CAP)
    parser.add_argument("--pooling", choices=["last", "mean"], default="last")
    args = parser.parse_args()

    run_multiseed(args.lang, args.seeds, n_train=args.n_train,
                  n_eval=args.n_eval, pooling=args.pooling)


if __name__ == "__main__":
    main()
