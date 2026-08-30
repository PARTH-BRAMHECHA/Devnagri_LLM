"""
Human Evaluation — Blinded Generation Comparison
==================================================
Stage 4's `llm_generated` BPC (1.7627 in the last Hindi run) is explicitly
caveated as measuring the model finding its own output predictable, NOT
generation quality -- so generation quality has never actually been
checked by a person. This script produces a blinded, randomized-order
rating sheet so a human rater can judge fluency/coherence across the
default vs DevAware tokenizer conditions, using the SAME fine-tuned model
weights for both (so any quality difference is attributable to the
tokenizer, not to a different amount of fine-tuning).

Workflow:
    1. python -m pipeline.human_eval --lang hindi --n-samples 20
       -> writes results/hindi/human_eval_sheet.csv (blinded, shuffled)
          and results/hindi/human_eval_key.json (answer key -- don't open
          until after rating, or blinding is defeated)
    2. Open the CSV, fill in fluency_1to5 / coherence_1to5 /
       would_pass_as_human_written_yn / notes for each row.
    3. python -m pipeline.human_eval --lang hindi --aggregate
       -> writes results/hindi/human_eval_results.json
"""

import argparse
import csv
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pipeline.config import RESULTS_DIR, LANGUAGES, INDIC_LLM_MODEL, SEED, ensure_dirs

GEN_CHARS_PER_SAMPLE = 400  # short enough for a rater to read quickly per row
N_SAMPLES_DEFAULT = 20


def _compressor_like(model, tokenizer, device: str):
    """generate_llm_text() (stage4_benchmark) only needs .model/.tokenizer/
    .device off its `compressor` argument -- this avoids constructing a
    full LLMCompressor (which also builds an arithmetic coder etc. that
    generation doesn't need)."""
    return SimpleNamespace(model=model, tokenizer=tokenizer, device=device)


def generate_paired_samples(lang: str, n_samples: int, seed: int = None,
                             device: str = None) -> list:
    from pipeline.stage2d_vocab_extend import load_finetuned_devaware_model
    from pipeline.stage4_benchmark import generate_llm_text
    from transformers import AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_seed = seed if seed is not None else SEED

    print(f"  Loading fine-tuned model for {lang} (seed={run_seed})...")
    model, devaware_tokenizer = load_finetuned_devaware_model(
        lang, device=device, merge_lora=True, seed=seed
    )
    default_tokenizer = AutoTokenizer.from_pretrained(INDIC_LLM_MODEL, trust_remote_code=True)

    cache_root = RESULTS_DIR / lang / "human_eval_cache"
    samples = []
    for i in range(n_samples):
        for cond_name, tok in [("devaware", devaware_tokenizer), ("default", default_tokenizer)]:
            print(f"  Generating sample {i + 1}/{n_samples} [{cond_name}]...")
            comp = _compressor_like(model, tok, device)
            text = generate_llm_text(
                comp, lang, n_chars=GEN_CHARS_PER_SAMPLE,
                cache_dir=cache_root / f"{cond_name}_{i}",
                force_regenerate=True,  # each rating-sheet sample should be fresh
            )
            samples.append({
                "sample_id": i, "condition": cond_name, "text": text[:GEN_CHARS_PER_SAMPLE],
            })

    # Blind + shuffle: the rater sees a random letter (A/B) per sample_id,
    # never the condition name, and rows are shuffled so a sample's two
    # variants aren't adjacent (avoids anchoring one rating off the other).
    rng = random.Random(run_seed)
    sheet_rows = []
    for i in range(n_samples):
        pair = [s for s in samples if s["sample_id"] == i]
        rng.shuffle(pair)
        letters = ["A", "B"]
        rng.shuffle(letters)
        for letter, s in zip(letters, pair):
            sheet_rows.append({
                "sample_id": i,
                "blind_id": f"{i}-{letter}",
                "_condition": s["condition"],  # kept only in the key, not the rater's CSV
                "text": s["text"],
            })
    rng.shuffle(sheet_rows)
    return sheet_rows


def write_rating_sheet(lang: str, rows: list):
    out_dir = RESULTS_DIR / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    key_path = out_dir / "human_eval_key.json"
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    csv_path = out_dir / "human_eval_sheet.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "blind_id", "text", "fluency_1to5", "coherence_1to5",
            "would_pass_as_human_written_yn", "notes",
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "blind_id": r["blind_id"], "text": r["text"],
                "fluency_1to5": "", "coherence_1to5": "",
                "would_pass_as_human_written_yn": "", "notes": "",
            })

    print(f"\n  ✓ Rating sheet ({len(rows)} blinded samples): {csv_path}")
    print(f"  ✓ Answer key -- do NOT open until after rating: {key_path}")
    print(f"\n  Fill in fluency_1to5 / coherence_1to5 (1=worst, 5=best) and "
          f"would_pass_as_human_written_yn for every row, then run:")
    print(f"    python -m pipeline.human_eval --lang {lang} --aggregate")


def aggregate_ratings(lang: str) -> dict:
    out_dir = RESULTS_DIR / lang
    key_path, csv_path = out_dir / "human_eval_key.json", out_dir / "human_eval_sheet.csv"
    if not key_path.exists() or not csv_path.exists():
        raise FileNotFoundError(
            f"No rating sheet found for {lang}. Run "
            f"`python -m pipeline.human_eval --lang {lang}` first."
        )

    with open(key_path, "r", encoding="utf-8") as f:
        key_rows = {r["blind_id"]: r["_condition"] for r in json.load(f)}

    ratings = {"devaware": [], "default": []}
    n_unrated = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cond = key_rows.get(row["blind_id"])
            if cond is None:
                continue
            if not row.get("fluency_1to5", "").strip():
                n_unrated += 1
                continue
            ratings[cond].append({
                "fluency": float(row["fluency_1to5"]),
                "coherence": float(row["coherence_1to5"]),
                "passed_as_human": row["would_pass_as_human_written_yn"].strip().lower() in ("y", "yes"),
            })

    summary = {}
    for cond, items in ratings.items():
        if not items:
            summary[cond] = {"error": "no rated rows for this condition yet"}
            continue
        summary[cond] = {
            "n": len(items),
            "fluency_mean": round(float(np.mean([x["fluency"] for x in items])), 3),
            "coherence_mean": round(float(np.mean([x["coherence"] for x in items])), 3),
            "pass_rate": round(float(np.mean([x["passed_as_human"] for x in items])), 3),
        }
    summary["_meta"] = {"n_unrated_rows_skipped": n_unrated}

    out_path = out_dir / "human_eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if n_unrated:
        print(f"\n  Note: {n_unrated} row(s) still unrated -- re-run --aggregate "
              f"after finishing the sheet for a complete summary.")
    print(f"\n  ✓ Saved: {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Blinded human eval of generated text")
    parser.add_argument("--lang", choices=LANGUAGES, default="hindi")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES_DEFAULT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--aggregate", action="store_true",
                         help="Aggregate an already-filled-in rating sheet "
                              "instead of generating a new one")
    args = parser.parse_args()

    ensure_dirs()
    if args.aggregate:
        aggregate_ratings(args.lang)
    else:
        rows = generate_paired_samples(args.lang, args.n_samples, seed=args.seed)
        write_rating_sheet(args.lang, rows)


if __name__ == "__main__":
    main()
