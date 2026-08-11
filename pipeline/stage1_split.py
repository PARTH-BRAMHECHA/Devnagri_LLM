"""
Stage 1c — Train/Test Splitting
================================
Document-level 95/5 split to avoid leakage.
Documents are defined as paragraphs separated by blank lines.

Usage:
    python -m pipeline.stage1_split [--lang hindi|marathi|sanskrit|all]
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from pipeline.config import (
    CLEAN_DIR, SPLIT_DIR, LANGUAGES, TRAIN_RATIO, CORPUS_TARGETS, ensure_dirs
)


def split_corpus(lang: str, seed: int = 42):
    """
    Document-level train/test split.

    A "document" = contiguous block of non-empty lines (paragraph).
    Split at document boundaries to avoid sentence-level leakage.
    """
    ensure_dirs()

    input_file = CLEAN_DIR / lang / "corpus_clean.txt"
    if not input_file.exists():
        print(f"  ERROR: Cleaned corpus not found at {input_file}")
        print(f"  Run stage1_clean first!")
        return

    input_size = input_file.stat().st_size
    target_size = CORPUS_TARGETS[lang]

    print(f"  Input: {input_file} ({input_size / (1024**2):.1f} MB)")
    print(f"  Target corpus size: {target_size / (1024**2):.1f} MB")
    print(f"  Split ratio: {TRAIN_RATIO:.0%} train / {1-TRAIN_RATIO:.0%} test")

    # Read and group into documents
    print("  Grouping lines into documents...")
    documents = []
    current_doc = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                current_doc.append(line)
            else:
                if current_doc:
                    documents.append("\n".join(current_doc))
                    current_doc = []
        if current_doc:
            documents.append("\n".join(current_doc))

    print(f"  Total documents: {len(documents):,}")

    # Trim to target size if needed
    total_bytes = sum(len(d.encode("utf-8")) for d in documents)
    if total_bytes > target_size:
        print(f"  Corpus ({total_bytes / (1024**2):.1f} MB) exceeds target ({target_size / (1024**2):.1f} MB)")
        print(f"  Trimming to target size...")

        # Shuffle first (so we get a representative sample)
        rng = random.Random(seed)
        rng.shuffle(documents)

        # Greedily pack shuffled documents up to the target size. Skip
        # (don't break on) any doc that doesn't fit in the remaining
        # budget — with real-sized documents there are plenty of
        # smaller ones later in the shuffle that still fit, whereas
        # breaking on the first oversized doc (the original behavior)
        # can leave the budget almost entirely unused, or even yield
        # zero documents if the very first shuffled doc is large.
        trimmed = []
        trimmed_bytes = 0
        for doc in documents:
            doc_bytes = len(doc.encode("utf-8"))
            if trimmed_bytes + doc_bytes > target_size:
                continue
            trimmed.append(doc)
            trimmed_bytes += doc_bytes

        documents = trimmed
        print(f"  Trimmed to {len(documents):,} documents ({trimmed_bytes / (1024**2):.1f} MB)")
    else:
        print(f"  Corpus size ({total_bytes / (1024**2):.1f} MB) is within target.")

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(documents)

    split_idx = int(len(documents) * TRAIN_RATIO)
    train_docs = documents[:split_idx]
    test_docs = documents[split_idx:]

    # Write splits
    train_dir = SPLIT_DIR / lang
    train_dir.mkdir(parents=True, exist_ok=True)

    train_file = train_dir / "train.txt"
    test_file = train_dir / "test.txt"

    train_bytes = 0
    with open(train_file, "w", encoding="utf-8") as f:
        for doc in train_docs:
            f.write(doc + "\n\n")
            train_bytes += len(doc.encode("utf-8")) + 2

    test_bytes = 0
    with open(test_file, "w", encoding="utf-8") as f:
        for doc in test_docs:
            f.write(doc + "\n\n")
            test_bytes += len(doc.encode("utf-8")) + 2

    # Report
    print(f"\n  Split results:")
    print(f"    Train: {len(train_docs):,} docs, {train_bytes / (1024**2):.1f} MB → {train_file}")
    print(f"    Test:  {len(test_docs):,} docs, {test_bytes / (1024**2):.1f} MB → {test_file}")

    # Save stats
    stats = {
        "language": lang,
        "total_documents": len(documents),
        "train_documents": len(train_docs),
        "test_documents": len(test_docs),
        "train_size_mb": round(train_bytes / (1024**2), 2),
        "test_size_mb": round(test_bytes / (1024**2), 2),
        "train_ratio": TRAIN_RATIO,
        "seed": seed,
    }
    stats_file = train_dir / "split_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Stats saved to: {stats_file}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 1c: Document-level train/test split")
    parser.add_argument("--lang", choices=LANGUAGES + ["all"], default="all",
                        help="Language to split (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    langs = LANGUAGES if args.lang == "all" else [args.lang]

    for lang in langs:
        print(f"\n{'='*60}")
        print(f"  SPLITTING: {lang.upper()}")
        print(f"{'='*60}")
        split_corpus(lang, seed=args.seed)

    print("\n✓ Stage 1c (Splitting) complete.")


if __name__ == "__main__":
    main()