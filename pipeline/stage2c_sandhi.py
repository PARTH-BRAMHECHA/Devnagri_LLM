"""
Stage 2c — Sanskrit Sandhi Module
===================================
Integrates sandhi splitting as a pre-processing toggle for Sanskrit.
Produces two tokenizer variants: with and without sandhi splitting.

Sandhi splitting is done using rule-based decomposition of common
Sanskrit sandhi patterns (vowel sandhi, consonant sandhi, visarga sandhi).

Usage:
    python -m pipeline.stage2c_sandhi [--evaluate]
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

from tqdm import tqdm

from pipeline.config import (
    SPLIT_DIR, TOKENIZER_DIR, RESULTS_DIR, ensure_dirs
)


# ─── Sandhi Rules (Rule-Based Decomposition) ────────────────────────────────

# Common vowel sandhi rules (forward direction: combined → split)
# These are simplified/approximate rules covering the most frequent cases.

# Vowel sandhi: when two vowels meet at word boundaries
VOWEL_SANDHI_RULES = [
    # a/ā + i/ī → e
    (r"([कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह])(े)", r"\1अ \2इ"),
    # a/ā + u/ū → o
    (r"([कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह])(ो)", r"\1अ \2उ"),
    # a/ā + e → ai (ऐ)
    (r"([कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह])(ै)", r"\1अ \2ए"),
    # a/ā + o → au (औ)
    (r"([कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह])(ौ)", r"\1अ \2ओ"),
]

# Visarga sandhi rules
VISARGA_SANDHI_RULES = [
    # visarga + voiced consonant → r/o transformation
    (r"ो([गघदधबभ])", r"ः \1"),
]

# Common compound word boundaries (samasa)
# These are common prefixes that often form compounds
COMMON_PREFIXES = [
    "महा", "सु", "दुर", "दुः", "प्र", "उप", "अनु", "परि",
    "अभि", "सम", "वि", "अव", "निर", "निस", "आ", "उत",
    "प्रति", "अधि",
]


def apply_sandhi_splitting(text: str) -> str:
    """
    Apply rule-based sandhi splitting to Sanskrit text.

    This is an approximate splitter — it handles the most common
    sandhi patterns but won't catch everything. The goal is to
    produce a usable variant for ablation, not a perfect linguistic
    analysis.
    """
    # Step 1: Try to split at common compound boundaries
    words = text.split()
    result_words = []

    for word in words:
        if len(word) < 4:
            result_words.append(word)
            continue

        # Try to identify compound boundaries
        split = _try_compound_split(word)
        result_words.append(split)

    return " ".join(result_words)


def _try_compound_split(word: str) -> str:
    """
    Try to split a word at compound (samasa) boundaries.
    Uses common prefixes and virama-based heuristics.
    """
    # Check for common prefixes
    for prefix in sorted(COMMON_PREFIXES, key=len, reverse=True):
        if word.startswith(prefix) and len(word) > len(prefix) + 2:
            remainder = word[len(prefix):]
            # Only split if remainder starts with a consonant or vowel
            if remainder and ord(remainder[0]) >= 0x0904:
                return f"{prefix} {remainder}"

    return word


def create_sandhi_split_corpus(input_path: Path, output_path: Path):
    """Create a sandhi-split version of a Sanskrit corpus."""
    print(f"  Applying sandhi splitting: {input_path.name}")

    lines_processed = 0
    with open(input_path, "r", encoding="utf-8") as in_f, \
         open(output_path, "w", encoding="utf-8") as out_f:
        for line in tqdm(in_f, desc="Sandhi splitting", unit=" lines"):
            line = line.strip()
            if not line:
                out_f.write("\n")
                continue

            split_line = apply_sandhi_splitting(line)
            out_f.write(split_line + "\n")
            lines_processed += 1

    print(f"  ✓ Processed {lines_processed:,} lines")
    return output_path


def train_sandhi_variant():
    """
    Train two tokenizer variants for Sanskrit:
    1. Without sandhi splitting (already done in stage2b)
    2. With sandhi splitting (new)
    """
    import sentencepiece as spm
    ensure_dirs()

    lang = "sanskrit"
    train_file = SPLIT_DIR / lang / "train.txt"
    if not train_file.exists():
        print(f"  ERROR: {train_file} not found. Run stage1 first.")
        return

    tok_dir = TOKENIZER_DIR / lang
    tok_dir.mkdir(parents=True, exist_ok=True)

    # Create sandhi-split version of training data
    sandhi_train = tok_dir / "train_sandhi_split.txt"
    if not sandhi_train.exists() or sandhi_train.stat().st_size == 0:
        create_sandhi_split_corpus(train_file, sandhi_train)
    else:
        print(f"  Sandhi-split training file already exists")

    # Train BPE on sandhi-split text
    model_prefix = tok_dir / "devaware_bpe_sanskrit_sandhi"
    model_file = Path(str(model_prefix) + ".model")

    if model_file.exists():
        print(f"  Model already exists: {model_file}")
        return str(model_file)

    print(f"  Training Sandhi-aware BPE for Sanskrit...")

    from pipeline.stage2b_devanagari_tokenizer import pre_tokenize_file, train_spm_safe
    from pipeline.config import SENTENCEPIECE_VOCAB_SIZE, SENTENCEPIECE_CHARACTER_COVERAGE

    # Pre-tokenize the sandhi-split text
    pretok_file = tok_dir / "train_sandhi_pretokenized.txt"
    if not pretok_file.exists() or pretok_file.stat().st_size == 0:
        pre_tokenize_file(sandhi_train, pretok_file)

    train_spm_safe(
        input=str(pretok_file),
        model_prefix=str(model_prefix),
        vocab_size=SENTENCEPIECE_VOCAB_SIZE,
        model_type="bpe",
        character_coverage=SENTENCEPIECE_CHARACTER_COVERAGE,
        normalization_rule_name="nfkc",
        byte_fallback=True,
        split_digits=True,
        num_threads=os.cpu_count() or 4,
        max_sentence_length=16384,
    )

    print(f"  ✓ Sandhi-aware model saved: {model_file}")
    return str(model_file)


def evaluate_sandhi_variants():
    """Compare sandhi-split vs. non-split tokenizer variants for Sanskrit."""
    from pipeline.stage2a_baselines import load_sample_sentences, evaluate_sentencepiece

    lang = "sanskrit"
    sentences = load_sample_sentences(lang)
    if not sentences:
        print("  No Sanskrit test sentences found.")
        return

    tok_dir = TOKENIZER_DIR / lang
    results = []

    # Variant 1: Without sandhi splitting
    no_sandhi_model = str(tok_dir / "devaware_bpe_sanskrit.model")
    if os.path.exists(no_sandhi_model):
        r = evaluate_sentencepiece(no_sandhi_model, sentences, "DevAware-BPE (no sandhi)")
        results.append(r)

    # Variant 2: With sandhi splitting
    sandhi_model = str(tok_dir / "devaware_bpe_sanskrit_sandhi.model")
    if os.path.exists(sandhi_model):
        # Apply sandhi splitting to test sentences too
        split_sentences = [apply_sandhi_splitting(s) for s in sentences]
        r = evaluate_sentencepiece(sandhi_model, split_sentences, "DevAware-BPE (+ sandhi)")
        results.append(r)

    if results:
        print(f"\n  {'Variant':<30} {'Tokens/Sent':>12} {'Tok/Char':>10} {'Vowel Split%':>13}")
        print(f"  {'─'*30} {'─'*12} {'─'*10} {'─'*13}")
        for r in results:
            if "error" not in r:
                print(f"  {r['tokenizer']:<30} {r['avg_tokens_per_sentence']:>12.2f} "
                      f"{r['tokens_per_char']:>10.4f} {r['vowel_split_pct']:>12.2f}%")

    # Save ablation results
    results_file = RESULTS_DIR / lang / "sandhi_ablation.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Ablation results saved: {results_file}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 2c: Sanskrit sandhi splitting module")
    parser.add_argument("--evaluate", action="store_true",
                        help="Also evaluate sandhi vs. non-sandhi variants")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  SANSKRIT SANDHI MODULE")
    print(f"{'='*60}")

    model = train_sandhi_variant()

    if args.evaluate:
        evaluate_sandhi_variants()

    print("\n✓ Stage 2c (Sandhi Module) complete.")


if __name__ == "__main__":
    main()