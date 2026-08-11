"""
Stage 2b — Devanagari-Aware Tokenizer
======================================
Builds a grapheme-cluster-protected BPE tokenizer that treats
consonant+matra+virama as a single pre-tokenization unit.

This is the core innovation: instead of letting BPE split across
grapheme cluster boundaries, we protect aksharas (orthographic syllables)
as atomic units during the pre-tokenization step.

Usage:
    python -m pipeline.stage2b_devanagari_tokenizer [--lang hindi|marathi|sanskrit|all]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import regex
import sentencepiece as spm
from tqdm import tqdm

from pipeline.config import (
    SPLIT_DIR, TOKENIZER_DIR, LANGUAGES, RESULTS_DIR,
    SENTENCEPIECE_VOCAB_SIZE, SENTENCEPIECE_CHARACTER_COVERAGE,
    ensure_dirs
)


# ─── Devanagari Unicode Constants ────────────────────────────────────────────

# Consonants: क-ह (0x0915-0x0939)
CONSONANT_RANGE = (0x0915, 0x0939)

# Dependent vowel signs (matras): ा-ौ (0x093E-0x094C)
MATRA_RANGE = (0x093E, 0x094C)

# Virama (halant): ् (0x094D)
VIRAMA = chr(0x094D)

# Nukta: ़ (0x093C)
NUKTA = chr(0x093C)

# Anusvara: ं (0x0902), Chandrabindu: ँ (0x0901), Visarga: ः (0x0903)
ANUSVARA = chr(0x0902)
CHANDRABINDU = chr(0x0901)
VISARGA = chr(0x0903)

# Full pattern for a Devanagari akshara (orthographic syllable):
#   consonant [nukta] [virama consonant [nukta]]* [matra] [anusvara|chandrabindu|visarga]
# This captures conjuncts like क्ष, ज्ञ, etc.
AKSHARA_PATTERN = regex.compile(
    r"[\u0915-\u0939][\u093C]?"           # Base consonant [+ nukta]
    r"(?:\u094D[\u0915-\u0939][\u093C]?)*" # Zero or more virama+consonant (conjuncts)
    r"[\u093E-\u094C]?"                    # Optional matra
    r"[\u0901-\u0903]?"                    # Optional anusvara/chandrabindu/visarga
)

# Independent vowels: अ-ऋ
VOWEL_PATTERN = regex.compile(r"[\u0904-\u0914][\u0901-\u0903]?")

# Full grapheme cluster (Unicode standard)
GRAPHEME_CLUSTER = regex.compile(r"\X")


def segment_into_aksharas(text: str) -> list:
    """
    Segment Devanagari text into aksharas (orthographic syllables).

    This protects grapheme clusters from being split by BPE:
    - Consonant + virama + consonant sequences stay together
    - Consonant + matra stays together
    - Independent vowels are standalone units
    - Non-Devanagari characters (spaces, numbers, punctuation) are separate tokens
    """
    segments = []
    i = 0

    while i < len(text):
        ch = text[i]
        cp = ord(ch)

        # Try to match a full akshara starting here
        if CONSONANT_RANGE[0] <= cp <= CONSONANT_RANGE[1]:
            m = AKSHARA_PATTERN.match(text, i)
            if m:
                segments.append(m.group())
                i = m.end()
                continue

        # Independent vowel
        if 0x0904 <= cp <= 0x0914:
            m = VOWEL_PATTERN.match(text, i)
            if m:
                segments.append(m.group())
                i = m.end()
                continue

        # Space → keep as boundary marker
        if ch == " ":
            segments.append("▁")  # SentencePiece space marker
            i += 1
            continue

        # Any other character: emit as-is
        segments.append(ch)
        i += 1

    return segments


def pre_tokenize_file(input_path: Path, output_path: Path):
    """
    Pre-tokenize a corpus file by segmenting Devanagari text into
    akshara-protected units separated by spaces.

    SentencePiece will then learn BPE merges on top of these pre-segmented
    units, ensuring it never splits across grapheme cluster boundaries.
    """
    print(f"  Pre-tokenizing: {input_path.name}")

    lines_processed = 0
    with open(input_path, "r", encoding="utf-8") as in_f, \
         open(output_path, "w", encoding="utf-8") as out_f:
        for line in tqdm(in_f, desc="Pre-tokenizing", unit=" lines"):
            line = line.strip()
            if not line:
                out_f.write("\n")
                continue

            segments = segment_into_aksharas(line)
            # Join segments with spaces so SentencePiece treats each as a potential atom
            out_f.write(" ".join(segments) + "\n")
            lines_processed += 1

    print(f"  ✓ Pre-tokenized {lines_processed:,} lines")
    return output_path


def train_spm_safe(**kwargs):
    """
    Train a SentencePiece model, automatically retrying with a reduced
    vocab_size if the corpus can't support the requested one.

    This happens whenever the number of unique symbols available to
    SentencePiece is smaller than `vocab_size` — e.g. on smaller corpora
    (Sanskrit especially) or after akshara pre-tokenization collapses the
    surface symbol space. SentencePiece reports the max supported size in
    its error message ("...Please set it to a value <= N."); we parse that
    and retry once instead of crashing the pipeline.
    """
    try:
        spm.SentencePieceTrainer.train(**kwargs)
    except RuntimeError as e:
        msg = str(e)
        m = re.search(r"<=\s*(\d+)", msg)
        if not m:
            raise
        max_vocab = int(m.group(1))
        requested = kwargs.get("vocab_size")
        print(f"  ⚠ vocab_size={requested} unsupported for this corpus "
              f"(max={max_vocab}). Retrying with vocab_size={max_vocab}.")
        kwargs = dict(kwargs)
        kwargs["vocab_size"] = max_vocab
        spm.SentencePieceTrainer.train(**kwargs)


def train_devanagari_aware_bpe(lang: str):
    """
    Train a Devanagari-aware BPE tokenizer:
    1. Pre-tokenize the training corpus into aksharas
    2. Train SentencePiece BPE on the pre-tokenized text
    """
    ensure_dirs()

    train_file = SPLIT_DIR / lang / "train.txt"
    if not train_file.exists():
        print(f"  ERROR: {train_file} not found. Run stage1 first.")
        return None

    tok_dir = TOKENIZER_DIR / lang
    tok_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Pre-tokenize
    pretok_file = tok_dir / "train_pretokenized.txt"
    if not pretok_file.exists() or pretok_file.stat().st_size == 0:
        pre_tokenize_file(train_file, pretok_file)
    else:
        print(f"  Pre-tokenized file already exists: {pretok_file}")

    # Step 2: Train SentencePiece BPE on pre-tokenized text
    model_prefix = tok_dir / f"devaware_bpe_{lang}"
    model_file = Path(str(model_prefix) + ".model")

    if model_file.exists():
        print(f"  Model already exists: {model_file}")
        return str(model_file)

    print(f"  Training Devanagari-aware BPE for {lang}...")

    train_spm_safe(
        input=str(pretok_file),
        model_prefix=str(model_prefix),
        vocab_size=SENTENCEPIECE_VOCAB_SIZE,
        model_type="bpe",
        character_coverage=SENTENCEPIECE_CHARACTER_COVERAGE,
        normalization_rule_name="nfkc",
        byte_fallback=True,
        split_digits=True,
        # Keep pre-segmented units as starting atoms
        treat_whitespace_as_suffix=False,
        num_threads=os.cpu_count() or 4,
        max_sentence_length=16384,
    )

    print(f"  ✓ Model saved: {model_file}")
    return str(model_file)


def evaluate_devanagari_tokenizer(model_path: str, lang: str):
    """Compare our Devanagari-aware tokenizer against baseline BPE."""
    from pipeline.stage2a_baselines import (
        load_sample_sentences, evaluate_sentencepiece, DEPENDENT_VOWELS
    )

    sentences = load_sample_sentences(lang)
    if not sentences:
        return

    # Evaluate our tokenizer
    print(f"\n  Evaluating Devanagari-aware BPE...")
    our_result = evaluate_sentencepiece(model_path, sentences, f"DevAware-BPE-{lang}")

    # Load baseline BPE for comparison
    baseline_model = str(TOKENIZER_DIR / lang / f"sp_bpe_{lang}.model")
    baseline_result = None
    if os.path.exists(baseline_model):
        print(f"  Evaluating baseline BPE...")
        baseline_result = evaluate_sentencepiece(baseline_model, sentences, f"SP-BPE-{lang}")

    # Print comparison
    print(f"\n  {'Metric':<25} {'DevAware BPE':>15} {'Baseline BPE':>15} {'Improvement':>12}")
    print(f"  {'─'*25} {'─'*15} {'─'*15} {'─'*12}")

    metrics = ["avg_tokens_per_sentence", "tokens_per_char", "vowel_split_pct"]
    labels = ["Tokens/Sentence", "Tokens/Character", "Vowel Split %"]

    for metric, label in zip(metrics, labels):
        our_val = our_result.get(metric, 0)
        if baseline_result and "error" not in baseline_result:
            bl_val = baseline_result.get(metric, 0)
            if bl_val > 0:
                improvement = ((bl_val - our_val) / bl_val) * 100
                print(f"  {label:<25} {our_val:>15.2f} {bl_val:>15.2f} {improvement:>11.1f}%")
            else:
                print(f"  {label:<25} {our_val:>15.2f} {bl_val:>15.2f} {'N/A':>12}")
        else:
            print(f"  {label:<25} {our_val:>15.2f} {'N/A':>15}")

    # Save results
    results = {
        "devanagari_aware": our_result,
        "baseline_bpe": baseline_result,
    }
    results_file = RESULTS_DIR / lang / "devanagari_tokenizer_comparison.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Results saved: {results_file}")

    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2b: Train Devanagari-aware tokenizer with grapheme cluster protection"
    )
    parser.add_argument("--lang", choices=LANGUAGES + ["all"], default="all",
                        help="Language to process (default: all)")
    args = parser.parse_args()

    langs = LANGUAGES if args.lang == "all" else [args.lang]

    for lang in langs:
        print(f"\n{'='*60}")
        print(f"  DEVANAGARI-AWARE TOKENIZER: {lang.upper()}")
        print(f"{'='*60}")

        model_path = train_devanagari_aware_bpe(lang)
        if model_path:
            evaluate_devanagari_tokenizer(model_path, lang)

    print("\n✓ Stage 2b (Devanagari-Aware Tokenizer) complete.")


if __name__ == "__main__":
    main()