"""
Stage 2b — Devanagari-Aware Tokenizer
======================================
Builds a grapheme-cluster-protected BPE tokenizer that treats
consonant+matra+virama as a single pre-tokenization unit.

This is the core innovation: instead of letting BPE split across
grapheme cluster boundaries, we protect aksharas (orthographic syllables)
as atomic units during the pre-tokenization step -- while still letting
BPE merge multiple adjacent aksharas into a single subword unit, which is
the whole point of training a custom tokenizer instead of just using
character-level splitting.

FIX (see pipeline/pua_remap.py for the full rationale): aksharas are no
longer joined with literal spaces before SentencePiece training. Spaces
were being treated as hard token boundaries by SentencePiece's
split_by_whitespace=True, which prevented BPE from ever merging two
aksharas together -- even within the same word -- and made the
Devanagari-aware tokenizer strictly worse than the plain baseline BPE
(0.502 vs 0.254 tokens/char). Each distinct akshara is now collapsed to a
single Private-Use-Area code point instead, so it stays atomic (can't be
split internally) but can still be merged with its neighbors by BPE. Real
word-boundary spaces are left untouched.

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
from pipeline.pua_remap import AksharaPUAMap


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

# Internal marker used during segmentation to stand for a real space, before
# it gets converted to a literal " " by the PUA remap step.
SPACE_MARKER = "▁"


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
            segments.append(SPACE_MARKER)
            i += 1
            continue

        # Any other character: emit as-is
        segments.append(ch)
        i += 1

    return segments


def _iter_corpus_lines(input_path: Path):
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            yield line.strip()


def build_pua_map(input_path: Path) -> AksharaPUAMap:
    """
    First pass over the training corpus: collect every distinct akshara
    string that actually occurs, and assign each one a PUA code point.
    """
    print(f"  Scanning aksharas in {input_path.name} to build PUA map...")
    aksharas = set()
    for line in tqdm(_iter_corpus_lines(input_path), desc="Scanning", unit=" lines"):
        if not line:
            continue
        for seg in segment_into_aksharas(line):
            if seg == SPACE_MARKER:
                continue
            # Only akshara-shaped segments need a PUA slot; single
            # passthrough characters (punctuation, digits, Latin) are left
            # as-is by remap_segments and don't need mapping.
            if len(seg) > 1 or (0x0900 <= ord(seg) <= 0x097F):
                aksharas.add(seg)

    pua_map = AksharaPUAMap()
    pua_map.build(aksharas)
    print(f"  ✓ Mapped {len(aksharas):,} distinct aksharas to PUA code points")
    return pua_map


def pre_tokenize_file(input_path: Path, output_path: Path, pua_map: AksharaPUAMap):
    """
    Pre-tokenize a corpus file by segmenting Devanagari text into aksharas
    and remapping each one to its PUA code point (see pua_remap.py). Real
    spaces are preserved as real spaces; only akshara-to-akshara boundaries
    lose their separator, which is what allows BPE to merge across them.
    """
    print(f"  Pre-tokenizing (PUA remap): {input_path.name}")

    lines_processed = 0
    with open(input_path, "r", encoding="utf-8") as in_f, \
         open(output_path, "w", encoding="utf-8") as out_f:
        for line in tqdm(in_f, desc="Pre-tokenizing", unit=" lines"):
            line = line.strip()
            if not line:
                out_f.write("\n")
                continue

            segments = segment_into_aksharas(line)
            remapped = pua_map.remap_segments(segments, space_marker=SPACE_MARKER)
            out_f.write(remapped + "\n")
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
    1. Build (or load) the akshara -> PUA code point map for this corpus
    2. Pre-tokenize the training corpus by remapping aksharas to PUA chars
    3. Train SentencePiece BPE on the PUA-remapped text
    """
    ensure_dirs()

    train_file = SPLIT_DIR / lang / "train.txt"
    if not train_file.exists():
        print(f"  ERROR: {train_file} not found. Run stage1 first.")
        return None

    tok_dir = TOKENIZER_DIR / lang
    tok_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build or load the PUA map (needed again later for decoding /
    # evaluation, so we always persist it rather than keeping it in memory
    # only).
    pua_map_path = tok_dir / f"akshara_pua_map_{lang}.json"
    if AksharaPUAMap.exists(pua_map_path):
        print(f"  Loading existing PUA map: {pua_map_path}")
        pua_map = AksharaPUAMap.load(pua_map_path)
    else:
        pua_map = build_pua_map(train_file)
        pua_map.save(pua_map_path)
        print(f"  ✓ PUA map saved: {pua_map_path}")

    # Step 2: Pre-tokenize (PUA remap)
    pretok_file = tok_dir / "train_pretokenized.txt"
    if not pretok_file.exists() or pretok_file.stat().st_size == 0:
        pre_tokenize_file(train_file, pretok_file, pua_map)
    else:
        print(f"  Pre-tokenized file already exists: {pretok_file}")

    # Step 3: Train SentencePiece BPE on the PUA-remapped text
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
        # "identity" (not "nfkc"): the input is already PUA-remapped, and
        # NFKC normalization behavior on Private-Use-Area code points is
        # undefined/unreliable across ICU versions -- normalizing before
        # the remap (Stage 1 cleaning) is the right place for that, not
        # here.
        normalization_rule_name="identity",
        byte_fallback=True,
        split_digits=True,
        num_threads=os.cpu_count() or 4,
        max_sentence_length=16384,
    )

    print(f"  ✓ Model saved: {model_file}")
    return str(model_file)


def evaluate_devanagari_tokenizer(model_path: str, lang: str):
    """Compare our Devanagari-aware tokenizer against baseline BPE.

    Uses DevAwareTokenizer (segmentation + PUA remap + SentencePiece) for
    "our" side, since the raw SentencePiece model only understands
    PUA-remapped text, not raw Devanagari. The baseline model is a plain
    SentencePiece model trained directly on raw text, so it's evaluated
    the normal way.
    """
    from pipeline.stage2a_baselines import load_sample_sentences, evaluate_sentencepiece
    from pipeline.devaware_tokenizer import DevAwareTokenizer

    sentences = load_sample_sentences(lang)
    if not sentences:
        return

    tok_dir = TOKENIZER_DIR / lang
    pua_map_path = tok_dir / f"akshara_pua_map_{lang}.json"

    print(f"\n  Evaluating Devanagari-aware BPE...")
    dev_tok = DevAwareTokenizer(model_path, pua_map_path)
    our_result = _evaluate_devaware(dev_tok, sentences, f"DevAware-BPE-{lang}")

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


def _evaluate_devaware(dev_tok, sentences: list, label: str) -> dict:
    """
    Same metrics as stage2a.evaluate_sentencepiece, computed through the
    DevAwareTokenizer wrapper. The vowel-split check is done on each
    piece's *decoded* Devanagari text (piece -> PUA chars -> original
    string), since pieces are now PUA characters, not literal matras.
    """
    from pipeline.stage2a_baselines import DEPENDENT_VOWELS

    total_tokens = 0
    total_chars = 0
    vowel_splits = 0
    total_vowels = 0

    for sent in sentences:
        pieces = dev_tok.encode_as_pieces(sent)
        total_tokens += len(pieces)
        total_chars += len(sent)

        for piece in pieces:
            decoded = dev_tok.piece_to_devanagari(piece)
            for ch in decoded:
                if ch in DEPENDENT_VOWELS:
                    total_vowels += 1
            clean = decoded.lstrip(" ")
            if clean and clean[0] in DEPENDENT_VOWELS:
                vowel_splits += 1

    avg_tokens = total_tokens / max(len(sentences), 1)
    split_pct = 100 * vowel_splits / max(total_vowels, 1)

    return {
        "tokenizer": label,
        "sentences": len(sentences),
        "avg_tokens_per_sentence": round(avg_tokens, 2),
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "tokens_per_char": round(total_tokens / max(total_chars, 1), 4),
        "vowel_split_pct": round(split_pct, 2),
    }


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