"""
Stage 2b — Devanagari-Aware Tokenizer
======================================
Builds a grapheme-cluster-protected BPE tokenizer that treats
consonant+matra+virama as a single pre-tokenization unit.

This is the core innovation: instead of letting BPE split across
grapheme cluster boundaries, we protect aksharas (orthographic syllables)
as atomic units during the pre-tokenization step -- while still letting
BPE freely *merge multiple whole aksharas* into a single token, exactly
like it merges plain characters in an ordinary corpus.

How protection works
---------------------
Every multi-codepoint akshara (e.g. a consonant+matra or a conjunct such
as क्ष) that appears in the training corpus is temporarily replaced by a
single placeholder character drawn from the Unicode Private Use Area
(U+E000-U+F8FF) before SentencePiece ever sees the text. Because each
placeholder is a single, indivisible "character", SentencePiece BPE can
never split *inside* an akshara, but it is completely free to merge a
placeholder with its neighbours -- exactly the behaviour we want.

After training, the placeholders are expanded back to their original
grapheme strings directly inside the trained `.model` file (see
`_expand_placeholders_in_model`), so the final model works exactly like
any other SentencePiece model: it can be loaded with
`spm.SentencePieceProcessor().Load(...)` and called on raw, un-modified
text everywhere else in the pipeline (stage2c, eval_utils, stage3, ...).

NOTE ON A PREVIOUS BUG
-----------------------
An earlier version of this module joined akshara segments with literal
space characters before training. SentencePiece treats whitespace as a
hard word boundary (no BPE merge is ever allowed to cross it), so that
approach accidentally prevented the tokenizer from ever combining more
than one akshara into a token -- it degenerated into (worse than)
character-level tokenization and used *more* tokens per sentence than
the plain baseline BPE, not fewer. The placeholder-symbol approach above
fixes this while still guaranteeing grapheme-cluster protection.

Usage:
    python -m pipeline.stage2b_devanagari_tokenizer [--lang hindi|marathi|sanskrit|all]
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
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

# ─── Placeholder symbols used to protect multi-codepoint aksharas ──────────
# Private Use Area: guaranteed not to collide with any real corpus text.
PUA_START = 0xE000
PUA_END = 0xF8FF
PUA_BUDGET = PUA_END - PUA_START + 1  # 6,400 placeholder codepoints


def segment_into_aksharas(text: str) -> list:
    """
    Segment Devanagari text into aksharas (orthographic syllables).

    - Consonant + virama + consonant sequences stay together
    - Consonant + matra stays together
    - Independent vowels are standalone units
    - Everything else (spaces, numbers, punctuation, other scripts) is
      emitted one character at a time.
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

        # Any other character (including spaces): emit as-is.
        segments.append(ch)
        i += 1

    return segments


def collect_akshara_frequencies(input_path: Path) -> Counter:
    """
    Scan a corpus and count how often each *multi-codepoint* akshara
    (the only kind that actually needs protecting -- a single codepoint
    has nothing internal to split) occurs.
    """
    freq = Counter()
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scanning aksharas", unit=" lines"):
            line = line.strip()
            if not line:
                continue
            for seg in segment_into_aksharas(line):
                if len(seg) > 1:
                    freq[seg] += 1
    return freq


def build_symbol_map(freq: Counter, budget: int = PUA_BUDGET) -> dict:
    """
    Assign a placeholder Private-Use-Area character to the most frequent
    multi-codepoint aksharas, up to `budget` distinct symbols. Aksharas
    that don't fit in the budget are left unprotected (rare long-tail
    conjuncts fall back to ordinary, unprotected BPE behaviour).
    """
    most_common = [akshara for akshara, _ in freq.most_common(budget)]
    if len(freq) > budget:
        print(f"  NOTE: {len(freq):,} distinct multi-character akshara clusters "
              f"found but only {budget:,} placeholder symbols are available; "
              f"the {len(freq) - budget:,} rarest clusters will be left "
              f"unprotected (same as plain BPE for those).")
    return {akshara: chr(PUA_START + i) for i, akshara in enumerate(most_common)}


def remap_file(input_path: Path, output_path: Path, symbol_map: dict):
    """
    Rewrite a corpus file, replacing every protected akshara with its
    single placeholder character. Unprotected segments (single-codepoint
    aksharas, spaces, punctuation, digits, ...) pass through unchanged, so
    the output is a lossless, length-preserving-per-akshara re-encoding of
    the input -- real spaces stay real spaces (SentencePiece still learns
    normal word boundaries), and the only thing that changes is that each
    protected akshara now looks like one atomic "character" to BPE.
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
            remapped = "".join(symbol_map.get(seg, seg) for seg in segments)
            out_f.write(remapped + "\n")
            lines_processed += 1

    print(f"  ✓ Pre-tokenized {lines_processed:,} lines")
    return output_path


def _expand_placeholders_in_model(model_path: Path, symbol_map: dict):
    """
    Post-process a trained SentencePiece model, expanding every placeholder
    character back into the original akshara string it stood for. This is
    what lets the final .model be used directly on raw, unmodified text by
    every other stage in the pipeline (no special wrapper needed).

    Safe because the placeholder substitution done in `remap_file` was a
    1:1, position-preserving replacement of whole aksharas: expanding every
    placeholder occurrence inside a learned piece reconstructs exactly the
    substring of the *original* raw text that piece corresponds to. Piece
    IDs, order, and scores are untouched, so the merge structure the model
    learned is fully preserved.
    """
    try:
        from sentencepiece import sentencepiece_model_pb2 as spm_pb2
    except ImportError as e:
        raise RuntimeError(
            "Expanding the Devanagari-aware model requires the 'protobuf' "
            "package (bundled with sentencepiece_model_pb2). "
            "Install it with: pip install protobuf"
        ) from e

    reverse_map = {placeholder: akshara for akshara, placeholder in symbol_map.items()}

    m = spm_pb2.ModelProto()
    with open(model_path, "rb") as f:
        m.ParseFromString(f.read())

    def expand(piece_text: str) -> str:
        if not piece_text or not any(ch in reverse_map for ch in piece_text):
            return piece_text
        return "".join(reverse_map.get(ch, ch) for ch in piece_text)

    for piece in m.pieces:
        piece.piece = expand(piece.piece)

    with open(model_path, "wb") as f:
        f.write(m.SerializeToString())


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
    # Keep the (very verbose, per-BPE-merge) SentencePiece training log
    # down to warnings/errors unless the caller explicitly overrides it.
    kwargs.setdefault("minloglevel", 1)

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
    1. Scan the corpus and build a placeholder-symbol map for protected aksharas
    2. Remap the training corpus using those placeholders
    3. Train SentencePiece BPE on the remapped text
    4. Expand the placeholders back to real text inside the trained model
    """
    ensure_dirs()

    train_file = SPLIT_DIR / lang / "train.txt"
    if not train_file.exists():
        print(f"  ERROR: {train_file} not found. Run stage1 first.")
        return None

    tok_dir = TOKENIZER_DIR / lang
    tok_dir.mkdir(parents=True, exist_ok=True)

    model_prefix = tok_dir / f"devaware_bpe_{lang}"
    model_file = Path(str(model_prefix) + ".model")

    if model_file.exists():
        print(f"  Model already exists: {model_file}")
        return str(model_file)

    # Step 1: build the akshara placeholder map from the training corpus
    symbol_map_file = tok_dir / "akshara_symbol_map.json"
    if symbol_map_file.exists():
        with open(symbol_map_file, "r", encoding="utf-8") as f:
            symbol_map = json.load(f)
    else:
        print(f"  Scanning aksharas to build the protected-cluster symbol map...")
        freq = collect_akshara_frequencies(train_file)
        symbol_map = build_symbol_map(freq)
        with open(symbol_map_file, "w", encoding="utf-8") as f:
            json.dump(symbol_map, f, ensure_ascii=False, indent=2)
        print(f"  Protecting {len(symbol_map):,} distinct akshara clusters "
              f"(of {len(freq):,} multi-character clusters found)")

    # Step 2: remap the training corpus using placeholder symbols
    pretok_file = tok_dir / "train_pretokenized.txt"
    if not pretok_file.exists() or pretok_file.stat().st_size == 0:
        remap_file(train_file, pretok_file, symbol_map)
    else:
        print(f"  Pre-tokenized file already exists: {pretok_file}")

    # Step 3: train SentencePiece BPE on the remapped text
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
        # We've already pre-segmented aksharas ourselves; don't let
        # SentencePiece additionally force hard splits at Unicode-script
        # boundaries (e.g. between a placeholder symbol and an adjacent
        # plain Devanagari character), or it would block exactly the
        # cross-akshara merges we're trying to enable.
        split_by_unicode_script=False,
        num_threads=os.cpu_count() or 4,
        max_sentence_length=16384,
    )

    # Step 4: expand placeholders back to real text inside the trained model
    _expand_placeholders_in_model(model_file, symbol_map)

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

    # Evaluate our tokenizer (the trained model has placeholders already
    # expanded, so it can be applied directly to raw sentences)
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
