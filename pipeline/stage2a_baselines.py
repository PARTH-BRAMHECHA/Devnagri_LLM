"""
Stage 2a — Baseline Tokenizers
================================
Train per-language SentencePiece BPE/Unigram tokenizers and evaluate
existing tokenizers (GPT-2, LLaMA, Gemma, IndicBERT) on the corpus.

Produces a baseline table of:
  - tokens/sentence
  - % dependent vowels split as separate tokens

Usage:
    python -m pipeline.stage2a_baselines [--lang hindi|marathi|sanskrit|all]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import sentencepiece as spm
from tqdm import tqdm

from pipeline.config import (
    SPLIT_DIR, TOKENIZER_DIR, LANGUAGES, RESULTS_DIR,
    SENTENCEPIECE_VOCAB_SIZE, SENTENCEPIECE_CHARACTER_COVERAGE,
    SENTENCEPIECE_MODEL_TYPE, BASELINE_TOKENIZERS, ensure_dirs
)
from pipeline.stage2b_devanagari_tokenizer import train_spm_safe

# Devanagari dependent vowel marks (matras)
# ा ि ी ु ू ृ ॄ ॅ ॆ े ै ॉ ॊ ो ौ
DEPENDENT_VOWELS = set(chr(c) for c in range(0x093E, 0x094D))  # matras
VIRAMA = chr(0x094D)  # ्


def load_sample_sentences(lang: str, max_sentences: int = 5000) -> list:
    """Load sample sentences from the test set for evaluation."""
    test_file = SPLIT_DIR / lang / "test.txt"
    if not test_file.exists():
        print(f"  WARNING: {test_file} not found. Using train.txt")
        test_file = SPLIT_DIR / lang / "train.txt"
    if not test_file.exists():
        print(f"  ERROR: No split data found for {lang}")
        return []

    sentences = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and len(line) > 20:
                sentences.append(line)
                if len(sentences) >= max_sentences:
                    break

    return sentences


def train_sentencepiece(lang: str, model_type: str = "bpe"):
    """Train a SentencePiece model on the train split."""
    ensure_dirs()

    train_file = SPLIT_DIR / lang / "train.txt"
    if not train_file.exists():
        print(f"  ERROR: {train_file} not found. Run stage1 first.")
        return None

    model_prefix = TOKENIZER_DIR / lang / f"sp_{model_type}_{lang}"
    model_file = Path(str(model_prefix) + ".model")

    if model_file.exists():
        print(f"  Model already exists: {model_file}")
        return str(model_file)

    print(f"  Training SentencePiece {model_type.upper()} for {lang}...")
    print(f"  Vocab size: {SENTENCEPIECE_VOCAB_SIZE}")
    print(f"  Character coverage: {SENTENCEPIECE_CHARACTER_COVERAGE}")

    train_spm_safe(
        input=str(train_file),
        model_prefix=str(model_prefix),
        vocab_size=SENTENCEPIECE_VOCAB_SIZE,
        model_type=model_type,
        character_coverage=SENTENCEPIECE_CHARACTER_COVERAGE,
        normalization_rule_name="nfkc",
        byte_fallback=True,
        split_digits=True,

        # Faster SentencePiece training
        seed_sentencepiece_size=100000,
        num_sub_iterations=1,
        num_threads=os.cpu_count() or 4,

        max_sentence_length=16384,
    )

    print(f"  ✓ Model saved: {model_file}")
    return str(model_file)


def evaluate_sentencepiece(model_path: str, sentences: list, label: str) -> dict:
    """Evaluate a SentencePiece model on sample sentences."""
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)

    total_tokens = 0
    total_chars = 0
    vowel_splits = 0
    total_vowels = 0

    for sent in sentences:
        pieces = sp.EncodeAsPieces(sent)
        total_tokens += len(pieces)
        total_chars += len(sent)

        # Check for split dependent vowels
        for piece in pieces:
            for ch in piece:
                if ch in DEPENDENT_VOWELS:
                    total_vowels += 1
            # A matra is "split" if it appears at the start of a token
            # (meaning it was separated from its consonant)
            clean_piece = piece.lstrip("▁")
            if clean_piece and clean_piece[0] in DEPENDENT_VOWELS:
                vowel_splits += 1

    avg_tokens = total_tokens / max(len(sentences), 1)
    split_pct = 100 * vowel_splits / max(total_vowels, 1)

    return {
        "tokenizer": label,
        "model_path": model_path,
        "sentences": len(sentences),
        "avg_tokens_per_sentence": round(avg_tokens, 2),
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "tokens_per_char": round(total_tokens / max(total_chars, 1), 4),
        "vowel_split_pct": round(split_pct, 2),
    }


def evaluate_hf_tokenizer(tokenizer_name: str, sentences: list, label: str) -> dict:
    """Evaluate a HuggingFace tokenizer on sample sentences."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
    except Exception as e:
        msg = str(e)
        hint = ""
        # Gated repos (e.g. most meta-llama/* checkpoints) return a 401/403
        # or an explicit "gated repo" message rather than a plain 404.
        if "gated" in msg.lower() or "401" in msg or "403" in msg:
            hint = (
                " -> This looks like a gated HuggingFace repo. Request access on the "
                "model page, then run `huggingface-cli login` (or set env var HF_TOKEN) "
                "before re-running this stage."
            )
        elif "not a valid model identifier" in msg.lower() or "404" in msg:
            hint = f" -> Check that '{tokenizer_name}' is the correct repo id."
        print(f"    Could not load {tokenizer_name}: {e}{hint}")
        return {
            "tokenizer": label,
            "model_path": tokenizer_name,
            "error": msg,
            "hint": hint.strip(" ->") or None,
        }

    total_tokens = 0
    total_chars = 0
    vowel_splits = 0
    total_vowels = 0

    for sent in sentences:
        tokens = tok.tokenize(sent)
        total_tokens += len(tokens)
        total_chars += len(sent)

        for token in tokens:
            for ch in token:
                if ch in DEPENDENT_VOWELS:
                    total_vowels += 1
            # Check for split matras at token start
            clean = token.lstrip("▁").lstrip("Ġ").lstrip("##")
            if clean and clean[0] in DEPENDENT_VOWELS:
                vowel_splits += 1

    avg_tokens = total_tokens / max(len(sentences), 1)
    split_pct = 100 * vowel_splits / max(total_vowels, 1)

    return {
        "tokenizer": label,
        "model_path": tokenizer_name,
        "sentences": len(sentences),
        "avg_tokens_per_sentence": round(avg_tokens, 2),
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "tokens_per_char": round(total_tokens / max(total_chars, 1), 4),
        "vowel_split_pct": round(split_pct, 2),
    }


def run_baselines(lang: str):
    """Train and evaluate all baseline tokenizers for a language."""
    ensure_dirs()

    print(f"\n  Loading sample sentences for {lang}...")
    sentences = load_sample_sentences(lang)
    if not sentences:
        print(f"  No sentences found for {lang}, skipping.")
        return

    print(f"  Loaded {len(sentences)} sentences")

    results = []

    # 1. Train and evaluate SentencePiece BPE
    print(f"\n  --- SentencePiece BPE ---")
    bpe_model = train_sentencepiece(lang, "bpe")
    if bpe_model:
        r = evaluate_sentencepiece(bpe_model, sentences, f"SP-BPE-{lang}")
        results.append(r)
        print(f"    Avg tokens/sent: {r['avg_tokens_per_sentence']}")
        print(f"    Vowel split %:   {r['vowel_split_pct']}")

    # 2. Train and evaluate SentencePiece Unigram
    print(f"\n  --- SentencePiece Unigram ---")
    uni_model = train_sentencepiece(lang, "unigram")
    if uni_model:
        r = evaluate_sentencepiece(uni_model, sentences, f"SP-Unigram-{lang}")
        results.append(r)
        print(f"    Avg tokens/sent: {r['avg_tokens_per_sentence']}")
        print(f"    Vowel split %:   {r['vowel_split_pct']}")

    # 3. Evaluate HuggingFace baseline tokenizers
    for tok_name in BASELINE_TOKENIZERS:
        short_name = tok_name.split("/")[-1]
        print(f"\n  --- {short_name} ---")
        r = evaluate_hf_tokenizer(tok_name, sentences, short_name)
        results.append(r)
        if "error" not in r:
            print(f"    Avg tokens/sent: {r['avg_tokens_per_sentence']}")
            print(f"    Vowel split %:   {r['vowel_split_pct']}")
        else:
            print(f"    ERROR: {r['error']}")

    # Save results
    results_file = RESULTS_DIR / lang / "baseline_tokenizer_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Results saved: {results_file}")

    # Print summary table
    print(f"\n  {'Tokenizer':<25} {'Tokens/Sent':>12} {'Tok/Char':>10} {'Vowel Split%':>13}")
    print(f"  {'─'*25} {'─'*12} {'─'*10} {'─'*13}")
    for r in results:
        if "error" in r:
            print(f"  {r['tokenizer']:<25} {'ERROR':>12}")
        else:
            print(f"  {r['tokenizer']:<25} {r['avg_tokens_per_sentence']:>12.2f} "
                  f"{r['tokens_per_char']:>10.4f} {r['vowel_split_pct']:>12.2f}%")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 2a: Train and evaluate baseline tokenizers")
    parser.add_argument("--lang", choices=LANGUAGES + ["all"], default="all",
                        help="Language to process (default: all)")
    args = parser.parse_args()

    langs = LANGUAGES if args.lang == "all" else [args.lang]

    for lang in langs:
        print(f"\n{'='*60}")
        print(f"  BASELINE TOKENIZERS: {lang.upper()}")
        print(f"{'='*60}")
        run_baselines(lang)

    print("\n✓ Stage 2a (Baselines) complete.")


if __name__ == "__main__":
    main()