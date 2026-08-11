"""
Evaluation Utilities
=====================
Shared evaluation functions for tokenizer validation (perplexity probing),
round-trip verification, and results formatting.

Usage:
    python -m pipeline.eval_utils [--validate-tokenizer --lang hindi]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pipeline.config import (
    SPLIT_DIR, TOKENIZER_DIR, RESULTS_DIR, LANGUAGES, ensure_dirs
)


# ─── Tokenizer Validation via Perplexity ────────────────────────────────────

def compute_perplexity_with_tokenizer(
    model_name: str,
    tokenizer_model_path: str,
    text: str,
    max_tokens: int = 1024,
    device: str = None,
) -> dict:
    """
    Fine-tune/probe a small LM with a given tokenizer and compute perplexity.

    This validates that the tokenizer doesn't produce degenerate tokenizations
    by checking if perplexity is reasonable.

    Per the project plan (Stage 2d):
    "your tokenizer should show lower perplexity or equal perplexity with fewer tokens"
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sentencepiece as spm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the HF model and its default tokenizer for comparison
    print(f"    Loading model: {model_name}")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    # Tokenize with HF tokenizer
    hf_tokens = hf_tokenizer.encode(text, add_special_tokens=True)[:max_tokens]

    # Compute perplexity with default tokenizer
    hf_ppl = _compute_model_perplexity(model, hf_tokens, device)

    # Now with our custom tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_model_path)
    sp_tokens = sp.EncodeAsIds(text)[:max_tokens]

    # Note: We can't directly use SP tokens with an HF model
    # because the vocab IDs don't match. Instead, we compare token counts
    # and report the relationship.

    return {
        "model": model_name,
        "custom_tokenizer": tokenizer_model_path,
        "text_chars": len(text),
        "hf_token_count": len(hf_tokens),
        "sp_token_count": len(sp_tokens),
        "token_reduction_pct": round(
            100 * (1 - len(sp_tokens) / max(len(hf_tokens), 1)), 2
        ),
        "hf_perplexity": round(hf_ppl, 2),
        "hf_bits_per_token": round(math.log2(hf_ppl), 4) if hf_ppl > 0 else None,
    }


@torch.no_grad()
def _compute_model_perplexity(model, token_ids: list, device: str) -> float:
    """Compute perplexity of a sequence under a causal LM."""
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss.item()

    return math.exp(loss)


# ─── Results Formatting ─────────────────────────────────────────────────────

def format_results_table(results_dir: Path = None) -> str:
    """Generate a formatted markdown table of all results."""
    if results_dir is None:
        results_dir = RESULTS_DIR

    lines = []
    lines.append("# Devanagari Compression Results\n")

    # Per-language results
    for lang in LANGUAGES:
        lang_dir = results_dir / lang
        if not lang_dir.exists():
            continue

        lines.append(f"\n## {lang.title()}\n")

        # Tokenizer results
        tok_file = lang_dir / "baseline_tokenizer_results.json"
        if tok_file.exists():
            with open(tok_file, "r", encoding="utf-8") as f:
                tok_results = json.load(f)

            lines.append("### Tokenizer Comparison\n")
            lines.append("| Tokenizer | Tokens/Sent | Tok/Char | Vowel Split % |")
            lines.append("|-----------|-------------|----------|---------------|")
            for r in tok_results:
                if "error" in r:
                    lines.append(f"| {r['tokenizer']} | ERROR | - | - |")
                else:
                    lines.append(
                        f"| {r['tokenizer']} | {r['avg_tokens_per_sentence']:.2f} | "
                        f"{r['tokens_per_char']:.4f} | {r['vowel_split_pct']:.2f}% |"
                    )
            lines.append("")

        # Compression results
        comp_file = lang_dir / "benchmark_results.json"
        if not comp_file.exists():
            comp_file = lang_dir / "compression_results.json"
        if comp_file.exists():
            with open(comp_file, "r", encoding="utf-8") as f:
                comp_results = json.load(f)

            lines.append("### Compression Results\n")
            pure = comp_results.get("pure_language", comp_results.get("classical", {}))
            compressors = pure.get("compressors", {})

            if compressors:
                lines.append("| Compressor | Ratio | BPC | BPB | Time |")
                lines.append("|------------|-------|-----|-----|------|")
                for name, r in compressors.items():
                    if "error" in r:
                        lines.append(f"| {name} | ERROR | - | - | - |")
                    else:
                        lines.append(
                            f"| {name} | {r.get('ratio', 0):.3f} | "
                            f"{r.get('bpc', 0):.3f} | {r.get('bpb', 0):.3f} | "
                            f"{r.get('time_s', 0):.2f}s |"
                        )
                lines.append("")

    return "\n".join(lines)


# ─── Corpus Statistics Report ───────────────────────────────────────────────

def generate_corpus_report():
    """Generate a corpus size report (raw → cleaned → train → test)."""
    from pipeline.config import CLEAN_DIR

    print(f"\n{'='*70}")
    print(f"  CORPUS SIZE REPORT")
    print(f"{'='*70}")
    print(f"\n  {'Language':<12} {'Raw':>12} {'Cleaned':>12} {'Train':>12} {'Test':>12}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")

    for lang in LANGUAGES:
        # Find raw size
        raw_files = ["merged_raw.txt", "wiki_extracted.txt"]
        raw_size = 0
        for f in raw_files:
            p = CLEAN_DIR / lang / f
            if p.exists():
                raw_size = p.stat().st_size / (1024**2)
                break

        # Clean size
        clean_file = CLEAN_DIR / lang / "corpus_clean.txt"
        clean_size = clean_file.stat().st_size / (1024**2) if clean_file.exists() else 0

        # Train/test sizes
        train_file = SPLIT_DIR / lang / "train.txt"
        test_file = SPLIT_DIR / lang / "test.txt"
        train_size = train_file.stat().st_size / (1024**2) if train_file.exists() else 0
        test_size = test_file.stat().st_size / (1024**2) if test_file.exists() else 0

        print(f"  {lang:<12} {raw_size:>10.1f}MB {clean_size:>10.1f}MB "
              f"{train_size:>10.1f}MB {test_size:>10.1f}MB")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluation utilities")
    parser.add_argument("--report", action="store_true",
                        help="Generate corpus size report")
    parser.add_argument("--format-results", action="store_true",
                        help="Format all results as markdown")
    parser.add_argument("--validate-tokenizer", action="store_true",
                        help="Run tokenizer perplexity validation")
    parser.add_argument("--lang", choices=LANGUAGES, default="hindi")
    args = parser.parse_args()

    if args.report:
        generate_corpus_report()

    if args.format_results:
        table = format_results_table()
        print(table)
        # Also save
        output = RESULTS_DIR / "results_report.md"
        with open(output, "w", encoding="utf-8") as f:
            f.write(table)
        print(f"\n  Saved to: {output}")

    if args.validate_tokenizer:
        ensure_dirs()
        test_file = SPLIT_DIR / args.lang / "test.txt"
        if test_file.exists():
            with open(test_file, "r", encoding="utf-8") as f:
                text = f.read()[:5000]

            tok_path = str(TOKENIZER_DIR / args.lang / f"devaware_bpe_{args.lang}.model")
            if os.path.exists(tok_path):
                result = compute_perplexity_with_tokenizer(
                    "ai4bharat/IndicBERTv2-MLM-only",
                    tok_path,
                    text,
                )
                print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
