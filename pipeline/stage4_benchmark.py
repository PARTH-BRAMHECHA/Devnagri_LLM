"""
Stage 4 — Comprehensive Benchmarking
======================================
Produces the master results table:
  Rows = language × text type
  Columns = method (classical compressors + LLM variants)
  Cells = BPC / compression ratio / empirical entropy

Usage:
    python -m pipeline.stage4_benchmark [--lang hindi|marathi|sanskrit|all] [--classical-only]
"""

import argparse
import bz2
import gzip
import json
import lzma
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pipeline.config import (
    SPLIT_DIR, TOKENIZER_DIR, RESULTS_DIR, LANGUAGES,
    CLASSICAL_COMPRESSORS, ensure_dirs
)


def compute_empirical_entropy(text: str) -> float:
    """
    Compute the 0th-order empirical character entropy of text (bits/char).
    Shannon entropy: H = -Σ p(c) * log2(p(c))
    """
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1

    total = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def compute_byte_entropy(data: bytes) -> float:
    """Compute 0th-order entropy at the byte level."""
    freq = {}
    for b in data:
        freq[b] = freq.get(b, 0) + 1

    total = len(data)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def benchmark_classical_compressors(text: str, label: str) -> dict:
    """Run all classical compressors and return metrics."""
    text_bytes = text.encode("utf-8")
    original_size = len(text_bytes)
    original_chars = len(text)

    results = {
        "label": label,
        "original_bytes": original_size,
        "original_chars": original_chars,
        "char_entropy": round(compute_empirical_entropy(text), 4),
        "byte_entropy": round(compute_byte_entropy(text_bytes), 4),
        "compressors": {},
    }

    # gzip (level 9)
    t0 = time.time()
    comp = gzip.compress(text_bytes, compresslevel=9)
    t1 = time.time()
    results["compressors"]["gzip-9"] = {
        "compressed_bytes": len(comp),
        "ratio": round(original_size / max(len(comp), 1), 4),
        "bpc": round(len(comp) * 8 / original_chars, 4),
        "bpb": round(len(comp) * 8 / original_size, 4),
        "time_s": round(t1 - t0, 3),
    }

    # bzip2 (level 9)
    t0 = time.time()
    comp = bz2.compress(text_bytes, compresslevel=9)
    t1 = time.time()
    results["compressors"]["bzip2-9"] = {
        "compressed_bytes": len(comp),
        "ratio": round(original_size / max(len(comp), 1), 4),
        "bpc": round(len(comp) * 8 / original_chars, 4),
        "bpb": round(len(comp) * 8 / original_size, 4),
        "time_s": round(t1 - t0, 3),
    }

    # LZMA (xz)
    t0 = time.time()
    comp = lzma.compress(text_bytes)
    t1 = time.time()
    results["compressors"]["lzma"] = {
        "compressed_bytes": len(comp),
        "ratio": round(original_size / max(len(comp), 1), 4),
        "bpc": round(len(comp) * 8 / original_chars, 4),
        "bpb": round(len(comp) * 8 / original_size, 4),
        "time_s": round(t1 - t0, 3),
    }

    # zstd (if available)
    try:
        import zstandard as zstd
        t0 = time.time()
        cctx = zstd.ZstdCompressor(level=19)
        comp = cctx.compress(text_bytes)
        t1 = time.time()
        results["compressors"]["zstd-19"] = {
            "compressed_bytes": len(comp),
            "ratio": round(original_size / max(len(comp), 1), 4),
            "bpc": round(len(comp) * 8 / original_chars, 4),
            "bpb": round(len(comp) * 8 / original_size, 4),
            "time_s": round(t1 - t0, 3),
        }
    except ImportError:
        results["compressors"]["zstd-19"] = {"error": "zstandard not installed"}

    # Hybrid: bzip2 on gzip-compressed data
    t0 = time.time()
    stage1 = gzip.compress(text_bytes, compresslevel=9)
    stage2 = bz2.compress(stage1, compresslevel=9)
    t1 = time.time()
    results["compressors"]["gzip+bzip2"] = {
        "compressed_bytes": len(stage2),
        "ratio": round(original_size / max(len(stage2), 1), 4),
        "bpc": round(len(stage2) * 8 / original_chars, 4),
        "bpb": round(len(stage2) * 8 / original_size, 4),
        "time_s": round(t1 - t0, 3),
    }

    # Hybrid: LZMA on gzip-compressed data
    t0 = time.time()
    stage1 = gzip.compress(text_bytes, compresslevel=9)
    stage2 = lzma.compress(stage1)
    t1 = time.time()
    results["compressors"]["gzip+lzma"] = {
        "compressed_bytes": len(stage2),
        "ratio": round(original_size / max(len(stage2), 1), 4),
        "bpc": round(len(stage2) * 8 / original_chars, 4),
        "bpb": round(len(stage2) * 8 / original_size, 4),
        "time_s": round(t1 - t0, 3),
    }

    return results


def run_benchmark(lang: str, classical_only: bool = False):
    """Run full benchmark for a language."""
    ensure_dirs()

    test_file = SPLIT_DIR / lang / "test.txt"
    if not test_file.exists():
        print(f"  ERROR: {test_file} not found. Run stage1 first.")
        return

    with open(test_file, "r", encoding="utf-8") as f:
        test_text = f.read()

    print(f"  Test set: {len(test_text):,} chars, "
          f"{len(test_text.encode('utf-8')) / (1024**2):.2f} MB")

    all_results = {
        "language": lang,
        "test_chars": len(test_text),
        "test_bytes": len(test_text.encode("utf-8")),
    }

    # Benchmark on full test text (pure-language text)
    print(f"\n  --- Classical Compressors (Pure Language Text) ---")
    classical_results = benchmark_classical_compressors(test_text, f"{lang}_pure")
    all_results["pure_language"] = classical_results

    print(f"\n  {'Compressor':<15} {'Ratio':>8} {'BPC':>8} {'BPB':>8} {'Time':>8}")
    print(f"  {'─'*15} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for name, r in classical_results["compressors"].items():
        if "error" in r:
            print(f"  {name:<15} {'ERROR':>8}")
        else:
            print(f"  {name:<15} {r['ratio']:>8.3f} {r['bpc']:>8.3f} "
                  f"{r['bpb']:>8.3f} {r['time_s']:>7.2f}s")

    print(f"\n  Empirical char entropy: {classical_results['char_entropy']:.4f} bits/char")
    print(f"  Empirical byte entropy: {classical_results['byte_entropy']:.4f} bits/byte")

    # LLM compression benchmarks (if not classical-only)
    if not classical_only:
        try:
            from pipeline.stage3_compress import LLMCompressor
            print(f"\n  --- LLM Compression ---")

            compressor = LLMCompressor("ai4bharat/Airavata")

            # BPC on sample (LLM is slow, use smaller sample)
            sample = test_text[:20_000]
            print(f"  Computing LLM BPC on {len(sample):,} chars...")
            bpc_result = compressor.compute_bpc(sample)
            all_results["llm_default_tokenizer"] = bpc_result

            print(f"    LLM BPC: {bpc_result['bpc']:.4f}")
            print(f"    Compression ratio: {bpc_result['compression_ratio']:.3f}")

        except Exception as e:
            print(f"  LLM benchmark error: {e}")
            all_results["llm_default_tokenizer"] = {"error": str(e)}

    # Save full results
    results_file = RESULTS_DIR / lang / "benchmark_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Benchmark results saved: {results_file}")

    return all_results


def generate_master_table(results_dir: Path = None):
    """
    Generate the master results table from all per-language benchmark files.
    This is the central figure for the paper.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    print(f"\n{'='*80}")
    print(f"  MASTER RESULTS TABLE")
    print(f"{'='*80}")

    all_data = {}
    for lang in LANGUAGES:
        results_file = results_dir / lang / "benchmark_results.json"
        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                all_data[lang] = json.load(f)

    if not all_data:
        print("  No benchmark results found. Run stage4_benchmark first.")
        return

    # Print table
    compressors = ["gzip-9", "bzip2-9", "lzma", "zstd-19", "gzip+bzip2", "gzip+lzma"]
    header = f"  {'Language':<12} {'Chars':>10} {'Bytes':>10} {'Entropy':>8}"
    for c in compressors:
        header += f" {c:>10}"
    header += f" {'LLM BPC':>10}"

    print(header)
    print("  " + "─" * (len(header) - 2))

    for lang, data in all_data.items():
        pure = data.get("pure_language", {})
        row = f"  {lang:<12} {data.get('test_chars', 0):>10,} {data.get('test_bytes', 0):>10,}"
        row += f" {pure.get('char_entropy', 0):>8.3f}"

        comp_data = pure.get("compressors", {})
        for c in compressors:
            r = comp_data.get(c, {})
            if "error" in r:
                row += f" {'ERR':>10}"
            elif r:
                row += f" {r.get('bpc', 0):>10.3f}"
            else:
                row += f" {'N/A':>10}"

        llm = data.get("llm_default_tokenizer", {})
        if "error" in llm:
            row += f" {'ERR':>10}"
        elif llm:
            row += f" {llm.get('bpc', 0):>10.3f}"
        else:
            row += f" {'N/A':>10}"

        print(row)

    # Save as JSON
    master_file = results_dir / "master_results.json"
    with open(master_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Master results saved: {master_file}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 4: Comprehensive benchmarking")
    parser.add_argument("--lang", choices=LANGUAGES + ["all"], default="all",
                        help="Language to benchmark (default: all)")
    parser.add_argument("--classical-only", action="store_true",
                        help="Only run classical compressors")
    parser.add_argument("--table-only", action="store_true",
                        help="Only generate master table from existing results")
    args = parser.parse_args()

    if args.table_only:
        generate_master_table()
        return

    langs = LANGUAGES if args.lang == "all" else [args.lang]

    for lang in langs:
        print(f"\n{'='*60}")
        print(f"  BENCHMARKING: {lang.upper()}")
        print(f"{'='*60}")
        run_benchmark(lang, classical_only=args.classical_only)

    # Generate master table
    generate_master_table()

    print("\n✓ Stage 4 (Benchmarking) complete.")


if __name__ == "__main__":
    main()
