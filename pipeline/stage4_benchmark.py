"""
Stage 4 — Comprehensive Benchmarking
======================================
Produces the master results table:
  Rows = language × text type
  Columns = method (classical compressors + LLM variants)
  Cells = BPC / compression ratio / empirical entropy

Usage:
    python -m pipeline.stage4_benchmark [--lang hindi|marathi|sanskrit|all] [--classical-only]

─────────────────────────────────────────────────────────────────────────────
CHANGES from the version that ran on Kaggle:

1. Was hardcoding "ai4bharat/Airavata" instead of importing INDIC_LLM_MODEL
   from config.py — the two could silently drift apart. Now imports the
   same constant Stage 3 uses.

2. Was reloading the 7B model AND recomputing LLM BPC from scratch for
   every language, duplicating the exact same measurement Stage 3 already
   made (just on a different sample size — 20K chars here vs 50K in Stage
   3). That's why Stage 4 alone took ~3.5 hours on top of Stage 3's ~4.5
   hours in the log you shared. Now:
     - `run_benchmark` first tries to reuse Stage 3's saved
       `compression_results.json` for that language (same metric, no
       recompute needed) if it exists and succeeded.
     - Only falls back to computing fresh (e.g. classical-only wasn't run,
       or Stage 3's LLM result errored out) — and even then, accepts an
       already-loaded `compressor` so the model is loaded once and shared
       across languages instead of once per language.

3. Fixed a real (separate) bug in generate_master_table(): the header line
   `header += " {'LLM BPC':>10}"` was a plain string, not an f-string, so
   it was literally printing the text `{'LLM BPC':>10}` in the table header
   instead of formatting "LLM BPC" right-aligned to width 10.
─────────────────────────────────────────────────────────────────────────────
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
    CLASSICAL_COMPRESSORS, INDIC_LLM_MODEL, ensure_dirs
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


def _load_stage3_llm_result(lang: str):
    """
    Look for usable LLM result(s) already computed by Stage 3 for this
    language, so Stage 4 doesn't reload the 7B model and recompute the same
    thing again.

    Returns a dict with up to two keys:
      "default"  -- model-default-tokenizer result (from "llm_compression")
      "devaware" -- "your tokenizer" result (from
                    "llm_compression_devaware_tokenizer"), present only if
                    Stage 3 was run with --use-devaware-tokenizer and it
                    succeeded.
    Returns None if compression_results.json doesn't exist, is unreadable,
    or has no usable default-tokenizer result (devaware alone, without a
    default-tokenizer baseline to compare against, isn't enough to satisfy
    the caller -- see run_benchmark).
    """
    results_file = RESULTS_DIR / lang / "compression_results.json"
    if not results_file.exists():
        return None
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    default = data.get("llm_compression")
    if not default or "error" in default:
        return None

    out = {"default": default}

    devaware = data.get("llm_compression_devaware_tokenizer")
    if devaware and "error" not in devaware:
        out["devaware"] = devaware

    base_devaware = data.get("llm_compression_base_devaware_tokenizer")
    if base_devaware and "error" not in base_devaware:
        out["base_devaware"] = base_devaware

    finetuned_default = data.get("llm_compression_finetuned_default_tokenizer")
    if finetuned_default and "error" not in finetuned_default:
        out["finetuned_default"] = finetuned_default

    return out


def generate_llm_text(compressor, lang: str, n_chars: int = 20_000,
                       cache_dir: Path = None, force_regenerate: bool = False) -> str:
    """
    Have the LLM generate `n_chars` of its own text in `lang`, so Stage 4
    can measure BPC on LLM-GENERATED text as a distinct condition from
    BPC on real (human-written) test text ("pure_language"). Without
    this, the pipeline only ever measured how well the LLM compresses
    real text -- it never checked whether the model's own output has a
    different (typically lower, since it's in-distribution for the model
    by construction) BPC, which is the actual comparison the "text-type
    conditions" part of the project plan calls for.

    Cached to `cache_dir/llm_generated.txt` (default: RESULTS_DIR/lang/)
    so repeated Stage 4 runs don't re-generate (generation is slow --
    autoregressive, one token at a time, same as the compression measurement
    itself). Pass `force_regenerate=True` to bypass the cache.

    Generation is looped in chunks (rather than one huge `max_new_tokens`
    call) since a single generate() call holding the whole KV cache for a
    very long sequence is a common source of the same OOM issues Stage 2d's
    docstring warns about; each chunk re-seeds from a short rolling context
    instead of keeping one unbounded cache alive.
    """
    cache_dir = cache_dir or (RESULTS_DIR / lang)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "llm_generated.txt"

    if cache_file.exists() and not force_regenerate:
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = f.read()
        if len(cached) >= n_chars:
            print(f"  Reusing cached LLM-generated text ({len(cached):,} chars) "
                  f"from {cache_file}")
            return cached[:n_chars]

    # Short seed prompts to bootstrap generation in each language -- just
    # enough to bias the model toward Devanagari continuation rather than
    # code-switching to English, not meant as a content prompt.
    seed_prompts = {
        "hindi": "भारत एक विशाल देश है जिसकी सांस्कृतिक विरासत",
        "marathi": "महाराष्ट्र हे भारतातील एक महत्त्वाचे राज्य आहे",
        "sanskrit": "प्राचीनकाले भारतवर्षे विद्यायाः महत्त्वं अत्यधिकम् आसीत्",
    }
    prompt = seed_prompts.get(lang, seed_prompts["hindi"])

    import torch
    tokenizer, model = compressor.tokenizer, compressor.model
    device = compressor.device

    generated_text = prompt
    chunk_new_tokens = 256
    max_chunks = 40  # safety cap regardless of n_chars, to bound worst-case time
    chunks_done = 0

    print(f"  Generating LLM text for {lang} (target {n_chars:,} chars, "
          f"seeded from a short prompt)...")
    while len(generated_text) < n_chars and chunks_done < max_chunks:
        # Roll the context forward using only the tail of what's been
        # generated so far, rather than feeding the whole growing string
        # back in -- keeps each generate() call's context bounded.
        rolling_context = generated_text[-2000:]
        input_ids = tokenizer(rolling_context, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=chunk_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
            )
        new_ids = output_ids[0][input_ids.shape[1]:]
        new_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        if not new_text.strip():
            # Model produced nothing new (e.g. hit EOS immediately) --
            # stop rather than looping forever on an empty continuation.
            break
        generated_text += new_text
        chunks_done += 1

    if len(generated_text) < n_chars:
        print(f"  ⚠ Only generated {len(generated_text):,} chars "
              f"(target was {n_chars:,}) -- model may be hitting EOS early "
              f"or generation was capped by max_chunks={max_chunks}. "
              f"Proceeding with what was generated.")

    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(generated_text)
    print(f"  ✓ Generated + cached {len(generated_text):,} chars -> {cache_file}")

    return generated_text[:n_chars]


def run_benchmark(lang: str, classical_only: bool = False, compressor=None):
    """
    Run full benchmark for a language.

    `compressor`: an already-loaded LLMCompressor to reuse (avoids
    reloading the 7B model per language). Only used if Stage 3's result
    can't be reused and a fresh LLM computation is actually needed.
    """
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
        print(f"\n  --- LLM Compression ---")

        reused = _load_stage3_llm_result(lang)
        if reused is not None:
            print(f"  Reusing LLM BPC already computed in Stage 3 "
                  f"(no reload, no recompute)")
            all_results["llm_default_tokenizer"] = reused["default"]
            print(f"    LLM BPC: {reused['default']['bpc']:.4f}")
            print(f"    Compression ratio: {reused['default']['compression_ratio']:.3f}")

            if "devaware" in reused:
                all_results["llm_devaware_tokenizer"] = reused["devaware"]
                print(f"    LLM BPC (your tokenizer): {reused['devaware']['bpc']:.4f}")
                print(f"    Compression ratio (your tokenizer): "
                      f"{reused['devaware']['compression_ratio']:.3f}")
            else:
                print(f"    (no 'your tokenizer' condition in Stage 3's result for "
                      f"{lang} -- was --use-devaware-tokenizer passed to Stage 3?)")

            # Ablation grid cells (tokenizer effect vs fine-tuning effect --
            # see stage3_compress.run_compression docstring). Only present
            # if Stage 3 was run with --use-devaware-tokenizer.
            if "base_devaware" in reused:
                all_results["llm_base_devaware_tokenizer"] = reused["base_devaware"]
                print(f"    LLM BPC (devaware tokenizer, NOT fine-tuned): "
                      f"{reused['base_devaware']['bpc']:.4f}")
            if "finetuned_default" in reused:
                all_results["llm_finetuned_default_tokenizer"] = reused["finetuned_default"]
                print(f"    LLM BPC (fine-tuned, default tokenizer): "
                      f"{reused['finetuned_default']['bpc']:.4f}")
        else:
            try:
                from pipeline.stage3_compress import LLMCompressor

                if compressor is None:
                    compressor = LLMCompressor(INDIC_LLM_MODEL)

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

        # ─── LLM-generated-text condition ───────────────────────────────
        # Everything above measures BPC on REAL (human-written) test text.
        # This measures the same thing on text the LLM generates itself --
        # a genuinely different condition, since a model's own output is
        # close to in-distribution for it by construction. Stored under
        # "llm_generated" with the same shape as "pure_language"
        # (classical compressors + char/byte entropy) plus an
        # "llm_default_tokenizer" BPC entry, so Stage 5's
        # `br.get("pure_language", ...)` -style lookups work unchanged
        # against this key too.
        try:
            from pipeline.stage3_compress import LLMCompressor

            if compressor is None:
                compressor = LLMCompressor(INDIC_LLM_MODEL)

            gen_text = generate_llm_text(compressor, lang, n_chars=20_000)

            print(f"\n  --- Classical Compressors (LLM-Generated Text) ---")
            gen_classical_results = benchmark_classical_compressors(gen_text, f"{lang}_llm_generated")

            for name, r in gen_classical_results["compressors"].items():
                if "error" in r:
                    print(f"  {name:<15} {'ERROR':>8}")
                else:
                    print(f"  {name:<15} {r['ratio']:>8.3f} {r['bpc']:>8.3f} "
                          f"{r['bpb']:>8.3f} {r['time_s']:>7.2f}s")

            print(f"\n  --- LLM Compression (LLM-Generated Text) ---")
            gen_bpc_result = compressor.compute_bpc(gen_text)
            print(f"    LLM BPC on its own generated text: {gen_bpc_result['bpc']:.4f}")
            print(f"    (vs {all_results.get('llm_default_tokenizer', {}).get('bpc', float('nan')):.4f} "
                  f"on real test text -- a lower number here mostly reflects "
                  f"the model finding its own output easy to predict, NOT a "
                  f"language-modeling improvement)")

            all_results["llm_generated"] = {
                **gen_classical_results,
                "llm_default_tokenizer": gen_bpc_result,
            }
        except Exception as e:
            print(f"  LLM-generated-text benchmark error: {e}")
            all_results["llm_generated"] = {"error": str(e)}

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
    header += f" {'LLM BPC':>10}"   # FIX: was a plain string, never formatted
    header += f" {'LLM BPC*':>10}"  # * = your tokenizer (devaware, fine-tuned)

    print(header)
    print("  " + "─" * (len(header) - 2))

    any_missing_devaware = []

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

        llm_dev = data.get("llm_devaware_tokenizer", {})
        if "error" in llm_dev:
            row += f" {'ERR':>10}"
        elif llm_dev:
            row += f" {llm_dev.get('bpc', 0):>10.3f}"
        else:
            row += f" {'N/A':>10}"
            any_missing_devaware.append(lang)

        print(row)

    print("\n  * LLM BPC* = your Devanagari-aware tokenizer + fine-tuned model.")
    if any_missing_devaware:
        print(f"  ⚠ 'your tokenizer' condition missing for: "
              f"{', '.join(any_missing_devaware)}. This table is NOT the "
              f"three-condition comparison the project plan calls for until "
              f"every row has an 'LLM BPC*' value -- re-run Stage 3 with "
              f"--use-devaware-tokenizer for those languages before treating "
              f"this as final.")

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
