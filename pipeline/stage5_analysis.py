"""
Stage 5 — Cross-Lingual Analysis
==================================
Analyzes how morphological complexity and resource availability
relate to compression performance across Hindi, Marathi, Sanskrit.

Key analyses:
  1. Tokenizer contribution vs. LLM contribution (ablation)
  2. BPC vs. morphological complexity measures
  3. Resource availability vs. compression performance

Usage:
    python -m pipeline.stage5_analysis
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pipeline.config import (
    SPLIT_DIR, TOKENIZER_DIR, RESULTS_DIR, LANGUAGES, ensure_dirs
)


# ─── Morphological Complexity Measures ───────────────────────────────────────

def compute_type_token_ratio(text: str, max_tokens: int = 100_000) -> float:
    """
    Compute Type-Token Ratio (TTR) as a simple morphological complexity proxy.
    Higher TTR = more unique word forms = likely richer morphology.
    """
    words = text.split()[:max_tokens]
    if not words:
        return 0.0
    types = len(set(words))
    return types / len(words)


def compute_avg_word_length(text: str, max_tokens: int = 100_000) -> float:
    """Average word length in characters (longer → more morphology)."""
    words = [w for w in text.split()[:max_tokens] if w]
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def compute_hapax_ratio(text: str, max_tokens: int = 100_000) -> float:
    """
    Hapax legomena ratio: fraction of words that appear exactly once.
    Higher → richer morphology (more unique inflected forms).
    """
    words = text.split()[:max_tokens]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    hapax = sum(1 for c in freq.values() if c == 1)
    return hapax / max(len(freq), 1)


def compute_entropy_rate(text: str) -> float:
    """Character-level entropy rate (bits/char)."""
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


def analyze_morphological_complexity(lang: str, text: str) -> dict:
    """Compute all morphological complexity measures for a language."""
    ttr = compute_type_token_ratio(text)
    avg_len = compute_avg_word_length(text)
    hapax = compute_hapax_ratio(text)
    entropy = compute_entropy_rate(text)

    words = text.split()[:100_000]
    n_words = len(words)
    n_types = len(set(words))

    return {
        "language": lang,
        "word_count": n_words,
        "type_count": n_types,
        "type_token_ratio": round(ttr, 4),
        "avg_word_length_chars": round(avg_len, 2),
        "hapax_ratio": round(hapax, 4),
        "char_entropy": round(entropy, 4),
        # Expected ordinal ranking: Hindi (moderate) < Marathi (rich) < Sanskrit (sandhi-heavy)
        "complexity_notes": {
            "hindi": "Moderate morphological complexity, postpositions",
            "marathi": "Rich inflectional morphology, agglutinative tendencies",
            "sanskrit": "Highly inflectional + sandhi + compounds",
        }.get(lang, ""),
    }


def aggregate_seed_variance(result_paths: dict, metric_path: tuple = ("llm_compression", "bpc")) -> dict:
    """
    Combine compression_results.json files from multiple seeded runs (see
    config.SEED / run_pipeline.py's --seed) into a mean/std summary, so a
    BPC delta can be checked against run-to-run noise before being reported
    as a real effect.

    `result_paths`: {seed_label: path_to_compression_results.json}, e.g.
        {
            "seed_1": RESULTS_DIR / "hindi" / "compression_results_seed1.json",
            "seed_2": RESULTS_DIR / "hindi" / "compression_results_seed2.json",
            "seed_3": RESULTS_DIR / "hindi" / "compression_results_seed3.json",
        }
    (Point these at wherever you saved each seed's run -- e.g. by setting
    RESULTS_DIR_SUFFIX per run, per config.py's multi-seed workflow comment.)

    `metric_path`: nested-key path into each result dict identifying the
    metric to aggregate, e.g. ("llm_compression", "bpc") for the plain
    baseline, or ("llm_compression_devaware_tokenizer", "bpc") for the
    devaware+fine-tuned condition.

    Returns a dict with per-seed values plus mean/std/n, or an "error" key
    if fewer than 2 usable seeds were found (variance is meaningless with
    one data point).
    """
    values = {}
    for seed_label, path in result_paths.items():
        path = Path(path)
        if not path.exists():
            print(f"  ⚠ {seed_label}: {path} not found, skipping")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ⚠ {seed_label}: could not read {path} ({e}), skipping")
            continue

        node = data
        ok = True
        for key in metric_path:
            if not isinstance(node, dict) or key not in node:
                ok = False
                break
            node = node[key]
        if not ok or not isinstance(node, (int, float)):
            print(f"  ⚠ {seed_label}: metric path {'.'.join(metric_path)} "
                  f"not found or not numeric in {path}, skipping")
            continue
        values[seed_label] = float(node)

    if len(values) < 2:
        return {
            "metric": ".".join(metric_path),
            "values": values,
            "error": f"Need at least 2 usable seeds to report variance; got {len(values)}. "
                     f"A single-seed number should not be reported as a stable effect.",
        }

    vals = list(values.values())
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)  # sample variance
    std = math.sqrt(variance)

    return {
        "metric": ".".join(metric_path),
        "values": values,
        "n": len(vals),
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
        "range": round(max(vals) - min(vals), 6),
    }


def compare_conditions_across_seeds(result_paths: dict) -> dict:
    """
    Convenience wrapper around aggregate_seed_variance: reports mean/std for
    both the "no fine-tune" baseline and the "devaware tokenizer, fine-tuned"
    condition across the same set of seeded runs, plus the delta between
    their means relative to their combined std -- a quick, rough signal for
    "is this delta bigger than the noise" (NOT a substitute for a proper
    significance test, but enough to catch a delta that's clearly within
    noise before writing it up as a finding).
    """
    baseline = aggregate_seed_variance(result_paths, ("llm_compression", "bpc"))
    devaware = aggregate_seed_variance(
        result_paths, ("llm_compression_devaware_tokenizer", "bpc")
    )

    out = {"baseline": baseline, "devaware_finetuned": devaware}

    if "mean" in baseline and "mean" in devaware:
        delta = baseline["mean"] - devaware["mean"]
        combined_std = math.sqrt(baseline["std"] ** 2 + devaware["std"] ** 2)
        out["delta_bpc"] = round(delta, 6)
        out["delta_pct"] = round(100 * delta / baseline["mean"], 3) if baseline["mean"] else None
        out["combined_std"] = round(combined_std, 6)
        if combined_std > 0:
            out["delta_over_combined_std"] = round(delta / combined_std, 3)
            if abs(delta) < combined_std:
                out["verdict"] = (
                    "Delta is smaller than the combined across-seed std -- "
                    "not distinguishable from run-to-run noise on this evidence."
                )
            else:
                out["verdict"] = (
                    "Delta exceeds the combined across-seed std -- more likely "
                    "a real effect, though this is a rough check, not a "
                    "significance test."
                )

    return out


# ─── Cross-Lingual Comparison ───────────────────────────────────────────────

def load_benchmark_results() -> dict:
    """Load benchmark results from all languages."""
    results = {}
    for lang in LANGUAGES:
        # Try benchmark results first, then compression results
        for filename in ["benchmark_results.json", "compression_results.json"]:
            path = RESULTS_DIR / lang / filename
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    results[lang] = json.load(f)
                break
    return results


def load_tokenizer_results() -> dict:
    """Load tokenizer comparison results."""
    results = {}
    for lang in LANGUAGES:
        for filename in ["devanagari_tokenizer_comparison.json",
                         "baseline_tokenizer_results.json"]:
            path = RESULTS_DIR / lang / filename
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    results[lang] = json.load(f)
                break
    return results


def run_analysis():
    """Run the full cross-lingual analysis."""
    ensure_dirs()

    print(f"\n{'='*70}")
    print(f"  CROSS-LINGUAL ANALYSIS")
    print(f"{'='*70}")

    # Step 1: Compute morphological complexity for each language
    print(f"\n  --- Morphological Complexity Analysis ---")
    complexity = {}

    for lang in LANGUAGES:
        test_file = SPLIT_DIR / lang / "test.txt"
        if not test_file.exists():
            print(f"  {lang}: test data not found, skipping")
            continue

        with open(test_file, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"\n  {lang.upper()}:")
        c = analyze_morphological_complexity(lang, text)
        complexity[lang] = c

        print(f"    Words: {c['word_count']:,}")
        print(f"    Types: {c['type_count']:,}")
        print(f"    TTR:   {c['type_token_ratio']:.4f}")
        print(f"    Avg word length: {c['avg_word_length_chars']:.2f} chars")
        print(f"    Hapax ratio:     {c['hapax_ratio']:.4f}")
        print(f"    Char entropy:    {c['char_entropy']:.4f} bits/char")

    # Step 2: Load compression results
    print(f"\n  --- Compression Results Cross-Comparison ---")
    benchmark_results = load_benchmark_results()
    tokenizer_results = load_tokenizer_results()

    # Step 3: Build comparison table
    comparison = {
        "morphological_complexity": complexity,
        "compression_results": {},
        "tokenizer_results": {},
    }

    for lang in LANGUAGES:
        if lang in benchmark_results:
            br = benchmark_results[lang]
            # Extract best classical BPC
            pure = br.get("pure_language", br.get("classical", {}))
            compressors = pure.get("compressors", {})

            best_classical_bpc = float("inf")
            best_classical_name = ""
            for name, r in compressors.items():
                if isinstance(r, dict) and "bpc" in r:
                    if r["bpc"] < best_classical_bpc:
                        best_classical_bpc = r["bpc"]
                        best_classical_name = name

            comparison["compression_results"][lang] = {
                "best_classical": best_classical_name,
                "best_classical_bpc": best_classical_bpc if best_classical_bpc < float("inf") else None,
                "char_entropy": pure.get("char_entropy"),
                "llm_bpc": br.get("llm_default_tokenizer", {}).get("bpc"),
            }

            # Pure (real, human-written) text vs LLM-generated text -- the
            # text-type condition the project plan called for but the
            # pipeline previously never produced (see stage4_benchmark's
            # generate_llm_text). A model's own generations are close to
            # in-distribution for it by construction, so this is expected
            # to show a lower BPC on "llm_generated" than on
            # "pure_language" -- report it as exactly that (a sanity/
            # distribution check), not as a claim the model got better at
            # compressing the LANGUAGE.
            llm_generated = br.get("llm_generated")
            if llm_generated and "error" not in llm_generated:
                comparison["compression_results"][lang]["llm_generated_bpc"] = (
                    llm_generated.get("llm_default_tokenizer", {}).get("bpc")
                )
                comparison["compression_results"][lang]["llm_generated_char_entropy"] = (
                    llm_generated.get("char_entropy")
                )

        if lang in tokenizer_results:
            tr = tokenizer_results[lang]
            if isinstance(tr, dict):
                devaware = tr.get("devanagari_aware", {})
                baseline = tr.get("baseline_bpe", {})
                comparison["tokenizer_results"][lang] = {
                    "devaware_tokens_per_sent": devaware.get("avg_tokens_per_sentence"),
                    "devaware_vowel_split": devaware.get("vowel_split_pct"),
                    "baseline_tokens_per_sent": baseline.get("avg_tokens_per_sentence") if baseline else None,
                    "baseline_vowel_split": baseline.get("vowel_split_pct") if baseline else None,
                }

    # Step 4: Print summary table
    print(f"\n  {'='*70}")
    print(f"  CROSS-LINGUAL SUMMARY")
    print(f"  {'='*70}")
    print(f"\n  {'Language':<12} {'TTR':>8} {'AvgLen':>8} {'Hapax':>8} {'Entropy':>8} {'BestBPC':>8}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for lang in LANGUAGES:
        c = complexity.get(lang, {})
        cr = comparison["compression_results"].get(lang, {})

        ttr = c.get("type_token_ratio", 0)
        avg_len = c.get("avg_word_length_chars", 0)
        hapax = c.get("hapax_ratio", 0)
        entropy = c.get("char_entropy", 0)
        bpc = cr.get("best_classical_bpc", 0) or 0

        print(f"  {lang:<12} {ttr:>8.4f} {avg_len:>8.2f} {hapax:>8.4f} "
              f"{entropy:>8.4f} {bpc:>8.3f}")

    # Step 5: Analysis narrative
    print(f"\n  --- Key Findings ---")

    if len(complexity) >= 2:
        # Sort by TTR (proxy for morphological complexity)
        sorted_langs = sorted(complexity.keys(),
                              key=lambda l: complexity[l]["type_token_ratio"])
        print(f"  Morphological complexity ranking (by TTR): "
              f"{' < '.join(l.title() for l in sorted_langs)}")

        # Check if this correlates with compression difficulty
        comp_results = comparison["compression_results"]
        if all(l in comp_results and comp_results[l].get("best_classical_bpc")
               for l in sorted_langs):
            bpc_ranking = sorted(comp_results.keys(),
                                 key=lambda l: comp_results[l]["best_classical_bpc"])
            print(f"  Compression difficulty ranking (by BPC): "
                  f"{' < '.join(l.title() for l in bpc_ranking)}")

            # Correlation check
            if sorted_langs == bpc_ranking:
                print(f"  → Positive correlation: higher morphological complexity "
                      f"= higher BPC (harder to compress)")
            else:
                print(f"  → Rankings differ — resource availability or other factors "
                      f"may dominate")

    # Save full analysis
    analysis_file = RESULTS_DIR / "cross_lingual_analysis.json"
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Analysis saved: {analysis_file}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 5: Cross-lingual analysis")
    parser.add_argument(
        "--seed-variance", nargs="+", default=None, metavar="LABEL=PATH",
        help="Report mean/std BPC across multiple seeded runs instead of the "
             "normal cross-lingual analysis. Pass seed_label=path pairs, e.g.: "
             "  python -m pipeline.stage5_analysis --seed-variance "
             "seed1=results/hindi/compression_results_seed1.json "
             "seed2=results/hindi/compression_results_seed2.json "
             "seed3=results/hindi/compression_results_seed3.json",
    )
    args = parser.parse_args()

    if args.seed_variance:
        result_paths = {}
        for item in args.seed_variance:
            if "=" not in item:
                print(f"  ⚠ Skipping malformed --seed-variance entry (expected LABEL=PATH): {item}")
                continue
            label, path = item.split("=", 1)
            result_paths[label] = path
        report = compare_conditions_across_seeds(result_paths)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    run_analysis()

    print("\n✓ Stage 5 (Cross-Lingual Analysis) complete.")


if __name__ == "__main__":
    main()