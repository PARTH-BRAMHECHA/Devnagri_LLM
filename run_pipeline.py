"""
Master Pipeline Runner
=======================
Orchestrates all stages of the Devanagari compression pipeline.

Usage:
    python run_pipeline.py --stage all          # Run everything
    python run_pipeline.py --stage 1            # Stage 1 only (extract → clean → split)
    python run_pipeline.py --stage 2            # Stage 2 only (tokenizers)
    python run_pipeline.py --stage 3            # Stage 3 only (compression)
    python run_pipeline.py --stage 4            # Stage 4 only (benchmarking)
    python run_pipeline.py --stage 5            # Stage 5 only (cross-lingual analysis)
    python run_pipeline.py --stage 1 --lang hindi  # Stage 1, Hindi only
    python run_pipeline.py --classical-only     # Skip LLM (stages 3-4 classical only)
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import LANGUAGES, ensure_dirs


def run_stage_1(lang: str = "all"):
    """Stage 1: Dataset Curation (Extract → Clean → Split)"""
    from pipeline.stage1_extract import EXTRACTORS, main as extract_main
    from pipeline.stage1_clean import clean_corpus
    from pipeline.stage1_split import split_corpus

    langs = LANGUAGES if lang == "all" else [lang]

    print("\n" + "=" * 70)
    print("  STAGE 1a: EXTRACTION")
    print("=" * 70)
    for l in langs:
        print(f"\n--- Extracting: {l.upper()} ---")
        EXTRACTORS[l]()

    print("\n" + "=" * 70)
    print("  STAGE 1b: CLEANING")
    print("=" * 70)
    for l in langs:
        print(f"\n--- Cleaning: {l.upper()} ---")
        clean_corpus(l)

    print("\n" + "=" * 70)
    print("  STAGE 1c: TRAIN/TEST SPLIT")
    print("=" * 70)
    for l in langs:
        print(f"\n--- Splitting: {l.upper()} ---")
        split_corpus(l)


def run_stage_2(lang: str = "all"):
    """Stage 2: Tokenizer Development"""
    from pipeline.stage2a_baselines import run_baselines
    from pipeline.stage2b_devanagari_tokenizer import (
        train_devanagari_aware_bpe, evaluate_devanagari_tokenizer,
    )
    from pipeline.stage2c_sandhi import train_sandhi_variant, evaluate_sandhi_variants

    langs = LANGUAGES if lang == "all" else [lang]

    print("\n" + "=" * 70)
    print("  STAGE 2a: BASELINE TOKENIZERS")
    print("=" * 70)
    for l in langs:
        run_baselines(l)

    print("\n" + "=" * 70)
    print("  STAGE 2b: DEVANAGARI-AWARE TOKENIZER")
    print("=" * 70)
    for l in langs:
        model_path = train_devanagari_aware_bpe(l)
        if model_path:
            evaluate_devanagari_tokenizer(model_path, l)

    print("\n" + "=" * 70)
    print("  STAGE 2c: SANSKRIT SANDHI MODULE")
    print("=" * 70)
    if lang == "all" or lang == "sanskrit":
        train_sandhi_variant()
        evaluate_sandhi_variants()
    else:
        print("  (Skipped — sandhi module is Sanskrit-only)")


def run_stage_3(lang: str = "all", classical_only: bool = False):
    """Stage 3: Compression Pipeline"""
    from pipeline.stage3_compress import run_compression, LLMCompressor
    from pipeline.config import INDIC_LLM_MODEL

    langs = LANGUAGES if lang == "all" else [lang]

    print("\n" + "=" * 70)
    print("  STAGE 3: COMPRESSION PIPELINE")
    print("=" * 70)

    # Load the LLM once and reuse it across every language instead of
    # reloading a 7B model from disk per language (previously: 3 loads in
    # stage 3, another 3 in stage 4 -- 6 total for one pipeline run).
    shared_compressor = None
    if not classical_only:
        shared_compressor = LLMCompressor(INDIC_LLM_MODEL)

    for l in langs:
        run_compression(l, verify=True, classical_only=classical_only,
                         compressor=shared_compressor)

    return shared_compressor


def run_stage_4(lang: str = "all", classical_only: bool = False, compressor=None):
    """Stage 4: Benchmarking"""
    from pipeline.stage4_benchmark import run_benchmark, generate_master_table

    langs = LANGUAGES if lang == "all" else [lang]

    print("\n" + "=" * 70)
    print("  STAGE 4: BENCHMARKING")
    print("=" * 70)
    for l in langs:
        # run_benchmark first tries to reuse Stage 3's saved LLM result for
        # this language (same metric, no recompute needed); `compressor` is
        # only used as a fallback if that's not available.
        run_benchmark(l, classical_only=classical_only, compressor=compressor)

    generate_master_table()


def run_stage_5():
    """Stage 5: Cross-Lingual Analysis"""
    from pipeline.stage5_analysis import run_analysis
    from pipeline.eval_utils import generate_corpus_report, format_results_table
    from pipeline.config import RESULTS_DIR

    print("\n" + "=" * 70)
    print("  STAGE 5: CROSS-LINGUAL ANALYSIS")
    print("=" * 70)
    run_analysis()

    print("\n" + "=" * 70)
    print("  CORPUS SIZE REPORT")
    print("=" * 70)
    generate_corpus_report()

    print("\n" + "=" * 70)
    print("  GENERATING RESULTS REPORT")
    print("=" * 70)
    table = format_results_table()
    output = RESULTS_DIR / "results_report.md"
    with open(output, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"  Report saved to: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Devanagari Compression Pipeline — Master Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --stage all                # Full pipeline
  python run_pipeline.py --stage 1                  # Data curation only
  python run_pipeline.py --stage 1 --lang hindi     # Hindi data only
  python run_pipeline.py --stage 2 --lang marathi   # Marathi tokenizers
  python run_pipeline.py --stage 4 --classical-only # Benchmarks without LLM
        """,
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "3", "4", "5", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--lang",
        choices=LANGUAGES + ["all"],
        default="all",
        help="Language to process (default: all)",
    )
    parser.add_argument(
        "--classical-only",
        action="store_true",
        help="Skip LLM-based compression (use only classical compressors)",
    )
    args = parser.parse_args()

    ensure_dirs()
    start = time.time()

    stages_to_run = (
        ["1", "2", "3", "4", "5"] if args.stage == "all" else [args.stage]
    )

    # Carries the LLM loaded in Stage 3 (if it ran) through to Stage 4 in
    # the same process, so Stage 4 never needs to reload the 7B model.
    llm_compressor = None

    for stage in stages_to_run:
        if stage == "1":
            run_stage_1(args.lang)
        elif stage == "2":
            run_stage_2(args.lang)
        elif stage == "3":
            llm_compressor = run_stage_3(args.lang, args.classical_only)
        elif stage == "4":
            run_stage_4(args.lang, args.classical_only, compressor=llm_compressor)
        elif stage == "5":
            run_stage_5()

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE — {elapsed/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()