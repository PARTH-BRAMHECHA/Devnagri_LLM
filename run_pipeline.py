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
import gc
import os
import sys
import time
from pathlib import Path

# Reduce CUDA allocator fragmentation -- must be set before the first CUDA
# allocation, so at import time, not inside a function. (stage2d_vocab_extend.py
# already sets this for its own process; setting it here too covers runs
# that hit Stage 3 without ever importing stage2d.)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

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


def run_stage_2d(lang: str = "all"):
    """Stage 2d: Vocabulary Extension + LoRA Fine-tune (optional, expensive).

    Not part of `--stage all` by default -- this trains an adapter per
    language and takes real GPU time. Run it explicitly:
        python run_pipeline.py --stage 2d --lang hindi
    then pass --use-devaware-tokenizer to Stage 3 to actually use the
    result for the "your tokenizer" compression condition.
    """
    from pipeline.stage2d_vocab_extend import build_vocab_extended_model

    langs = LANGUAGES if lang == "all" else [lang]

    print("\n" + "=" * 70)
    print("  STAGE 2d: VOCABULARY EXTENSION + FINE-TUNE")
    print("=" * 70)
    for l in langs:
        build_vocab_extended_model(l)


def run_stage_3(lang: str = "all", classical_only: bool = False, verify: bool = False,
                 bootstrap_ci: bool = False,
                 use_devaware_tokenizer: bool = False):
    """Stage 3: Compression Pipeline

    NOTE: `verify` defaults to False. Round-trip verification calls
    LLMCompressor.compress() *and* .decompress() on top of the normal
    compute_bpc() pass -- effectively 3x the model forward passes -- and
    on a 7B model with no KV reuse in the old code, this is what turned
    a Kaggle run into an 8+ hour hang stuck at "3/252 tokens". It's a
    one-time correctness check, not something that needs to run on every
    language on every pipeline run. Pass --verify explicitly (or
    verify=True here) to opt back in.

    `use_devaware_tokenizer`: if True, also loads each language's Stage 2d
    checkpoint (must already exist -- run `--stage 2d` first) and reports
    a genuine "your tokenizer" compression condition alongside the model
    default. If a language has no Stage 2d checkpoint yet, that language
    falls back to the two-condition table with a printed warning, rather
    than failing the whole run.
    """
    from pipeline.stage3_compress import run_compression, load_test_data, LLMCompressor
    from pipeline.config import INDIC_LLM_MODEL

    langs = LANGUAGES if lang == "all" else [lang]

    print("\n" + "=" * 70)
    print("  STAGE 3: COMPRESSION PIPELINE")
    if verify:
        print("  (round-trip verification ENABLED — this triples model calls)")
    if use_devaware_tokenizer:
        print("  (DevAware tokenizer condition ENABLED — requires Stage 2d checkpoints)")
    print("=" * 70)

    # Load the LLM once and reuse it across every language instead of
    # reloading a 7B model from disk per language (previously: 3 loads in
    # stage 3, another 3 in stage 4 -- 6 total for one pipeline run).
    shared_compressor = None
    base_vocab_size = None
    if not classical_only:
        # quantize_4bit=True only when the DevAware adapter will also need
        # to live on the GPU alongside this base model -- a full bf16 7B
        # model alone already uses ~13.8 GiB of a 14.56 GiB Kaggle T4,
        # leaving no room for the adapter/resized-embedding load and
        # causing the Stage 3 CUDA OOM at PeftModel.from_pretrained(). See
        # LLMCompressor.__init__'s quantize_4bit docstring for the full
        # explanation. Plain Stage 3/4 runs (no --use-devaware-tokenizer)
        # are unaffected and still load at full precision.
        shared_compressor = LLMCompressor(
            INDIC_LLM_MODEL, quantize_4bit=use_devaware_tokenizer
        )
        # Record the clean vocab size BEFORE any devaware attach ever
        # resizes the shared model's embeddings, so detach can restore it.
        base_vocab_size = shared_compressor.model.get_input_embeddings().weight.shape[0]

    for l in langs:
        devaware_compressor = None
        base_devaware_compressor = None
        precomputed_llm_result = None
        precomputed_base_devaware_result = None

        if not classical_only:
            # Capture the clean "no fine-tune, model-default-tokenizer"
            # baseline BEFORE anything below ever mutates
            # shared_compressor.model in place (vocab-extend or LoRA
            # merge). Passing this in as `precomputed_llm_result` is not
            # an optimization -- it's the fix for the exact bug that
            # produced a "fine-tuned+default-tokenizer" row identical to
            # this baseline to 4+ significant figures (see
            # stage3_compress._warn_if_suspiciously_identical and
            # run_compression's precomputed_llm_result docstring). If we
            # let run_compression call compressor.compute_bpc() itself
            # after devaware_compressor has already been attached below,
            # it silently measures the fine-tuned model under the
            # "no fine-tune" label.
            _, test_sample = load_test_data(l)
            precomputed_llm_result = shared_compressor.compute_bpc(test_sample)

        if use_devaware_tokenizer and not classical_only:
            # Order matters here and mirrors the fix above: extend the
            # vocab (untrained) and measure THAT before ever LoRA-merging
            # fine-tuned weights onto the same model object, or the
            # "tokenizer only, not fine-tuned" ablation would silently
            # measure the fine-tuned model too -- the identical bug one
            # hop later. See _load_base_devaware_compressor.
            base_devaware_compressor = _load_base_devaware_compressor(l, shared_compressor)
            if base_devaware_compressor is not None:
                precomputed_base_devaware_result = base_devaware_compressor.compute_bpc(test_sample)

            devaware_compressor = _load_devaware_compressor(l, shared_compressor)

        run_compression(l, verify=verify, classical_only=classical_only,
                         compressor=shared_compressor,
                         devaware_compressor=devaware_compressor,
                         base_devaware_compressor=base_devaware_compressor,
                         precomputed_llm_result=precomputed_llm_result,
                         precomputed_base_devaware_result=precomputed_base_devaware_result,
                         bootstrap_ci=bootstrap_ci)

        # Detach the adapter / restore the shared base model's original
        # embeddings before the next language, so we never hold two full
        # 7B models (or stacked vocab extensions) on the GPU at once.
        if devaware_compressor is not None:
            from pipeline.stage2d_vocab_extend import detach_devaware_adapter
            restored_base = detach_devaware_adapter(
                devaware_compressor.model, shared_compressor.tokenizer, base_vocab_size
            )
            shared_compressor.model = restored_base
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        elif base_devaware_compressor is not None:
            # Fine-tuned checkpoint wasn't available (see
            # _load_devaware_compressor's FileNotFoundError branch) but
            # the base vocab-extension still happened and needs undoing.
            from pipeline.stage2d_vocab_extend import detach_devaware_adapter
            restored_base = detach_devaware_adapter(
                base_devaware_compressor.model, shared_compressor.tokenizer, base_vocab_size
            )
            shared_compressor.model = restored_base
            gc.collect()
            import torch
            torch.cuda.empty_cache()

    return shared_compressor


def _load_devaware_compressor(lang: str, shared_compressor):
    """Attach Stage 2d's fine-tuned adapter for `lang` onto the ALREADY
    loaded shared_compressor's base model (avoids a second full 7B model
    on the GPU, which was the actual cause of the Stage 3 CUDA OOM), or
    return None (with a warning) if no checkpoint is available."""
    from pipeline.stage2d_vocab_extend import load_finetuned_devaware_model
    from pipeline.stage3_compress import LLMCompressor
    from pipeline.config import INDIC_LLM_MODEL

    try:
        model, tokenizer = load_finetuned_devaware_model(
            lang,
            base_model=shared_compressor.model if shared_compressor is not None else None,
        )
    except FileNotFoundError as e:
        print(f"  ⚠ {e}")
        print(f"  ⚠ Skipping 'your tokenizer' condition for {lang} "
              f"(falling back to model-default / classical only).")
        return None

    return LLMCompressor(INDIC_LLM_MODEL, tokenizer=tokenizer, model=model)


def _load_base_devaware_compressor(lang: str, shared_compressor):
    """Build the 'devaware tokenizer, base model, NOT fine-tuned' ablation
    condition: extends the ALREADY loaded shared_compressor's model with
    DevAware's novel merged tokens and smart-initializes the new embedding
    rows, but stops there -- no LoRA wrap, no fine-tuning. Together with
    _load_devaware_compressor's fine-tuned result, this completes the 2x2
    ablation grid (tokenizer x fine-tuning) needed to tell whether a BPC
    gain comes from the tokenizer itself or from fine-tuning; see
    extend_tokenizer_and_smart_init's docstring.

    Mutates shared_compressor.model in place (same reasoning as
    _load_devaware_compressor -- a second full 7B copy doesn't fit on a
    14.56 GiB T4 alongside the first). The caller MUST measure this
    condition's BPC before calling _load_devaware_compressor for the same
    language, since that call LoRA-merges fine-tuned weights onto this
    same (now vocab-extended) model object -- measuring after that point
    would silently score the fine-tuned model under the "not fine-tuned"
    label. See run_stage_3's precomputed_base_devaware_result capture.

    Returns None (with a printed warning) if this language's DevAware
    tokenizer has no novel merges to add (mirrors
    build_vocab_extended_model's own "nothing to fine-tune" case) or if
    Stage 2b hasn't been run for this language yet.
    """
    from transformers import AutoTokenizer
    from pipeline.stage2d_vocab_extend import extend_tokenizer_and_smart_init
    from pipeline.devaware_tokenizer import DevAwareTokenizer
    from pipeline.stage3_compress import LLMCompressor
    from pipeline.config import INDIC_LLM_MODEL, TOKENIZER_DIR

    if shared_compressor is None:
        return None

    tok_dir = TOKENIZER_DIR / lang
    spm_path = tok_dir / f"devaware_bpe_{lang}.model"
    pua_path = tok_dir / f"akshara_pua_map_{lang}.json"
    if not spm_path.exists() or not pua_path.exists():
        print(f"  ⚠ No DevAware tokenizer found at {tok_dir} for {lang} "
              f"(run Stage 2b first). Skipping the 'tokenizer only, not "
              f"fine-tuned' ablation for {lang}.")
        return None

    dev_tok = DevAwareTokenizer(spm_model_path=spm_path, pua_map_path=pua_path)
    # Fresh tokenizer object (not shared_compressor.tokenizer) so the
    # in-place add_tokens() below never mutates the clean tokenizer that
    # precomputed_llm_result / the finetuned_default ablation both rely on.
    base_tokenizer = AutoTokenizer.from_pretrained(INDIC_LLM_MODEL, trust_remote_code=True)

    model, extended_tokenizer, new_tokens = extend_tokenizer_and_smart_init(
        shared_compressor.model, base_tokenizer, dev_tok
    )
    if not new_tokens:
        print(f"  ⚠ DevAware tokenizer adds nothing new for {lang} -- "
              f"skipping the 'tokenizer only, not fine-tuned' ablation "
              f"(it would be identical to model_default).")
        return None

    shared_compressor.model = model
    return LLMCompressor(INDIC_LLM_MODEL, tokenizer=extended_tokenizer, model=model)


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
        choices=["1", "2", "2d", "3", "4", "5", "all"],
        default="all",
        help="Pipeline stage to run (default: all). '2d' (vocab extension + "
             "fine-tune) is NOT included in 'all' -- it's expensive and "
             "opt-in; run it explicitly, then pass --use-devaware-tokenizer "
             "to Stage 3.",
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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run lossless round-trip verification in Stage 3 (compress+decompress "
             "on top of the normal BPC pass, ~3x model calls). Off by default -- "
             "this is what caused the multi-hour Kaggle hang.",
    )
    parser.add_argument(
        "--use-devaware-tokenizer",
        action="store_true",
        help="In Stage 3, also report the 'your tokenizer' compression condition "
             "using Stage 2d's vocab-extended, fine-tuned model. Requires "
             "`--stage 2d` to have been run for the target language(s) first.",
    )
    parser.add_argument(
        "--bootstrap-ci",
        action="store_true",
        help="In Stage 3, attach a bootstrap CI on bits-per-token to each LLM "
             "condition (from the same forward passes -- no extra model calls). "
             "Complements multi-seed variance (config.SEED): quantifies "
             "test-set uncertainty at a fixed seed, not across fine-tune runs.",
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
        elif stage == "2d":
            run_stage_2d(args.lang)
        elif stage == "3":
            llm_compressor = run_stage_3(
                args.lang, args.classical_only, args.verify,
                bootstrap_ci=args.bootstrap_ci,
                use_devaware_tokenizer=args.use_devaware_tokenizer,
            )
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
