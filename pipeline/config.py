"""
Central configuration for the Devanagari compression pipeline.
All paths, targets, and hyperparameters live here.
"""

import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "Raw"
CLEAN_DIR = DATA_DIR / "cleaned"
SPLIT_DIR = DATA_DIR / "splits"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
EVAL_DIR = PROJECT_ROOT / "eval"

# ─── Languages ───────────────────────────────────────────────────────────────
LANGUAGES = ["hindi", "marathi", "sanskrit"]

# ─── Corpus Size Targets (bytes) ────────────────────────────────────────────
# Per the project plan: Hindi 100MB, Marathi 100MB, Sanskrit "best available, target 20–40MB"
CORPUS_TARGETS = {
    "hindi":    100 * 1024 * 1024,   # 100 MB
    "marathi":  100 * 1024 * 1024,   # 100 MB
    "sanskrit":  40 * 1024 * 1024,   #  40 MB (best-effort; may be less)
}

# ─── Raw Data Sources ────────────────────────────────────────────────────────
RAW_SOURCES = {
    "hindi": {
        "wiki_xml": RAW_DIR / "hindi" / "hiwiki-latest-pages-articles.xml" / "hiwiki-latest-pages-articles.xml",
    },
    "marathi": {
        "cc100":     RAW_DIR / "marathi" / "mr.txt" / "mr.txt",
        "l3cube":    RAW_DIR / "marathi" / "L3CubeMahaCorpus-news" / "content" / "sample_data" / "fulldataset_dedup.txt",
    },
    "sanskrit": {
        "wiki_xml":  RAW_DIR / "sanskrit" / "sawiki-latest-pages-articles.xml" / "sawiki-latest-pages-articles.xml",
        "gretil_dir": RAW_DIR / "sanskrit" / "1_sanskr" / "1_sanskr" / "tei",
    },
}

# ─── Cleaning ────────────────────────────────────────────────────────────────
# Unicode Devanagari block: U+0900 – U+097F  (base)
# Devanagari Extended: U+A8E0 – U+A8FF
# Vedic Extensions: U+1CD0 – U+1CFF
DEVANAGARI_RANGES = [
    (0x0900, 0x097F),  # Devanagari
    (0xA8E0, 0xA8FF),  # Devanagari Extended
    (0x1CD0, 0x1CFF),  # Vedic Extensions
]

# Minimum Devanagari character ratio for a line to be kept
MIN_DEVANAGARI_RATIO = 0.70

# Minimum line length (chars) after cleaning
MIN_LINE_LENGTH = 20

# ─── Train / Test Split ─────────────────────────────────────────────────────
TRAIN_RATIO = 0.95  # 95/5 document-level split

# ─── Tokenizer Hyperparameters ───────────────────────────────────────────────
SENTENCEPIECE_VOCAB_SIZE = 32_000
SENTENCEPIECE_CHARACTER_COVERAGE = 0.9999
SENTENCEPIECE_MODEL_TYPE = "bpe"  # or "unigram"

# ─── Compression ─────────────────────────────────────────────────────────────
# Context window for LLM during compression (in tokens)
LLM_CONTEXT_LENGTH = 512

# Batch size for LLM inference during compression
COMPRESSION_BATCH_SIZE = 1

# Model to use for LLM-driven compression
INDIC_LLM_MODEL = "ai4bharat/Airavata"  # or "sarvamai/sarvam-1"

# ─── Evaluation ──────────────────────────────────────────────────────────────
# Classical compressors to benchmark against
CLASSICAL_COMPRESSORS = ["gzip", "bzip2", "zstd", "lzma"]

# Number of test sentences for round-trip verification
ROUNDTRIP_TEST_SIZE = 500

# ─── Baseline Tokenizers (for comparison) ────────────────────────────────────
BASELINE_TOKENIZERS = [
    "openai-community/gpt2",
    "meta-llama/Llama-2-7b-hf",
    "google/gemma-2b",
    "ai4bharat/IndicBERTv2-MLM-only",
]

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE = LOGS_DIR / "pipeline.log"


def ensure_dirs():
    """Create all required directories."""
    for d in [CLEAN_DIR, SPLIT_DIR, TOKENIZER_DIR, MODEL_DIR,
              RESULTS_DIR, LOGS_DIR, EVAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for lang in LANGUAGES:
        (CLEAN_DIR / lang).mkdir(parents=True, exist_ok=True)
        (SPLIT_DIR / lang).mkdir(parents=True, exist_ok=True)
        (TOKENIZER_DIR / lang).mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / lang).mkdir(parents=True, exist_ok=True)
