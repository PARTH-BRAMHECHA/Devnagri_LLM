# Devnagri_LLM — Devanagari Compression Pipeline

Research pipeline studying how well a general-purpose Indic LLM (AI4Bharat's
Airavata) compresses Hindi, Marathi, and Sanskrit text, and whether a
script-aware ("DevAware") tokenizer — extended with multi-akshara merges and
fine-tuned into the model — improves on that.

The core question the pipeline is built to answer: **does a Devanagari-aware
tokenizer actually reduce bits-per-character on real text, and if so, is the
gain coming from the tokenizer itself or from the fine-tuning it requires?**

## Pipeline stages

```
pipeline/
  stage1_extract.py            Wikipedia dump → raw text extraction
  stage1_clean.py              Cleaning, dedup, script filtering
  stage1_split.py              train/dev/test splits per language
  stage2a_baselines.py         Classical compressor baselines (gzip/bz2/lzma/...)
  stage2b_devanagari_tokenizer.py   Builds the DevAware SentencePiece tokenizer
  stage2c_sandhi.py            Sandhi-aware analysis utilities
  stage2d_vocab_extend.py      Vocab extension + LoRA fine-tune of Airavata
  stage3_compress.py           Arithmetic-coding compression using the LLM
                                as the probability model; BPC computation
  stage4_benchmark.py          Assembles classical + LLM results into the
                                master per-language, per-text-type table
  stage5_analysis.py           Cross-lingual / morphological-complexity
                                analysis and the final summary report
run_pipeline.py                 CLI entrypoint wiring all stages together
```

`pipeline/config.py` is the single source of truth for paths, language list,
and every hyperparameter (LoRA rank, fine-tune LR/steps, corpus size caps,
etc.) — check there before editing a stage script directly.

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the packages that Kaggle's preinstalled defaults
silently conflict with (torchao, peft, bitsandbytes, protobuf — see the
comments in the file for the specific version-mismatch each pin avoids).
`torch`/`torchvision`/`torchaudio` are deliberately **not** pinned — install
whatever CUDA-matched build your environment already ships, since
reinstalling from PyPI here risks losing GPU support.

## Running

```bash
# Full pipeline, all languages
python run_pipeline.py --stage all

# One stage, one language
python run_pipeline.py --stage 1 --lang hindi
python run_pipeline.py --stage 2 --lang marathi

# Benchmarks without loading the LLM (fast, classical compressors only)
python run_pipeline.py --stage 4 --classical-only

# Stage 2d (vocab extension + fine-tune) is expensive and opt-in — NOT
# included in --stage all. Run it explicitly per language, then pass
# --use-devaware-tokenizer to Stage 3 to report the "your tokenizer"
# condition (and its ablation cells — see below).
python run_pipeline.py --stage 2d --lang hindi
python run_pipeline.py --stage 3 --lang hindi --use-devaware-tokenizer
```

`--verify` adds a lossless round-trip check on top of Stage 3's BPC pass
(~3x model calls) — off by default; it's what caused a multi-hour hang on
Kaggle previously.

## What's in the compression results

Stage 3 (`compression_results.json`) reports up to **four** LLM conditions
per language, forming a 2×2 grid of *tokenizer* × *fine-tuning*:

| | Default tokenizer | DevAware tokenizer |
|---|---|---|
| **Base model (not fine-tuned)** | `llm_compression` | `llm_compression_base_devaware_tokenizer` |
| **Fine-tuned** | `llm_compression_finetuned_default_tokenizer` | `llm_compression_devaware_tokenizer` |

The two off-diagonal cells (`base_devaware_tokenizer` and
`finetuned_default_tokenizer`) exist so a BPC gain in the bottom-right cell
can be attributed to the tokenizer, the fine-tuning, or both, instead of
being reported as one conflated number. All four are computed against the
**same shared base model** — sequentially, with embeddings resized and
restored between conditions — so a single GPU never needs to hold more than
one full model's worth of weights at once.

Stage 4 (`benchmark_results.json`) additionally reports an
**`llm_generated`** text-type condition alongside `pure_language`: BPC on
text the model generates itself, not just on real (human-written) test
text. Expect this number to be *lower* than `pure_language` — that mostly
reflects the model finding its own output easy to predict, not a
language-modeling improvement, and is reported as exactly that rather than
as a win.

## Stage 2d fine-tuning: reading the logs

Each language's checkpoint directory (`FINETUNE_DIR/<lang>/final/`)
contains a `training_log.json` with both train loss **and held-out
eval loss**, logged every `FINETUNE_EVAL_EVERY` steps against a slice
carved off the end of the training corpus (`FINETUNE_EVAL_HOLDOUT_CHARS`,
never trained on). Train loss alone can't distinguish "still converging,"
"converged, this is just batch variance," and "overfitting" — the eval-loss
curve can.

Current defaults (`pipeline/config.py`):

```python
FINETUNE_LR = 5e-5              # was 2e-4 — see note below
FINETUNE_WARMUP_STEPS = 100     # was 50
FINETUNE_STEPS = 1000
FINETUNE_EVAL_HOLDOUT_CHARS = 200_000
FINETUNE_EVAL_EVERY = 100
```

`FINETUNE_LR` was lowered from `2e-4`: that value is reasonable for
LoRA-only updates, but this fine-tune also fully updates `embed_tokens` and
`lm_head` (not just LoRA deltas), and 2e-4 was aggressive enough there to
produce a loss that bounced between ~2.6 and ~4.6 late in training instead
of settling — a stability issue, not an undertraining issue. If you still
see instability at 5e-5, raise `FINETUNE_WARMUP_STEPS` further before
raising the LR back up.

## Known constraints

- Stage 2d honors `FINETUNE_MAX_WALL_SECONDS` (default 3h/language) and
  checkpoints + exits cleanly if hit — re-running the same command resumes
  from the last checkpoint rather than starting over.
- Stage 3/4 assume a single GPU; the shared-base-model pattern in
  `run_pipeline.py` (`_load_devaware_compressor`,
  `_load_base_devaware_compressor`) exists specifically to avoid holding
  two 7B models in memory at once.
- `llm_generated` text generation in Stage 4 is autoregressive and slow —
  it's cached to `RESULTS_DIR/<lang>/llm_generated.txt` after the first run
  so repeated Stage 4 runs don't regenerate.

## Output layout

```
results/<lang>/compression_results.json    Stage 3 output (the 2x2 grid)
results/<lang>/benchmark_results.json      Stage 4 output (per text-type)
results/<lang>/llm_generated.txt           Cached LLM-generated text
results/results_report.md                  Stage 5 final summary
```
