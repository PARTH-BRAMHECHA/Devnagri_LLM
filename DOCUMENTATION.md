# Devnagri_LLM — Pipeline Documentation & Results

Repo: https://github.com/PARTH-BRAMHECHA/Devnagri_LLM

This document explains (1) what the codebase does, stage by stage, and (2)
what the Hindi results say so far, including what's solid and what still
needs work. Marathi and Sanskrit are out of scope for now — the pipeline
supports all three languages structurally, but only Hindi has been run
end-to-end.

---

## 1. What this project is

The core question: **does a tokenizer that respects Devanagari script
structure (grapheme clusters / "aksharas") actually help an LLM compress
and understand Hindi text better than the model's own default tokenizer?**

Two more questions sit underneath that one, and the pipeline is built to
answer all three with a real number, not an assumption:

1. Is a Devanagari-aware tokenizer intrinsically more efficient (fewer
   tokens, doesn't split vowel marks off their consonants) than existing
   tokenizers — SentencePiece baselines, GPT-2, LLaMA-2, Gemma, IndicBERT,
   Sarvam, OpenHathi, Airavata's own, Aya-101?
2. If you actually plug that tokenizer into a real LLM (not just measure
   it standalone), does the model compress held-out Hindi text better —
   and is that a real effect or noise?
3. Does that intrinsic improvement translate into anything a downstream
   task or a human reader would notice?

The answer to (1) is a clear yes. The answer to (2) is a qualified yes —
solid at one seed, not yet re-verified across seeds. The answer to (3) is
"generation quality: modestly yes; downstream classification: no evidence
either way yet, and here's why."

---

## 2. Repository structure

```
Devnagri_LLM/
├── run_pipeline.py              # orchestrator — runs stages 1 through 5 in order
├── requirements.txt
├── README.md / KAGGLE_RUN_GUIDE.md
├── data/
│   ├── Raw/<lang>/              # downloaded source dumps (Wikipedia XML, CC100, GRETIL...)
│   ├── cleaned/<lang>/          # after Stage 1b
│   └── splits/<lang>/           # train.txt / test.txt, after Stage 1c
├── tokenizer/<lang>/            # trained SentencePiece models + PUA maps (Stage 2)
├── models/                      # (persistent, symlinked on Kaggle) fine-tuned checkpoints
├── results/<lang>/              # every stage's JSON output lands here
└── pipeline/
    ├── config.py                 # all paths, hyperparameters, language/model constants
    ├── stage1_extract.py         # 1a: raw dump -> plain text
    ├── stage1_clean.py           # 1b: normalize, filter, dedup
    ├── stage1_split.py           # 1c: train/test split
    ├── stage2a_baselines.py      # baseline tokenizer comparison
    ├── stage2b_devanagari_tokenizer.py   # trains the DevAware tokenizer
    ├── pua_remap.py               # akshara <-> Private-Use-Area codepoint mapping
    ├── devaware_tokenizer.py      # runtime wrapper around the trained DevAware model
    ├── stage2c_sandhi.py          # Sanskrit sandhi-splitting module (Sanskrit only)
    ├── stage2d_vocab_extend.py    # vocab extension + LoRA fine-tune onto Airavata
    ├── stage3_compress.py         # classical + LLM compression measurement (BPC)
    ├── eval_utils.py              # shared metrics: entropy, bootstrap CI, table formatting
    ├── stage4_benchmark.py        # master results table across text types
    ├── stage5_analysis.py         # cross-lingual analysis (needs 2+ languages — not run yet)
    ├── stage6_downstream.py       # XNLI classification probe (downstream task validation)
    ├── human_eval.py              # blinded human-rating sheet generator/aggregator
    └── error_diagnostics.py       # traceback + CUDA-OOM diagnostics wrapper
```

---

## 3. Pipeline, stage by stage

### Stage 1 — Data preparation

**1a — `stage1_extract.py`.** Pulls raw text out of each language's source
format: Hindi and Sanskrit come from MediaWiki XML dumps (streamed with
`iterparse` so a multi-GB dump never has to fit in memory at once);
Marathi comes from CC100 + L3Cube plain-text corpora, merged. Output:
raw plain text per language, capped at each language's `CORPUS_TARGETS`
byte budget (Hindi/Marathi 100MB, Sanskrit 40MB best-effort).

**1b — `stage1_clean.py`.** NFC-normalizes the text, strips boilerplate,
deduplicates, and filters out lines that aren't mostly Devanagari
(`MIN_DEVANAGARI_RATIO = 0.70`) or are too short. Also truncates any
single line over 20,000 characters before regex-processing it — a
guard against multi-KB merged-table/infobox artifacts from the XML
extraction stalling the cleaning pass.

**1c — `stage1_split.py`.** Splits into train/test at the **document**
level (paragraphs separated by blank lines), not the sentence level —
this avoids leaking near-duplicate sentences from the same document
across the train/test boundary, which would make test-set BPC look
better than it really is.

### Stage 2 — Tokenizers

**2a — `stage2a_baselines.py`.** Trains per-language SentencePiece
BPE/Unigram tokenizers from scratch, and separately loads and evaluates
8 existing tokenizers on the same corpus: GPT-2, LLaMA-2, Gemma, IndicBERT,
Sarvam, OpenHathi, Airavata's own tokenizer, and Aya-101. For each, it
reports tokens/sentence and **vowel-split %** — the percentage of the
time a dependent vowel mark (matra) ends up as a separate token from the
consonant it modifies, which is linguistically wrong for Devanagari (a
matra isn't a standalone sound) and inflates token count. Gated/missing
HF models fail gracefully into a per-tokenizer `error` field rather than
crashing the whole comparison.

**2b — `stage2b_devanagari_tokenizer.py`, `pua_remap.py`,
`devaware_tokenizer.py`.** This is the actual contribution. The idea:
Devanagari is a grapheme-cluster script — a consonant, an optional
dependent vowel mark, and an optional virama together form one
orthographic syllable ("akshara") that should never be split mid-cluster
by a subword tokenizer. `stage2b` segments text into aksharas as a
pre-tokenization step before BPE training.

The `pua_remap.py` file documents (and fixes) a real bug that shipped in
an earlier version: aksharas were joined with literal spaces before
SentencePiece training, and SentencePiece's `split_by_whitespace=True`
then treated every akshara as a hard token boundary — which stopped BPE
from ever merging two aksharas into one subword, making the
"Devanagari-aware" tokenizer worse than the plain baseline (0.502 vs
0.254 tokens/char). The fix: each distinct akshara is collapsed to a
single Private-Use-Area Unicode codepoint instead of being
space-separated. A PUA codepoint is atomic, so BPE still can't split
inside an akshara, but *can* merge adjacent PUA codepoints into a single
token — which is what actually lets the tokenizer learn genuine
multi-akshara subword units. `devaware_tokenizer.py` wraps this whole
segment → PUA-remap → SentencePiece pipeline (and its inverse for
decoding) behind an ordinary tokenizer-shaped interface so downstream
code doesn't need to know about any of this.

**2c — `stage2c_sandhi.py`** (Sanskrit only, not yet run). Rule-based
sandhi splitting (vowel, consonant, visarga sandhi) as a preprocessing
toggle, producing tokenizer variants with and without sandhi splitting.

**2d — `stage2d_vocab_extend.py`.** This is the piece that makes Stage 3's
"your tokenizer" condition real instead of aspirational. You cannot just
swap a pretrained LM's tokenizer for an unrelated one — the LM head was
trained against its own vocabulary's embeddings, and a different
tokenizer's token ids are meaningless to it. Two honest options exist:
report the DevAware tokenizer as an intrinsic metric only (Stage 2a/2b's
numbers), or actually adapt the model. This file does the latter:

1. **Vocabulary extension, not replacement.** Pulls out only the
   genuinely novel multi-akshara merges the DevAware tokenizer can
   produce that Airavata's own tokenizer can't already express as one
   token, and adds those as new tokens on top of the existing vocabulary
   — old text still tokenizes exactly as before.
2. **Smart embedding init.** Each new token's embedding starts as the
   mean of Airavata's embeddings for the sub-tokens it used to take to
   spell the same string (the standard vocab-extension trick), instead of
   random init.
3. **QLoRA + full embedding fine-tune.** The frozen backbone loads in
   4-bit (NF4, double-quant) via bitsandbytes to leave GPU headroom for
   the trainable pieces (LoRA adapters + fully fine-tuned
   `embed_tokens`/`lm_head`) on a 16GB Kaggle T4. `lm_head` is excluded
   from quantization since it needs a real (not 4-bit) gradient.
4. **Continued pretraining** (causal LM loss) on the language's own
   corpus with the extended tokenizer, bounded by a hard wall-clock
   budget so it always checkpoints before a Kaggle session gets killed —
   re-running the same command resumes automatically.

`load_finetuned_devaware_model()` in this file is the shared loader every
later stage (3, 6, human-eval) uses to get back a `(model, tokenizer)`
pair from a finished Stage 2d checkpoint. It has two load paths: reuse an
already-loaded base model in place (Stage 3's shared compressor, to avoid
holding two 7B copies on a 16GB GPU at once), or a fresh full-precision
load (Stage 6/human-eval, which run as their own standalone process). The
function ends by casting the whole model to one consistent dtype
(bf16/fp16/fp32 depending on device) before returning — this was
originally missing and caused a `float != bfloat16` crash in Stage 6 (see
§5, "bugs found and fixed").

### Stage 3 — Compression measurement (`stage3_compress.py`, `eval_utils.py`)

Measures **bits-per-character (BPC)** — how many bits it would take to
encode the held-out test text — for every condition:

- **Classical compressors**: gzip, bzip2, lzma, zstd (and gzip+bzip2 /
  gzip+lzma double-compression variants), as a non-neural reference
  point.
- **LLM, model's default tokenizer, no fine-tune** — Airavata as
  shipped.
- **LLM, DevAware tokenizer, fine-tuned** (Stage 2d's output) — the
  "your tokenizer" condition.
- **LLM, DevAware tokenizer, base model NOT fine-tuned** — vocabulary
  extended and smart-initialized, but never trained. This exists
  specifically so the *tokenizer* effect and the *fine-tuning* effect
  can be told apart instead of conflated into one number.
- **LLM, fine-tuned model, default tokenizer** — the fourth cell of the
  2×2 (tokenizer × fine-tuning) ablation grid, so a BPC improvement can
  be attributed correctly.

BPC is computed by summing per-token cross-entropy under the model
(uniform-probability charged for the very first token, since there's no
context to predict it from) across sliding windows over the test set, then
dividing by character count. A `--bootstrap-ci` flag (added after the
original single-point-estimate version) resamples the forward-pass
windows already computed — no extra model calls — to attach a 95%
confidence interval on bits-per-token to each LLM condition, quantifying
how much that number could plausibly move on a different draw of the
same test set at a fixed seed. This is a different (complementary) kind
of uncertainty from `config.SEED`'s multi-seed fine-tuning-run variance,
which the codebase also supports.

`eval_utils.py` holds the shared pieces: character/byte entropy, the
`bootstrap_bpc_ci` function above, and results-table formatting.

### Stage 4 — Benchmarking (`stage4_benchmark.py`)

Builds the master results table (language × text-type rows, method
columns). Reuses Stage 3's already-computed `compression_results.json`
for the "real corpus" row instead of reloading the 7B model and
recomputing the same measurement a second time (an earlier version did
exactly that, roughly doubling total runtime for no new information).
Also computes BPC on **LLM-generated text** as a separate condition —
important caveat, explicit in the code and worth repeating here: a low
BPC on the model's *own* generated text mostly reflects the model
finding its own output predictable, not a genuine compression win, and
should never be read as validating anything about generation quality.

### Stage 5 — Cross-lingual analysis (`stage5_analysis.py`)

Not run yet — needs results from at least two languages (relates
morphological complexity / resource availability to compression
performance across Hindi, Marathi, Sanskrit). Skipped automatically by
`run_pipeline.py` when only one language has results.

### Stage 6 — Downstream task eval (`stage6_downstream.py`)

Added specifically to address the "BPC is only an intrinsic metric"
gap. Runs a **frozen-representation linear probe**: encode XNLI-Hindi
(3-way natural language inference) premise/hypothesis pairs through the
Stage 2d fine-tuned model, pool the final hidden state into one feature
vector, and train an `sklearn` `LogisticRegression` on top — once with
the DevAware tokenizer, once with the default tokenizer, **same
fine-tuned model weights both times**, so any accuracy difference is
attributable to the tokenizer alone, mirroring how Stage 3 isolates the
tokenizer effect from the fine-tuning effect for BPC. Deliberately *not*
end-to-end fine-tuning — the question is whether the existing
representations differ in usefulness, not whether more gradient steps
can close a gap.

Pooling strategy is a `--pooling {last, mean}` flag. `last` (the current
default) takes the final token's hidden state, which is the correct
choice for a decoder-only/causal model — only the last token has
attended to the whole premise+hypothesis sequence under the causal mask,
so mean-pooling (the original implementation, kept as an A/B option)
dilutes that signal with a large block of premise-only states.

### Human evaluation (`human_eval.py`)

Generates blinded, randomized-order sample pairs — same fine-tuned
model, tokenizer swapped — and writes a CSV rating sheet plus a separate
answer-key JSON, so a human rater never sees which condition produced
which sample while rating. `--aggregate` reads a filled-in sheet back
against the key and reports per-condition fluency/coherence
means and pass-as-human-written rate.

### `error_diagnostics.py`

A thin wrapper (`run_with_diagnostics`) used by Stage 6 and human-eval's
`main()`. On any exception it force-flushes stdout/stderr, prints the
full traceback explicitly, and — if the error looks CUDA/memory-related
— dumps `torch.cuda.memory_summary()` plus a plain-English hint. Exists
because Kaggle's "Save & Run All" commit mode doesn't reliably surface a
subprocess's stderr into the saved notebook output the same way
interactive execution does; without this, a crash showed up as nothing
but "exited with code 1" and no visible cause.

### `run_pipeline.py`

Orchestrates Stages 1 through 5 (`--stage {1,2,2d,3,4,5,all}`). Stage 6
and human-eval are standalone modules, run directly (`python -m
pipeline.stage6_downstream`, `python -m pipeline.human_eval`) rather than
through this orchestrator, since they're evaluation add-ons rather than
pipeline-critical stages.

---

## 4. Results — Hindi (current)

Test corpus: 2,244,197 characters / 5,703,559 bytes. All numbers below are
from the most recent successful Kaggle run.

### 4.1 Corpus baseline entropy

Character entropy 5.121 bits/char, byte entropy 3.690 bits/byte — the
theoretical floor any compressor is working against.

### 4.2 Classical compressors (reference point, not the interesting result)

| Method | Ratio | BPC |
|---|---|---|
| gzip | 4.84× | 4.201 |
| zstd | 6.70× | 3.032 |
| lzma | 6.83× | 2.977 |
| bzip2 | 7.11× | 2.861 |

### 4.3 Tokenizer comparison — vowel-split % (the intrinsic result)

| Tokenizer | Tokens/sent | Tok/char | Vowel-split % |
|---|---|---|---|
| **DevAware (this project)** | 65.54 | 0.2456 | **0.03%** |
| SP-BPE (plain baseline) | 67.80 | 0.2540 | 2.37% |
| SP-Unigram | 67.96 | — | 2.85% |
| gpt2 | 358.66 | — | 0.0%¹ |
| Llama-2-7b | 285.52 | — | 100.0%¹ |
| gemma-2b | 106.25 | — | 21.97% |
| IndicBERTv2 | 65.64 | — | 3.77% |
| sarvam-1 | 78.33 | — | 5.88% |
| OpenHathi-7B | 100.06 | — | 9.14% |
| Airavata (own tokenizer) | 100.06 | — | 9.14% |
| aya-101 | 107.13 | — | 12.32% |

¹ gpt2 is byte-level BPE so a 0% vowel-split rate is a trivial artifact
of the tokenization scheme, not a meaningful comparison point. Llama-2's
100% is the opposite extreme — its tokenizer wasn't built for Devanagari
at all and splits essentially every vowel mark off its consonant.

This is now backed by **8 comparison tokenizers instead of the original
1** (IndicBERTv2 only), and DevAware wins on vowel-split-% against every
one of them by a wide margin, while also using fewer tokens/sentence than
the plain SentencePiece baseline. This is the strongest, cleanest result
in the project.

### 4.4 LLM compression — the four-condition table

50,000-character held-out sample, 95% bootstrap CI on bits-per-token
where computed:

| Condition | BPC | Compression ratio | Bits/token | 95% CI |
|---|---|---|---|---|
| Model default tokenizer, no fine-tune | 2.157 | 9.47× | 5.527 | — |
| DevAware tokenizer, base model, **not** fine-tuned | 2.217 | 9.22× | 6.181 | — |
| **DevAware tokenizer, fine-tuned** | **1.921** | **10.64×** | 5.355 | [5.211, 5.504] |
| Default tokenizer, fine-tuned | 1.947 | 10.49× | 4.990 | [4.853, 5.127] |

Reading this correctly requires the full 2×2, not just the headline
number:

- Fine-tuning **alone** helps (2.157 → 1.947 with the default
  tokenizer).
- The DevAware tokenizer **without** fine-tuning is actually slightly
  *worse* than the default tokenizer's baseline (2.217 vs 2.157) — the
  new vocabulary's smart-initialized embeddings aren't useful until the
  model has been trained on them. This is expected and, honestly, a
  useful negative result: it shows the vocabulary-extension benefit is
  not "free," it has to be earned by the fine-tune.
- Combined (DevAware + fine-tuned) gives the best BPC of all four
  conditions: 1.921, beating fine-tuned-with-default-tokenizer's 1.947.
- **The 95% CIs on bits-per-token don't overlap** (5.211–5.504 vs
  4.853–5.127) — interesting, but read this carefully: it's the CI
  *within one test-set draw at one fine-tuning seed*, not variance
  *across* different fine-tuning runs. It rules out "this specific
  measurement is dominated by which sentences happened to be in the test
  set," but it does not yet rule out "a different random initialization
  of the fine-tune would have landed somewhere else." That second check
  (multi-seed) hasn't been run yet — the infrastructure for it
  (`config.SEED`, `--seed` on Stage 2d) exists in the code but a second
  seed hasn't actually been executed.

All four conditions beat every classical compressor by a wide margin
(bzip2's 2.861 BPC vs. the LLM conditions' 1.9–2.2), which isn't
surprising — it's the standard result that a trained LM's next-token
predictions are a much better lossless-compression prior than a generic
byte-pattern compressor for natural-language text — but it does confirm
the harness itself is measuring what it should.

### 4.5 Downstream task (XNLI probe) — inconclusive, not a validated result

Frozen-representation logistic regression probe on 3-way Hindi NLI
(XNLI), 4,000 training examples, 1,000 eval examples, mean-pooled hidden
states (the run this data comes from predates the `--pooling last` fix):

| Condition | Accuracy | Macro-F1 |
|---|---|---|
| DevAware tokenizer | 35.4% | 0.353 |
| Default tokenizer | 35.7% | 0.357 |

Random-guess baseline for 3-way classification is 33.3%. **Both
conditions are barely above chance.** The honest read is not "the two
tokenizers perform equally on this task" — it's "this particular probe
setup isn't extracting a usable signal from either condition," which is
a different and more fixable problem. Leading hypothesis: mean-pooling
is the wrong choice for a decoder-only/causal model, since only the
final token's hidden state has attended to the complete
premise+hypothesis sequence under the causal attention mask; earlier
positions' states are partial views. Code now supports `--pooling last`
to test this directly; it hasn't been re-run yet. If accuracy is still
near chance with last-token pooling, the next things to try, in order,
are more probe training data (4,000 examples for a 4,096-dim probe is
thin) and then a different downstream task better matched to what a
frozen causal-LM representation is good at.

**Do not report this delta (-0.3 percentage points) as evidence of
anything** — it's noise around a chance-level floor, not a measured
effect.

### 4.6 Human evaluation — pilot rating only, not real data

20 blinded, 400-character generated samples (10 per condition, same
fine-tuned weights, tokenizer swapped) were rated — **by Claude, as a
single non-independent, non-native, uncontrolled pilot rater**, purely to
get a fast read on whether it's worth collecting real human ratings. This
is explicitly not citable evidence:

| Condition | Fluency (1–5) | Coherence (1–5) | Pass-as-human-written |
|---|---|---|---|
| DevAware | 2.8 | **2.9** | 30% |
| Default | 2.7 | **2.2** | 30% |

Both conditions produce grammatically real Hindi with the same
generation-quality ceiling: roughly 70% of samples contain at least one
garbled or fabricated token (fused nonsense compounds, invented proper
nouns) somewhere in a ~400-character sample — a base-model/fine-tune
limitation, not something either tokenizer choice fixes. Where they
diverge is coherence: DevAware's errors tend to be a single bad word
dropped into an otherwise on-topic paragraph; Default had more
*structural* breaks — a fabricated list of Indian authors, a garbled
country list, and one sample that abandoned the prompt topic entirely
partway through. That's the same direction as the BPC result, which is
at least internally consistent, but it comes from one non-blind-in-the-
statistical-sense rater and should not be treated as validated until
real human raters (2–3 independent, with an agreement statistic like
Cohen's κ) rate the same sheet.

### 4.7 Overall assessment

| Question | Status |
|---|---|
| Is DevAware intrinsically more efficient at the script level? | **Yes, strongly** — 8 baselines, near-zero vowel-splitting, fewer tokens/sentence. |
| Does it improve real LLM compression? | **Yes, at one seed** — clear 2×2 ablation, non-overlapping within-test-set CIs. Needs a second/third fine-tuning seed before calling it stable. |
| Does it help a downstream task? | **Not yet demonstrated either way** — probe setup needs the pooling fix (and possibly more data) before this question has been actually asked. |
| Does it improve generation quality a person would notice? | **Weak, single-rater signal favoring DevAware on coherence** — needs real human raters to count as evidence. |

---

## 5. Bugs found and fixed along the way (worth knowing about if extending this code)

- **Vowel-splitting via literal-space joining** (`pua_remap.py`) — the
  DevAware tokenizer was originally *worse* than the plain baseline
  because SentencePiece treated the spaces between aksharas as hard
  token boundaries. Fixed by mapping each akshara to a single PUA
  codepoint instead.
- **Stage 3/4 duplicate 7B model loads** — Stage 4 was reloading the
  full model and recomputing BPC from scratch per language, duplicating
  Stage 3's measurement and roughly doubling runtime. Fixed to reuse
  Stage 3's saved results.
- **"Fine-tuned + default tokenizer" silently measuring the wrong
  model** — when the devaware compressor is built by merging Stage 2d's
  LoRA adapter onto the *shared* base model in place (to save GPU memory
  on a single-GPU Kaggle setup), that merge mutates the shared model's
  weights. Any condition meant to represent the "not yet fine-tuned"
  state has to be captured *before* that merge happens, or it silently
  reports the fine-tuned model under the wrong label. `stage3_compress.py`
  now takes precomputed results for exactly this reason, with the
  ordering enforced in `run_pipeline.run_stage_3`.
- **CUDA OOM on Kaggle T4 (16GB) during Stage 2d/3 model loading** — a
  sequence of small fixes (`mean_resizing=False`, `low_cpu_mem_usage=True`,
  `autocast_adapter_dtype=False`) each closed a several-hundred-MB to
  several-GB transient memory spike during vocabulary resizing / PEFT
  adapter loading that was pushing a ~14GB model over a 14.56GB usable
  budget.
- **`float != bfloat16` crash in Stage 6/human-eval** —
  `load_finetuned_devaware_model()` moved the model to the GPU without
  specifying a dtype, so `resize_token_embeddings`/PEFT's
  `modules_to_save` handling could leave `embed_tokens`/`lm_head` in
  float32 while the rest of the (bf16-loaded) model stayed bfloat16.
  Stage 3's own forward pass happened to never hit the mismatched pair
  in a way that surfaced it; Stage 6's full forward pass
  (`output_hidden_states=True`) and human-eval's `.generate()` both did,
  crashing at the very first attention layer. Fixed by explicitly
  casting the whole model to one consistent dtype before returning it.
- **Silent subprocess failures in the Kaggle notebook** — `subprocess.call()`
  from within a notebook cell doesn't reliably surface a child process's
  stderr into the saved cell output under Kaggle's "Save & Run All"
  commit mode, so a real crash showed up as nothing but "exited with
  code 1." Fixed by switching to `subprocess.run(capture_output=True)`
  and explicitly printing `stdout`/`stderr` in the cell, plus the
  `error_diagnostics.py` wrapper for full tracebacks and CUDA-memory
  context.

---

## 6. Reproducing this

```bash
# Stages 1-4, Hindi, using the DevAware fine-tuned tokenizer, with CI:
python run_pipeline.py --stage 1 --lang hindi
python run_pipeline.py --stage 2 --lang hindi
python run_pipeline.py --stage 2d --lang hindi
python run_pipeline.py --stage 3 --lang hindi --use-devaware-tokenizer --bootstrap-ci
python run_pipeline.py --stage 4 --lang hindi

# Downstream task probe:
python -m pipeline.stage6_downstream --lang hindi --pooling last

# Human-eval sheet:
python -m pipeline.human_eval --lang hindi --n-samples 20
# ... rate results/hindi/human_eval_sheet.csv by hand, then:
python -m pipeline.human_eval --lang hindi --aggregate
```

On Kaggle specifically, the accompanying notebook (`devnagri-restarte*.ipynb`)
handles session-budget guarding, persistent-checkpoint restore across
sessions, and dependency setup — see `KAGGLE_RUN_GUIDE.md` in the repo.
