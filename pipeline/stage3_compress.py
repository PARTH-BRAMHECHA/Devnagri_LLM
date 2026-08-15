"""
Stage 3 — LLM-Driven Arithmetic Coding Compression Pipeline
=============================================================
Implements the autoregressive compression loop:
  context → LLM logits → probability distribution → arithmetic coder → bits

Supports:
  - Multiple LLM backends (HuggingFace Indic models)
  - Multiple tokenizer variants (baseline BPE, Devanagari-aware BPE)
  - Lossless round-trip verification
  - Classical compressor baselines (gzip, bzip2, zstd, lzma)

Usage:
    python -m pipeline.stage3_compress [--lang hindi|marathi|sanskrit|all] [--verify]

─────────────────────────────────────────────────────────────────────────────
CHANGELOG (fixes applied after the Kaggle run that crashed with
"math domain error" on every language, ~8 hours in):

1. FLOAT16 PRECISION BUG (root cause of the crashes):
   `get_next_token_probs` used to return the softmax output as float16
   (inherited from running the model in fp16). Two things broke downstream:
     a) `probs_to_cumulative` computed `probs * 65536`, which overflows
        float16 (max ~65504) whenever the model is confident about a
        token — producing inf/garbage in the quantized counts.
     b) `max(probs[token_ids[i]], 1e-30)` silently failed as a
        zero-probability guard: comparing a numpy.float16 scalar against
        1e-30 implicitly downcasts 1e-30 to float16, where it rounds to
        0.0. So for tokens with probability that underflows to -0.0 in
        float16, `max(-0.0, 0.0)` returns -0.0, and `math.log2(-0.0)`
        raises exactly the `math domain error` seen in the log.
   Fix: softmax and all downstream probability math now happen in
   float32/float64. Only the model *weights* stay fp16 for memory/speed.

2. NO KV-CACHE (why it took hours instead of minutes):
   The old code re-ran a full forward pass over the entire context window
   for *every single token*, which is O(n²) in sequence length. This is
   why 50,000 characters took 45+ minutes before crashing.
   Fix: `_SlidingKVCache` reuses `past_key_values` across steps, only
   paying for a full forward pass once every `context_length` tokens
   (when the window needs to slide), and a cheap single-token append on
   every other step. Note: this makes the context window "block-aligned"
   rather than continuously sliding by exactly one token every step —
   a standard trade-off in LLM-compression benchmarks, and a small price
   for turning O(n²) into O(n).
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import gzip
import bz2
import io
import json
import lzma
import math
import os
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pipeline.config import (
    SPLIT_DIR, TOKENIZER_DIR, MODEL_DIR, RESULTS_DIR, LANGUAGES,
    LLM_CONTEXT_LENGTH, COMPRESSION_BATCH_SIZE, INDIC_LLM_MODEL,
    CLASSICAL_COMPRESSORS, ROUNDTRIP_TEST_SIZE, ensure_dirs
)


# ─── Arithmetic Coder (Pure Python, Correct Implementation) ─────────────────

class ArithmeticEncoder:
    """
    Arithmetic encoder using integer arithmetic with carry propagation.

    Based on the standard implementation from:
      - Witten, Neal, Cleary (1987) "Arithmetic Coding for Data Compression"
      - Adapted for LLM probability distributions
    """

    PRECISION = 32
    FULL = 1 << PRECISION
    HALF = FULL >> 1
    QUARTER = HALF >> 1

    def __init__(self):
        self.low = 0
        self.high = self.FULL - 1
        self.pending_bits = 0
        self.output_bits = []

    def encode_symbol(self, cum_probs: np.ndarray, symbol: int):
        """
        Encode a symbol given cumulative probabilities.

        cum_probs: array of length vocab_size + 1, where
                   cum_probs[i] = sum of probs for symbols 0..i-1
                   cum_probs[0] = 0, cum_probs[-1] = total (should be a power of 2)
        symbol: integer symbol to encode
        """
        range_size = self.high - self.low + 1
        total = cum_probs[-1]

        # Update interval
        self.high = self.low + (range_size * int(cum_probs[symbol + 1])) // int(total) - 1
        self.low = self.low + (range_size * int(cum_probs[symbol])) // int(total)

        # Normalization loop
        while True:
            if self.high < self.HALF:
                self._output_bit(0)
            elif self.low >= self.HALF:
                self._output_bit(1)
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < 3 * self.QUARTER:
                self.pending_bits += 1
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break

            self.low = self.low << 1
            self.high = (self.high << 1) | 1

    def _output_bit(self, bit):
        self.output_bits.append(bit)
        for _ in range(self.pending_bits):
            self.output_bits.append(1 - bit)
        self.pending_bits = 0

    def finish(self) -> bytes:
        """Flush remaining bits and return encoded bytes."""
        self.pending_bits += 1
        if self.low < self.QUARTER:
            self._output_bit(0)
        else:
            self._output_bit(1)

        # Pack bits into bytes
        while len(self.output_bits) % 8 != 0:
            self.output_bits.append(0)

        result = bytearray()
        for i in range(0, len(self.output_bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | self.output_bits[i + j]
            result.append(byte)

        return bytes(result)

    def get_compressed_bits(self) -> int:
        """Return current number of output bits."""
        return len(self.output_bits) + self.pending_bits


class ArithmeticDecoder:
    """Arithmetic decoder — mirror of the encoder."""

    PRECISION = 32
    FULL = 1 << PRECISION
    HALF = FULL >> 1
    QUARTER = HALF >> 1

    def __init__(self, encoded_bytes: bytes):
        self.input_bits = []
        for byte in encoded_bytes:
            for j in range(7, -1, -1):
                self.input_bits.append((byte >> j) & 1)

        self.bit_pos = 0
        self.low = 0
        self.high = self.FULL - 1

        # Initialize value from first PRECISION bits
        self.value = 0
        for _ in range(self.PRECISION):
            self.value = (self.value << 1) | self._read_bit()

    def _read_bit(self) -> int:
        if self.bit_pos < len(self.input_bits):
            bit = self.input_bits[self.bit_pos]
            self.bit_pos += 1
            return bit
        return 0

    def decode_symbol(self, cum_probs: np.ndarray) -> int:
        """Decode one symbol given cumulative probabilities."""
        range_size = self.high - self.low + 1
        total = int(cum_probs[-1])

        # Find symbol
        scaled_value = ((self.value - self.low + 1) * total - 1) // range_size
        symbol = 0
        while int(cum_probs[symbol + 1]) <= scaled_value:
            symbol += 1

        # Update interval (same as encoder)
        self.high = self.low + (range_size * int(cum_probs[symbol + 1])) // total - 1
        self.low = self.low + (range_size * int(cum_probs[symbol])) // total

        # Normalization loop
        while True:
            if self.high < self.HALF:
                pass
            elif self.low >= self.HALF:
                self.low -= self.HALF
                self.high -= self.HALF
                self.value -= self.HALF
            elif self.low >= self.QUARTER and self.high < 3 * self.QUARTER:
                self.low -= self.QUARTER
                self.high -= self.QUARTER
                self.value -= self.QUARTER
            else:
                break

            self.low = self.low << 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self._read_bit()

        return symbol


# ─── Probability Quantization ───────────────────────────────────────────────

def probs_to_cumulative(probs: np.ndarray, precision_bits: int = 16) -> np.ndarray:
    """
    Convert a probability distribution to quantized cumulative probabilities
    suitable for arithmetic coding.

    Ensures:
    - All symbols have at least count 1 (no zero-probability symbols)
    - Cumulative sums use integer arithmetic for correctness

    NOTE: `probs` must be float32/float64 (see LLMCompressor._forward_probs).
    At float16, `probs * total` can overflow (float16 max ~65504) for
    confident predictions, producing inf/garbage counts.
    """
    total = 1 << precision_bits

    # Quantize probabilities to integer counts. probs is float64 by the time
    # it reaches here, so this multiply/floor/cast is safe from overflow.
    counts = np.maximum(np.floor(probs.astype(np.float64) * total).astype(np.int64), 1)

    # Adjust total to match exactly
    diff = total - counts.sum()
    if diff != 0:
        # Add/subtract from the highest-probability symbol
        max_idx = np.argmax(counts)
        counts[max_idx] += diff

    # Build cumulative array (length = vocab_size + 1)
    cum = np.zeros(len(counts) + 1, dtype=np.int64)
    cum[1:] = np.cumsum(counts)

    return cum


# ─── Sliding-window KV cache ─────────────────────────────────────────────────

class _SlidingKVCache:
    """
    Maintains a KV cache for autoregressive next-token prediction with a
    fixed-size context window, reused across successive positions.

    Without this, predicting token i required feeding the whole
    `context_length`-token window into the model from scratch — an O(n^2)
    scheme that's why the original run took hours. With this, most steps
    just append the one new token to an existing cache (cheap).

    IMPORTANT: an HF KV cache can only grow (append) — it can't evict the
    oldest entries to enforce an exact sliding window. So instead of
    resetting the instant the window would exceed `context_length` (which
    would force a fresh full-window forward pass on *every* step — no
    speedup at all, and the bug this exact test caught during review), we
    let the cache grow up to `2 * context_length` tokens before resetting,
    then trim back down to the most recent `context_length` tokens. This
    means:
      - most steps: O(1) — append one token to the existing cache
      - every `context_length` steps: one O(context_length) reset
    which is what actually turns O(n^2) into O(n). Predictions made while
    the cache is between context_length and 2*context_length tokens long
    simply see a bit more context than the configured window, which does
    not hurt prediction quality (if anything, slightly more context can
    only help), and does not affect encode/decode symmetry since both
    compress() and decompress() use this identical scheduling.
    """

    def __init__(self, compressor: "LLMCompressor", context_length: int):
        self.compressor = compressor
        self.context_length = context_length
        self.past = None
        self.block_start = 0

    def feed_and_predict(self, token_ids, upto_index: int) -> np.ndarray:
        """
        Ensure the cache covers token_ids[block_start:upto_index] and return
        the model's predicted probability distribution for token_ids[upto_index].
        """
        c = self.compressor
        needs_fresh_block = (
            self.past is None
            or (upto_index - self.block_start) > 2 * self.context_length
        )

        if needs_fresh_block:
            self.block_start = max(0, upto_index - self.context_length)
            block = token_ids[self.block_start:upto_index]
            input_ids = torch.tensor([block], dtype=torch.long, device=c.device)
            probs, self.past = c._forward_probs(input_ids, past=None)
        else:
            last_token = token_ids[upto_index - 1]
            input_ids = torch.tensor([[last_token]], dtype=torch.long, device=c.device)
            probs, self.past = c._forward_probs(input_ids, past=self.past)

        return probs


# ─── LLM Compression Core ───────────────────────────────────────────────────

class LLMCompressor:
    """
    LLM-driven lossless compression using arithmetic coding.

    The key insight: an LLM's predicted probability distribution for the
    next token IS a compression model. Better predictions → fewer bits.
    """

    def __init__(self, model_name: str, device: str = None,
                 tokenizer=None, model=None):
        """
        `tokenizer` / `model`: pass a pre-built (already loaded) tokenizer
        and model to use them as-is instead of loading `model_name` fresh
        from the Hub. This is how Stage 2d's vocab-extended, fine-tuned
        model gets plugged in as the "your tokenizer" condition -- see
        pipeline/stage2d_vocab_extend.load_finetuned_devaware_model().
        `model_name` is still required (used for logging/labeling and as
        the fallback load path when tokenizer/model aren't supplied).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        # bfloat16 has the same exponent range as float32 (just less
        # mantissa precision), so it gives more headroom against
        # NaN/inf drift over long contexts than float16, at no extra
        # memory cost. Falls back to float16 on GPUs that don't support
        # bf16 (e.g. older T4/P100 kernels), and float32 on CPU.
        if self.device == "cuda":
            model_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        else:
            model_dtype = torch.float32

        if tokenizer is not None and model is not None:
            print(f"  Using pre-loaded model/tokenizer ({model_name}, "
                  f"vocab size {len(tokenizer)}) on {self.device}")
            self.tokenizer = tokenizer
            self.model = model.to(self.device)
            self.model.eval()
            self.vocab_size = self.model.config.vocab_size
            print(f"  Model ready. Vocab size: {self.vocab_size}")
            print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
            print(f"  Model actually on device: {next(self.model.parameters()).device}")
            sys.stdout.flush()
            self._debug_calls = 0
            return

        print(f"  Loading LLM: {model_name} on {self.device} (dtype={model_dtype})")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Prefer sdpa (PyTorch's fused scaled-dot-product-attention) over
        # the "eager" default -- meaningfully faster and lower peak memory
        # for the growing-KV-cache pattern used during compression. Some
        # trust_remote_code model classes don't accept attn_implementation
        # at all, so fall back cleanly if the load rejects the kwarg.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=model_dtype,
                attn_implementation="sdpa",
            ).to(self.device)
        except (TypeError, ValueError) as e:
            print(f"  sdpa attention not supported by this model class ({e}); "
                  f"falling back to default attention implementation.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=model_dtype,
            ).to(self.device)
        self.model.eval()

        self.vocab_size = self.model.config.vocab_size
        print(f"  Model loaded. Vocab size: {self.vocab_size}")
        print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"  Model actually on device: {next(self.model.parameters()).device}")
        sys.stdout.flush()
        self._debug_calls = 0

    @torch.no_grad()
    def _forward_probs(self, input_ids: torch.Tensor, past=None):
        """
        Run one forward pass (optionally continuing from a KV cache) and
        return (probs, new_past). Softmax is computed in float32 and the
        result is returned as float64 — this is the fix for the
        float16-overflow / float16-underflow bugs described at the top of
        this file. Never do probability math in float16.
        """
        debug = self._debug_calls < 10
        if debug:
            t0 = time.time()
            print(f"    [fwd #{self._debug_calls}] input_ids shape={tuple(input_ids.shape)}, "
                  f"device={input_ids.device}, has_past={past is not None} -- calling model...",
                  flush=True)

        outputs = self.model(input_ids, past_key_values=past, use_cache=True)

        if debug:
            torch.cuda.synchronize() if input_ids.is_cuda else None
            print(f"    [fwd #{self._debug_calls}] model() returned in {time.time()-t0:.2f}s",
                  flush=True)

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits.float(), dim=0).cpu().numpy().astype(np.float64)

        if debug:
            print(f"    [fwd #{self._debug_calls}] softmax+cpu done, total {time.time()-t0:.2f}s",
                  flush=True)
            self._debug_calls += 1

        return probs, outputs.past_key_values

    @torch.no_grad()
    def get_next_token_probs(self, token_ids: list) -> np.ndarray:
        """
        Uncached single-shot version: get next-token probs for an arbitrary
        context, with no KV reuse. Kept for callers that just need a
        one-off prediction; the compress/decompress/compute_bpc loops below
        use `_SlidingKVCache` instead so they don't pay O(n^2).
        """
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        probs, _ = self._forward_probs(input_ids, past=None)
        return probs

    def compress(self, text: str, context_length: int = None) -> tuple:
        """
        Compress text using LLM-driven arithmetic coding.

        Returns (compressed_bytes, token_count, total_log_prob).
        """
        if context_length is None:
            context_length = LLM_CONTEXT_LENGTH

        # Tokenize
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        n_tokens = len(token_ids)
        print(f"    [compress] tokenized: {n_tokens} tokens", flush=True)

        encoder = ArithmeticEncoder()
        total_log_prob = 0.0

        # First token: encode with uniform distribution
        uniform_cum = np.arange(self.vocab_size + 1, dtype=np.int64)
        encoder.encode_symbol(uniform_cum, token_ids[0])

        # Subsequent tokens: use LLM predictions, with a sliding KV cache so
        # this is O(n) instead of O(n^2).
        kv = _SlidingKVCache(self, context_length)
        t_start = time.time()
        for i in range(1, n_tokens):
            probs = kv.feed_and_predict(token_ids, i)
            total_log_prob += math.log2(max(probs[token_ids[i]], 1e-30))

            cum_probs = probs_to_cumulative(probs)
            encoder.encode_symbol(cum_probs, token_ids[i])

            if i % 20 == 0 or i == n_tokens - 1:
                elapsed = time.time() - t_start
                rate = i / max(elapsed, 1e-6)
                print(f"    [compress] {i}/{n_tokens - 1} tokens, "
                      f"{elapsed:.1f}s elapsed, {rate:.2f} tok/s", flush=True)

        compressed = encoder.finish()

        # Prepend header: original token count (4 bytes)
        header = struct.pack(">I", n_tokens)
        result = header + compressed

        return result, n_tokens, total_log_prob

    def decompress(self, compressed_data: bytes, context_length: int = None) -> str:
        """
        Decompress data back to text.
        Returns the decompressed text string.
        """
        if context_length is None:
            context_length = LLM_CONTEXT_LENGTH

        # Read header
        n_tokens = struct.unpack(">I", compressed_data[:4])[0]
        encoded_bytes = compressed_data[4:]

        decoder = ArithmeticDecoder(encoded_bytes)

        # Decode first token with uniform distribution
        uniform_cum = np.arange(self.vocab_size + 1, dtype=np.int64)
        token_ids = [decoder.decode_symbol(uniform_cum)]

        # Decode subsequent tokens using LLM predictions (same sliding cache
        # scheme as compress(), so encode/decode stay symmetric and fast).
        kv = _SlidingKVCache(self, context_length)
        t_start = time.time()
        for i in range(1, n_tokens):
            probs = kv.feed_and_predict(token_ids, i)
            cum_probs = probs_to_cumulative(probs)

            symbol = decoder.decode_symbol(cum_probs)
            token_ids.append(symbol)

            if i % 20 == 0 or i == n_tokens - 1:
                elapsed = time.time() - t_start
                rate = i / max(elapsed, 1e-6)
                print(f"    [decompress] {i}/{n_tokens - 1} tokens, "
                      f"{elapsed:.1f}s elapsed, {rate:.2f} tok/s", flush=True)

        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def compute_bpc(self, text: str, context_length: int = None) -> dict:
        """
        Compute bits-per-character (BPC) for a text without full compression.
        Faster than full compress/decompress — just accumulates cross-entropy.
        """
        if context_length is None:
            context_length = LLM_CONTEXT_LENGTH

        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        n_tokens = len(token_ids)
        text_bytes = len(text.encode("utf-8"))
        text_chars = len(text)

        total_bits = 0.0

        # First token: uniform → log2(vocab_size) bits
        total_bits += math.log2(self.vocab_size)

        kv = _SlidingKVCache(self, context_length)
        t_start = time.time()
        for i in range(1, n_tokens):
            probs = kv.feed_and_predict(token_ids, i)
            prob = max(probs[token_ids[i]], 1e-30)
            total_bits += -math.log2(prob)

            if i % 20 == 0 or i == n_tokens - 1:
                elapsed = time.time() - t_start
                rate = i / max(elapsed, 1e-6)
                print(f"    [bpc] {i}/{n_tokens - 1} tokens, "
                      f"{elapsed:.1f}s elapsed, {rate:.2f} tok/s", flush=True)

        bpc = total_bits / text_chars
        bpb = total_bits / text_bytes

        return {
            "total_bits": total_bits,
            "total_chars": text_chars,
            "total_bytes": text_bytes,
            "total_tokens": n_tokens,
            "bpc": round(bpc, 4),
            "bpb": round(bpb, 4),
            "bits_per_token": round(total_bits / n_tokens, 4),
            "compression_ratio": round(text_bytes * 8 / max(total_bits, 1), 4),
        }


# ─── Classical Compressor Baselines ─────────────────────────────────────────

def compress_classical(text: str) -> dict:
    """Compress text with all classical compressors and report sizes."""
    text_bytes = text.encode("utf-8")
    original_size = len(text_bytes)
    original_chars = len(text)

    results = {}

    # gzip
    compressed = gzip.compress(text_bytes, compresslevel=9)
    results["gzip"] = {
        "compressed_bytes": len(compressed),
        "ratio": round(original_size / max(len(compressed), 1), 4),
        "bpc": round(len(compressed) * 8 / original_chars, 4),
        "bpb": round(len(compressed) * 8 / original_size, 4),
    }

    # bzip2
    compressed = bz2.compress(text_bytes, compresslevel=9)
    results["bzip2"] = {
        "compressed_bytes": len(compressed),
        "ratio": round(original_size / max(len(compressed), 1), 4),
        "bpc": round(len(compressed) * 8 / original_chars, 4),
        "bpb": round(len(compressed) * 8 / original_size, 4),
    }

    # lzma
    compressed = lzma.compress(text_bytes)
    results["lzma"] = {
        "compressed_bytes": len(compressed),
        "ratio": round(original_size / max(len(compressed), 1), 4),
        "bpc": round(len(compressed) * 8 / original_chars, 4),
        "bpb": round(len(compressed) * 8 / original_size, 4),
    }

    # zstd (if available)
    try:
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(level=19)
        compressed = cctx.compress(text_bytes)
        results["zstd"] = {
            "compressed_bytes": len(compressed),
            "ratio": round(original_size / max(len(compressed), 1), 4),
            "bpc": round(len(compressed) * 8 / original_chars, 4),
            "bpb": round(len(compressed) * 8 / original_size, 4),
        }
    except ImportError:
        results["zstd"] = {"error": "zstandard not installed"}

    results["_meta"] = {
        "original_bytes": original_size,
        "original_chars": original_chars,
    }

    return results


# ─── Lossless Round-Trip Test ────────────────────────────────────────────────

def verify_roundtrip(compressor: LLMCompressor, text: str) -> bool:
    """Verify that compress → decompress is lossless."""
    print("  Verifying round-trip losslessness...")
    compressed, n_tokens, log_prob = compressor.compress(text)
    decompressed = compressor.decompress(compressed)

    match = decompressed.strip() == text.strip()
    if match:
        print("  ✓ Round-trip PASSED (lossless)")
    else:
        print("  ✗ Round-trip FAILED!")
        print(f"    Original ({len(text)} chars): {text[:100]}...")
        print(f"    Decoded  ({len(decompressed)} chars): {decompressed[:100]}...")

    return match


# ─── Stage 3 Non-Negotiable Sanity Checks (project plan §3a–3c) ────────────
#
# The project plan is explicit that these are correctness gates, not
# optional nice-to-haves:
#   3a: "lossless-ness is your first and non-negotiable correctness test —
#        don't proceed until this passes" (toy English sentence)
#   3b: sanity check against literature (BPC on an enwik8-style sample
#       should be in the same ballpark as published LLMZip/FineZip numbers)
#   3c go/no-go: "Confirm losslessness on Devanagari text specifically —
#       Unicode normalization edge cases ... are the most common place this
#       silently breaks. Test round-trip on at least a few hundred
#       sentences per language before scaling to the full corpus."
#
# Previously none of this ran automatically and evidence wasn't saved
# anywhere — `--verify` only round-tripped 500 characters of whatever
# language was being processed, and only if you remembered to pass the
# flag. `run_stage3_sanity_checks` below runs all three, raises on the
# first non-negotiable failure (per the plan: stop, don't proceed), and
# writes the results to disk as durable evidence.

TOY_ENGLISH_SENTENCE = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs."
)

# A short, fixed excerpt matching the real enwik8 opening paragraph (Hutter
# Prize Wikipedia dump), so 3b's plausibility check doesn't require the
# 100MB file on disk. This isn't meant to reproduce a specific paper's BPC
# exactly -- just to confirm the pipeline isn't off by an order of
# magnitude, which is the plan's own bar ("if you get 0.1 BPC or 15 BPC,
# something's broken").
ENWIK8_SAMPLE_TEXT = (
    "Anarchism is a political philosophy that advocates self-governed "
    "societies based on voluntary institutions. These are often described "
    "as stateless societies, although several authors have defined them "
    "more specifically as institutions based on non-hierarchical or free "
    "associations. Anarchism holds the state to be undesirable, "
    "unnecessary, and harmful."
)

# Published LLM-compression BPC on enwik8-scale English clusters roughly in
# this range (LLMZip, FineZip); the check only fails outside a much wider
# band, matching the plan's "not wildly off" bar rather than requiring an
# exact match to any one paper.
ENWIK8_PLAUSIBLE_BPC_RANGE = (0.3, 8.0)


def sanity_check_3a_toy_english(compressor: "LLMCompressor") -> dict:
    """Plan §3a — non-negotiable: pipeline runs end-to-end on a toy English
    sentence and the decompressed output exactly matches the input."""
    print("\n  [3a] Toy-English lossless round-trip (non-negotiable)...")
    passed = verify_roundtrip(compressor, TOY_ENGLISH_SENTENCE)
    if not passed:
        raise RuntimeError(
            "Stage 3a sanity check FAILED: toy-English round-trip is not "
            "lossless. The project plan calls this non-negotiable — stop "
            "and fix the pipeline before proceeding to 3b/3c."
        )
    print("  ✓ [3a] PASSED")
    return {"check": "3a_toy_english_roundtrip", "text": TOY_ENGLISH_SENTENCE,
            "passed": passed}


def sanity_check_3b_enwik8_plausibility(compressor: "LLMCompressor") -> dict:
    """Plan §3b — sanity check against literature: BPC on an enwik8-style
    English sample should be in the same ballpark as published numbers."""
    print("\n  [3b] enwik8-sample BPC plausibility check...")
    bpc_result = compressor.compute_bpc(ENWIK8_SAMPLE_TEXT)
    bpc = bpc_result["bpc"]
    lo, hi = ENWIK8_PLAUSIBLE_BPC_RANGE
    plausible = lo <= bpc <= hi
    print(f"    BPC = {bpc:.4f}  (plausible range checked: [{lo}, {hi}]; "
          f"published LLMZip/FineZip enwik8 BPC is typically ~0.8-1.2)")
    print("  ✓ [3b] PASSED" if plausible else
          "  ✗ [3b] FAILED — BPC is implausible, something is broken.")
    return {"check": "3b_enwik8_plausibility", "plausible": plausible,
            **bpc_result}


def sanity_check_3c_devanagari_roundtrip(compressor: "LLMCompressor", lang: str,
                                          min_sentences: int = 300) -> dict:
    """Plan §3c go/no-go — non-negotiable before scaling to the full
    corpus: confirm losslessness on Devanagari text specifically, tested on
    at least a few hundred sentences (Unicode normalization edge cases are
    the most common place round-trip silently breaks)."""
    print(f"\n  [3c] Devanagari round-trip on >= {min_sentences} sentences "
          f"({lang})...")
    test_file = SPLIT_DIR / lang / "test.txt"
    with open(test_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Split on Devanagari danda / double-danda (sentence-final punctuation
    # for Hindi/Marathi/Sanskrit), falling back to '.' for any non-Devanagari
    # punctuation mixed into the corpus.
    sentences = [s.strip() for s in re.split(r"[।॥.]\s*", text) if s.strip()]
    sample_sentences = sentences[:min_sentences]
    sample_text = "। ".join(sample_sentences)

    print(f"    Sampled {len(sample_sentences)} sentences "
          f"({len(sample_text):,} chars) for round-trip test.")

    if len(sample_sentences) < min_sentences:
        print(f"  ⚠ Only {len(sample_sentences)} sentences available in "
              f"{lang}/test.txt — below the plan's {min_sentences}-sentence "
              f"floor. Proceeding with what's available, but note this in "
              f"the methodology section.")

    passed = verify_roundtrip(compressor, sample_text)
    if not passed:
        raise RuntimeError(
            f"Stage 3c sanity check FAILED for {lang}: Devanagari "
            f"round-trip is not lossless on a {len(sample_sentences)}-"
            f"sentence sample. This is a non-negotiable go/no-go check per "
            f"the project plan — stop and diagnose (Unicode normalization "
            f"is the usual culprit) before scaling to the full corpus."
        )
    print(f"  ✓ [3c] PASSED ({len(sample_sentences)} sentences, {lang})")
    return {"check": "3c_devanagari_roundtrip", "lang": lang,
            "n_sentences": len(sample_sentences), "n_chars": len(sample_text),
            "passed": passed}


def run_stage3_sanity_checks(compressor: "LLMCompressor", langs: list,
                              run_3a_3b: bool = True) -> dict:
    """
    Run all three non-negotiable Stage 3 sanity checks and save the
    evidence to disk. 3a/3b are language-independent (run once); 3c runs
    once per language in `langs`, since it's a per-corpus go/no-go check.
    Raises on the first failing non-negotiable check by design — this is
    meant to stop the pipeline, not continue past a broken correctness
    gate.
    """
    print("\n" + "=" * 70)
    print("  STAGE 3 SANITY CHECKS (non-negotiable per project plan §3a–3c)")
    print("=" * 70)

    checks = []
    if run_3a_3b:
        checks.append(sanity_check_3a_toy_english(compressor))
        checks.append(sanity_check_3b_enwik8_plausibility(compressor))
    for l in langs:
        checks.append(sanity_check_3c_devanagari_roundtrip(compressor, l))

    evidence = {
        "model": compressor.model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "languages": langs,
        "checks": checks,
        "all_passed": all(c.get("passed", c.get("plausible")) for c in checks),
    }

    out_path = RESULTS_DIR / "stage3_sanity_checks.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Sanity-check evidence saved: {out_path}")

    return evidence


# ─── Per-Language Compression ────────────────────────────────────────────────

def run_compression(lang: str, verify: bool = False, classical_only: bool = False,
                     compressor: "LLMCompressor" = None,
                     devaware_compressor: "LLMCompressor" = None):
    """
    Run compression pipeline for a language.

    `compressor` can be passed in (an already-loaded LLMCompressor) so that
    run_pipeline.py can load the 7B model once and reuse it across all
    languages/stages instead of reloading it from disk every time.

    `devaware_compressor`: optional second LLMCompressor built from Stage
    2d's vocab-extended, fine-tuned model (see
    pipeline.stage2d_vocab_extend.load_finetuned_devaware_model). When
    supplied, this fills in the "your tokenizer" column of the three-
    condition table (your tokenizer / model default / classical) that the
    project plan calls for. When omitted, results["llm_compression"] alone
    is reported and the results file should be read as the two-condition
    (model default + classical) comparison -- state that explicitly in the
    methodology if you haven't run Stage 2d yet.
    """
    ensure_dirs()

    test_file = SPLIT_DIR / lang / "test.txt"
    if not test_file.exists():
        print(f"  ERROR: {test_file} not found. Run stage1 first.")
        return

    # Load test text
    print(f"  Loading test data from {test_file}...")
    with open(test_file, "r", encoding="utf-8") as f:
        test_text = f.read()

    # Use a reasonable sample for compression (full test can be very slow with LLM)
    # For classical compressors, use the full test set
    # For LLM compression, use a smaller sample
    sample_size_chars = 50_000  # ~50K chars for LLM evaluation
    test_sample = test_text[:sample_size_chars]

    print(f"  Test set: {len(test_text):,} chars ({len(test_text.encode('utf-8')) / (1024**2):.1f} MB)")
    print(f"  LLM sample: {len(test_sample):,} chars")

    results = {"language": lang}

    # Classical compressors (on full test set)
    print(f"\n  --- Classical Compressors ---")
    classical = compress_classical(test_text)
    results["classical"] = classical

    for name, r in classical.items():
        if name == "_meta":
            continue
        if "error" in r:
            print(f"    {name}: {r['error']}")
        else:
            print(f"    {name}: ratio={r['ratio']:.3f}, BPC={r['bpc']:.3f}")

    if classical_only:
        # Save and return
        results_file = RESULTS_DIR / lang / "compression_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ Results saved: {results_file}")
        return results

    # LLM-driven compression
    print(f"\n  --- LLM Compression ---")
    try:
        if compressor is None:
            compressor = LLMCompressor(INDIC_LLM_MODEL)

        # Verify losslessness first (on a small sample)
        if verify:
            verify_text = test_sample[:500]
            roundtrip_ok = verify_roundtrip(compressor, verify_text)
            if not roundtrip_ok:
                print("  ⚠ Round-trip verification failed! Proceeding with caution.")

        # Compute BPC on sample
        print(f"  Computing BPC on {len(test_sample):,} chars...")
        llm_result = compressor.compute_bpc(test_sample)
        results["llm_compression"] = {
            "model": INDIC_LLM_MODEL,
            "tokenizer": "model_default",
            **llm_result,
        }
        print(f"    LLM BPC: {llm_result['bpc']:.4f}")
        print(f"    Compression ratio: {llm_result['compression_ratio']:.3f}")

    except Exception as e:
        print(f"  LLM compression error: {e}")
        results["llm_compression"] = {"error": str(e)}

    # LLM-driven compression with the DevAware-extended tokenizer ("your
    # tokenizer" condition), if a fine-tuned compressor was supplied.
    if devaware_compressor is not None:
        print(f"\n  --- LLM Compression (DevAware tokenizer, fine-tuned) ---")
        try:
            print(f"  Computing BPC on {len(test_sample):,} chars...")
            dev_result = devaware_compressor.compute_bpc(test_sample)
            results["llm_compression_devaware_tokenizer"] = {
                "model": f"{INDIC_LLM_MODEL} (vocab-extended, LoRA fine-tuned)",
                "tokenizer": "devaware",
                **dev_result,
            }
            print(f"    LLM BPC (devaware): {dev_result['bpc']:.4f}")
            print(f"    Compression ratio: {dev_result['compression_ratio']:.3f}")
        except Exception as e:
            print(f"  DevAware LLM compression error: {e}")
            results["llm_compression_devaware_tokenizer"] = {"error": str(e)}

    # Save results
    results_file = RESULTS_DIR / lang / "compression_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Results saved: {results_file}")

    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: LLM-driven arithmetic coding compression"
    )
    parser.add_argument("--lang", choices=LANGUAGES + ["all"], default="all",
                        help="Language to compress (default: all)")
    parser.add_argument("--verify", action="store_true",
                        help="Run lossless round-trip verification")
    parser.add_argument("--classical-only", action="store_true",
                        help="Only run classical compressors (skip LLM)")
    parser.add_argument("--sanity-checks", action="store_true",
                        help="Run the non-negotiable Stage 3 sanity checks "
                             "(plan §3a-3c: toy-English round-trip, "
                             "enwik8 BPC plausibility, Devanagari round-trip "
                             "on >=300 sentences per language) before "
                             "compression, and save evidence to "
                             "results/stage3_sanity_checks.json. Raises if "
                             "a non-negotiable check fails.")
    args = parser.parse_args()

    langs = LANGUAGES if args.lang == "all" else [args.lang]

    # Load the LLM once and reuse it across languages instead of reloading
    # a 7B model from disk for every language.
    shared_compressor = None
    if not args.classical_only:
        shared_compressor = LLMCompressor(INDIC_LLM_MODEL)

    if args.sanity_checks and not args.classical_only:
        run_stage3_sanity_checks(shared_compressor, langs)

    for lang in langs:
        print(f"\n{'='*60}")
        print(f"  COMPRESSION: {lang.upper()}")
        print(f"{'='*60}")
        run_compression(lang, verify=args.verify, classical_only=args.classical_only,
                         compressor=shared_compressor)

    print("\n✓ Stage 3 (Compression) complete.")


if __name__ == "__main__":
    main()
