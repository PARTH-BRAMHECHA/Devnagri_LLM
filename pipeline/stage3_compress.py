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
"""

import argparse
import gzip
import bz2
import io
import json
import lzma
import math
import os
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
    - precision_bits auto-scales with vocab_size, so the "everyone gets at
      least 1 count" floor never eats the whole budget (or exceeds it) for
      large-vocabulary models
    """
    vocab_size = len(probs)

    # Guard against NaN/Inf probabilities (can slip through from upstream
    # numerical issues); fall back to a uniform distribution rather than
    # silently producing a corrupt/negative cumulative table.
    if not np.all(np.isfinite(probs)):
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        s = probs.sum()
        probs = (probs / s) if s > 0 else np.full(vocab_size, 1.0 / vocab_size)

    # Every symbol needs at least count 1, so precision_bits must give
    # enough headroom above vocab_size or quantization silently corrupts
    # (previously: total=65536 with a 32000+ vocab left almost no bits to
    # actually encode the probability distribution, and would go negative
    # for any vocab_size > 65536).
    min_precision_bits = max(precision_bits, math.ceil(math.log2(max(vocab_size, 2))) + 4)
    total = 1 << min_precision_bits

    # Quantize probabilities to integer counts
    counts = np.maximum(np.floor(probs * total).astype(np.int64), 1)

    # Adjust total to match exactly
    diff = total - counts.sum()
    if diff != 0:
        # Add/subtract from the highest-probability symbol, but never let
        # a count drop to zero or below.
        max_idx = np.argmax(counts)
        counts[max_idx] += diff
        if counts[max_idx] < 1:
            # Extremely unlikely (would need a pathological distribution),
            # but stay safe: borrow the shortfall from other symbols.
            shortfall = 1 - counts[max_idx]
            counts[max_idx] = 1
            order = np.argsort(-counts)
            for idx in order:
                if idx == max_idx:
                    continue
                take = min(shortfall, counts[idx] - 1)
                counts[idx] -= take
                shortfall -= take
                if shortfall <= 0:
                    break

    # Build cumulative array (length = vocab_size + 1)
    cum = np.zeros(len(counts) + 1, dtype=np.int64)
    cum[1:] = np.cumsum(counts)

    return cum


# ─── LLM Compression Core ───────────────────────────────────────────────────

class LLMCompressor:
    """
    LLM-driven lossless compression using arithmetic coding.

    The key insight: an LLM's predicted probability distribution for the
    next token IS a compression model. Better predictions → fewer bits.
    """

    def __init__(self, model_name: str, device: str = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Loading LLM: {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

        self.vocab_size = self.model.config.vocab_size
        print(f"  Model loaded. Vocab size: {self.vocab_size}")

    @torch.no_grad()
    def get_next_token_probs(self, token_ids: list) -> np.ndarray:
        """Get next-token probability distribution from the model."""
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        outputs = self.model(input_ids)
        logits = outputs.logits[0, -1, :]

        # Always softmax in float32, regardless of the model's own dtype.
        # When the model is loaded in float16 (as we do on GPU), extreme
        # logit values from a 7B model can overflow inside softmax's
        # exp(), producing NaN/Inf probabilities. Those NaNs then corrupt
        # the arithmetic coder downstream (observed in production as
        # "RuntimeWarning: overflow encountered in cast" followed by
        # "math domain error" on every single compression run).
        probs = torch.softmax(logits.float(), dim=0).cpu().numpy()

        if not np.all(np.isfinite(probs)):
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            total = probs.sum()
            if total <= 0:
                probs = np.full_like(probs, 1.0 / probs.shape[0])
            else:
                probs = probs / total

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

        encoder = ArithmeticEncoder()
        total_log_prob = 0.0

        # First token: encode with uniform distribution
        uniform_cum = np.arange(self.vocab_size + 1, dtype=np.int64)
        encoder.encode_symbol(uniform_cum, token_ids[0])

        # Subsequent tokens: use LLM predictions
        for i in tqdm(range(1, n_tokens), desc="Compressing", unit=" tokens",
                      disable=(n_tokens < 100)):
            # Context: last `context_length` tokens
            ctx_start = max(0, i - context_length)
            context = token_ids[ctx_start:i]

            probs = self.get_next_token_probs(context)
            total_log_prob += math.log2(max(probs[token_ids[i]], 1e-30))

            cum_probs = probs_to_cumulative(probs)
            encoder.encode_symbol(cum_probs, token_ids[i])

        compressed = encoder.finish()

        # Prepend header: original token count (4 bytes)
        header = struct.pack(">I", n_tokens)
        result = header + compressed

        return result, n_tokens, total_log_prob

    def decompress(self, compressed_data: bytes) -> str:
        """
        Decompress data back to text.
        Returns the decompressed text string.
        """
        # Read header
        n_tokens = struct.unpack(">I", compressed_data[:4])[0]
        encoded_bytes = compressed_data[4:]

        decoder = ArithmeticDecoder(encoded_bytes)

        # Decode first token with uniform distribution
        uniform_cum = np.arange(self.vocab_size + 1, dtype=np.int64)
        token_ids = [decoder.decode_symbol(uniform_cum)]

        # Decode subsequent tokens using LLM predictions
        for i in tqdm(range(1, n_tokens), desc="Decompressing", unit=" tokens",
                      disable=(n_tokens < 100)):
            ctx_start = max(0, i - LLM_CONTEXT_LENGTH)
            context = token_ids[ctx_start:i]

            probs = self.get_next_token_probs(context)
            cum_probs = probs_to_cumulative(probs)

            symbol = decoder.decode_symbol(cum_probs)
            token_ids.append(symbol)

        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def compute_bpc(self, text: str) -> dict:
        """
        Compute bits-per-character (BPC) for a text without full compression.
        Faster than full compress/decompress — just accumulates cross-entropy.
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        n_tokens = len(token_ids)
        text_bytes = len(text.encode("utf-8"))
        text_chars = len(text)

        total_bits = 0.0

        # First token: uniform → log2(vocab_size) bits
        total_bits += math.log2(self.vocab_size)

        for i in range(1, n_tokens):
            ctx_start = max(0, i - LLM_CONTEXT_LENGTH)
            context = token_ids[ctx_start:i]

            probs = self.get_next_token_probs(context)
            prob = max(probs[token_ids[i]], 1e-30)
            total_bits += -math.log2(prob)

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


# ─── Per-Language Compression ────────────────────────────────────────────────

def run_compression(lang: str, verify: bool = False, classical_only: bool = False):
    """Run compression pipeline for a language."""
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
    args = parser.parse_args()

    langs = LANGUAGES if args.lang == "all" else [args.lang]

    for lang in langs:
        print(f"\n{'='*60}")
        print(f"  COMPRESSION: {lang.upper()}")
        print(f"{'='*60}")
        run_compression(lang, verify=args.verify, classical_only=args.classical_only)

    print("\n✓ Stage 3 (Compression) complete.")


if __name__ == "__main__":
    main()
