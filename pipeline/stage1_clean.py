"""
Stage 1b — Cleaning
====================
NFC normalization, script validation, boilerplate stripping, deduplication.

Goal: <1% non-Devanagari characters remaining; dedup rate logged.

Usage:
    python -m pipeline.stage1_clean [--lang hindi|marathi|sanskrit|all]
"""

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from pipeline.config import (
    CLEAN_DIR, LANGUAGES, DEVANAGARI_RANGES,
    MIN_DEVANAGARI_RATIO, MIN_LINE_LENGTH, ensure_dirs
)

# Any raw line longer than this is almost certainly a merged/mangled
# extraction artifact (broken table, infobox, swallowed newlines, etc.)
# rather than real running text. Truncating it up front avoids feeding
# multi-KB strings into backtracking-prone regexes and the per-char
# devanagari_ratio scan, which is what causes multi-second stalls on
# individual lines.
MAX_LINE_LEN = 20000

# How often (in lines) to print a timing checkpoint during the cleaning
# pass, so a future stall shows exactly which line index it happened at
# instead of just freezing the tqdm counter.
LOG_EVERY = 50000


def is_devanagari(ch: str) -> bool:
    """Check if a character falls in any Devanagari Unicode range."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in DEVANAGARI_RANGES)


def devanagari_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Devanagari."""
    alpha_chars = [ch for ch in text if ch.isalpha()]
    if not alpha_chars:
        return 0.0
    dev_count = sum(1 for ch in alpha_chars if is_devanagari(ch))
    return dev_count / len(alpha_chars)


def normalize_unicode(text: str) -> str:
    """NFC normalization — canonical composition for Devanagari."""
    return unicodedata.normalize("NFC", text)


def strip_boilerplate(text: str) -> str:
    """Remove common boilerplate patterns from web/wiki text."""
    # URLs
    text = re.sub(r"https?://\S+", "", text)

    # Email addresses — token-based instead of a greedy \S+@\S+\.\S+
    # regex. The old pattern has three unanchored \S+ runs, which is
    # cheap on short normal lines but can blow up to O(n^2)/O(n^3)
    # backtracking on one long line with no '@' in it at all. Splitting
    # on whitespace and checking tokens directly is O(n) regardless of
    # content.
    if "@" in text:
        text = " ".join(
            "" if ("@" in tok and "." in tok) else tok
            for tok in text.split(" ")
        )

    # HTML entities
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    # Consecutive punctuation (likely artifacts)
    text = re.sub(r"[।,.!?;:]{3,}", "।", text)
    # Excess whitespace
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def clean_line(line: str) -> str:
    """Clean a single line: normalize, strip, validate."""
    # Bail out on pathological lines BEFORE any regex or unicode work
    # touches them. This is the main fix for the stall: a single
    # tens-of-KB "line" (mangled table/infobox, swallowed newlines) can
    # otherwise take tens of seconds to process on its own.
    if len(line) > MAX_LINE_LEN:
        line = line[:MAX_LINE_LEN]

    # NFC normalize
    line = normalize_unicode(line)
    # Strip boilerplate
    line = strip_boilerplate(line)
    # Remove leading/trailing whitespace
    line = line.strip()

    return line


def deduplicate_lines(lines: list) -> tuple:
    """
    Remove exact duplicates using line-level hashing.
    Returns (deduplicated_lines, stats).

    NOTE: blank lines ("") are paragraph/document boundary markers
    inserted by clean_corpus(), not content — they must NEVER be
    deduplicated. Every blank line hashes identically, so treating
    them like normal content collapses tens of thousands of distinct
    document boundaries down to a single leftover blank line, which
    then makes near_dedup_paragraphs() (and the downstream train/test
    splitter) see the entire corpus as 1-2 giant "documents" instead
    of hundreds of thousands of separate ones.
    """
    seen_hashes = set()
    unique_lines = []
    dup_count = 0

    for line in lines:
        if not line:
            # Boundary marker — always pass through, never counted
            # as a duplicate.
            unique_lines.append(line)
            continue

        h = hashlib.md5(line.encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_lines.append(line)
        else:
            dup_count += 1

    return unique_lines, dup_count


def near_dedup_paragraphs(lines: list, ngram_size: int = 5, threshold: float = 0.8,
                           max_para_chars: int = 4000) -> tuple:
    """
    Fast near-duplicate removal using lightweight MinHash signatures.
    Groups consecutive non-empty lines as paragraphs, then removes
    paragraphs whose MinHash signature matches a seen one above threshold.
    """
    import struct

    # Group lines into paragraphs (separated by empty lines).
    #
    # IMPORTANT: paragraph boundaries here rely entirely on blank-line
    # markers surviving from clean_corpus(). If clean_corpus() drops a
    # rejected line via `continue` (no "" appended), two originally
    # separate documents get silently fused into one paragraph. On a
    # corpus where most lines get rejected (script/length filters), this
    # can collapse the ENTIRE dataset into 1-2 gigantic "paragraphs" —
    # which is exactly what produces "Paragraphs to check: 2" and then a
    # multi-hour hang below, since minhash_signature is O(len(paragraph))
    # and the progress log (every 50k paragraphs) never fires when there
    # are only 2 paragraphs total.
    #
    # Additionally cap the amount of text actually hashed per paragraph
    # (max_para_chars) as a hard safety net — even if boundaries are
    # fixed upstream, one legitimately huge paragraph can't blow up
    # runtime, since we hash a representative head+tail sample instead
    # of the full text.
    paragraphs = []
    current = []
    for line in lines:
        if line.strip():
            current.append(line)
        else:
            if current:
                paragraphs.append("\n".join(current))
                current = []
    if current:
        paragraphs.append("\n".join(current))

    print(f"      Paragraphs to check: {len(paragraphs):,}")

    if paragraphs:
        lengths = [len(p) for p in paragraphs]
        print(f"      Paragraph size — max: {max(lengths):,} chars, "
              f"avg: {sum(lengths)//len(lengths):,} chars")
        if len(paragraphs) < 20 and max(lengths) > 50000:
            print("      ⚠ WARNING: very few, very large paragraphs detected. "
                  "This usually means blank-line boundaries were lost upstream "
                  "(e.g. rejected lines in clean_corpus not re-inserting '' "
                  "markers). Near-dedup will still run safely (paragraphs are "
                  "capped at "
                  f"{max_para_chars:,} chars for hashing), but duplicate "
                  "detection quality will be poor at this granularity — "
                  "consider fixing boundary preservation in clean_corpus().")

    # --- Fast MinHash approach ---
    NUM_HASHES = 32
    LARGE_PRIME = (1 << 61) - 1
    # Pre-generate hash coefficients
    import random
    rng = random.Random(12345)
    hash_a = [rng.randint(1, LARGE_PRIME - 1) for _ in range(NUM_HASHES)]
    hash_b = [rng.randint(0, LARGE_PRIME - 1) for _ in range(NUM_HASHES)]

    def minhash_signature(text):
        """Compute a MinHash signature (tuple of NUM_HASHES min-hash values)."""
        # Generate character n-gram hashes on-the-fly (no set storage)
        sig = [LARGE_PRIME] * NUM_HASHES
        text_len = len(text)
        if text_len < ngram_size:
            h = hash(text)
            for k in range(NUM_HASHES):
                val = (hash_a[k] * h + hash_b[k]) % LARGE_PRIME
                sig[k] = val
            return tuple(sig)

        for i in range(text_len - ngram_size + 1):
            h = hash(text[i:i + ngram_size])
            for k in range(NUM_HASHES):
                val = (hash_a[k] * h + hash_b[k]) % LARGE_PRIME
                if val < sig[k]:
                    sig[k] = val
        return tuple(sig)

    def signature_similarity(sig1, sig2):
        """Estimate Jaccard similarity from MinHash signatures."""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / NUM_HASHES

    # Use hash-based bucketing for faster lookup
    seen_sigs = []
    unique_paragraphs = []
    near_dup_count = 0

    t_start = time.time()
    t_last_log = t_start

    for idx, para in enumerate(paragraphs):
        if len(para) < 30:
            # Short paragraphs: keep without dedup check
            unique_paragraphs.append(para)
            continue

        # Cap the text actually fed to minhash_signature. Hashing is
        # O(len(text)) with NUM_HASHES multiplications per position, so
        # an uncapped multi-million-character paragraph (see boundary
        # note above) can make a single call take minutes to hours.
        # A head+tail sample is enough to fingerprint near-duplicates
        # for this use case and bounds worst-case cost per paragraph.
        if len(para) > max_para_chars:
            half = max_para_chars // 2
            para_for_hash = para[:half] + para[-half:]
        else:
            para_for_hash = para

        sig = minhash_signature(para_for_hash)
        is_dup = False

        # Compare against recent signatures only (sliding window)
        window = seen_sigs[-200:] if len(seen_sigs) > 200 else seen_sigs
        for seen_sig in window:
            if signature_similarity(sig, seen_sig) >= threshold:
                is_dup = True
                near_dup_count += 1
                break

        if not is_dup:
            seen_sigs.append(sig)
            unique_paragraphs.append(para)

        # Time-based heartbeat (every 5s) in addition to the count-based
        # log below — this is what actually catches the "2 paragraphs,
        # each huge" case, since the count-based check below never
        # fires when there are only a handful of paragraphs total.
        now = time.time()
        if now - t_last_log >= 5:
            print(f"      Near-dedup heartbeat: paragraph {idx+1:,}/{len(paragraphs):,}, "
                  f"{now - t_start:.1f}s elapsed, {near_dup_count:,} near-dups so far")
            t_last_log = now

        if idx > 0 and idx % 50000 == 0:
            print(f"      Near-dedup progress: {idx:,}/{len(paragraphs):,} "
                  f"(removed {near_dup_count:,})")

    # Flatten back to lines
    result_lines = []
    for para in unique_paragraphs:
        result_lines.extend(para.split("\n"))
        result_lines.append("")  # Preserve paragraph breaks

    return result_lines, near_dup_count


def find_pathological_lines(input_file: Path, threshold: int = MAX_LINE_LEN, limit: int = 20):
    """
    Diagnostic helper: scan the raw input file and report any lines
    longer than `threshold` characters, without running them through
    any cleaning logic. Useful for confirming what's causing a stall
    before/after applying the length cap.
    """
    print(f"  Scanning for lines > {threshold:,} chars in {input_file.name}...")
    found = 0
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if len(line) > threshold:
                found += 1
                print(f"    line {i:,}: {len(line):,} chars — preview: {line[:100]!r}")
                if found >= limit:
                    print(f"    ...stopping after {limit} matches")
                    break
    if found == 0:
        print("    None found.")
    return found


def clean_corpus(lang: str):
    """
    Full cleaning pipeline for a language:
    1. NFC normalization
    2. Script validation (Devanagari ratio check)
    3. Boilerplate stripping
    4. Exact deduplication
    5. Near-duplicate removal
    """
    ensure_dirs()

    # Find the input file (either merged_raw.txt or wiki_extracted.txt)
    input_dir = CLEAN_DIR / lang
    candidates = ["merged_raw.txt", "wiki_extracted.txt",
                  "cc100_extracted.txt", "l3cube_extracted.txt",
                  "gretil_extracted.txt"]

    # Prefer merged, then individual sources
    input_file = None
    for name in candidates:
        f = input_dir / name
        if f.exists() and f.stat().st_size > 0:
            input_file = f
            break

    if input_file is None:
        print(f"  ERROR: No extracted data found for {lang} in {input_dir}")
        print(f"  Run stage1_extract first!")
        return

    print(f"  Input: {input_file} ({input_file.stat().st_size / (1024**2):.1f} MB)")

    # Read and clean lines
    print("  Step 1: Reading and normalizing...")
    raw_lines = []
    max_seen_len = 0
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            max_seen_len = max(max_seen_len, len(line))
            raw_lines.append(line)

    total_raw = len(raw_lines)
    print(f"    Raw lines: {total_raw:,}")
    print(f"    Longest raw line: {max_seen_len:,} chars"
          + (" (will be truncated during cleaning)" if max_seen_len > MAX_LINE_LEN else ""))

    # Step 2: Clean each line
    print("  Step 2: Cleaning lines (NFC + boilerplate + script validation)...")
    cleaned = []
    rejected_script = 0
    rejected_short = 0
    truncated_count = 0

    t_start = time.time()
    for idx, line in enumerate(tqdm(raw_lines, desc="Cleaning", unit="lines")):
        if len(line) > MAX_LINE_LEN:
            truncated_count += 1

        line = clean_line(line)

        # Periodic timing checkpoint so a future stall is self-diagnosing:
        # if this stops printing, the last index shown is where it hung.
        if idx > 0 and idx % LOG_EVERY == 0:
            elapsed = time.time() - t_start
            rate = idx / elapsed if elapsed > 0 else 0
            print(f"    ...at line {idx:,}/{total_raw:,} "
                  f"({elapsed:.1f}s elapsed, {rate:,.0f} lines/s)")

        # Skip empty lines (but preserve paragraph breaks)
        if not line:
            cleaned.append("")
            continue

        # Length check
        if len(line) < MIN_LINE_LENGTH:
            rejected_short += 1
            # Insert a boundary marker in place of the rejected line.
            # Without this, near_dedup_paragraphs() (which groups on
            # blank lines) silently fuses everything on either side of
            # a run of rejected lines into one paragraph. On a corpus
            # where ~65% of lines get rejected (as here), that collapses
            # 500k+ lines into 1-2 gigantic paragraphs and near-dedup
            # effectively hangs trying to hash them.
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        # Devanagari ratio check
        ratio = devanagari_ratio(line)
        if ratio < MIN_DEVANAGARI_RATIO:
            rejected_script += 1
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        cleaned.append(line)

    print(f"    After cleaning: {len(cleaned):,} lines")
    print(f"    Truncated (>{MAX_LINE_LEN:,} chars): {truncated_count:,}")
    print(f"    Rejected (script): {rejected_script:,} ({100*rejected_script/max(total_raw,1):.1f}%)")
    print(f"    Rejected (short):  {rejected_short:,} ({100*rejected_short/max(total_raw,1):.1f}%)")

    # Step 3: Exact dedup
    print("  Step 3: Exact deduplication...")
    deduped, exact_dup_count = deduplicate_lines(cleaned)
    print(f"    Exact duplicates removed: {exact_dup_count:,} "
          f"({100*exact_dup_count/max(len(cleaned),1):.1f}%)")

    # Step 4: Near-dedup
    print("  Step 4: Near-duplicate removal...")
    final_lines, near_dup_count = near_dedup_paragraphs(deduped)
    print(f"    Near-duplicates removed: {near_dup_count:,} paragraphs")

    # Step 5: Write cleaned output
    output_file = input_dir / "corpus_clean.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for line in final_lines:
            f.write(line + "\n")

    final_size = output_file.stat().st_size
    print(f"  ✓ Cleaned corpus: {output_file}")
    print(f"    Final size: {final_size / (1024**2):.1f} MB")

    # Step 6: Verify Devanagari purity
    print("  Step 5: Verifying script purity...")
    sample_size = min(10000, len(final_lines))
    sample_lines = [l for l in final_lines[:sample_size * 2] if l.strip()][:sample_size]
    if sample_lines:
        avg_ratio = sum(devanagari_ratio(l) for l in sample_lines) / len(sample_lines)
        print(f"    Average Devanagari ratio (sample): {100*avg_ratio:.1f}%")
        non_dev = 1.0 - avg_ratio
        status = "✓ PASS" if non_dev < 0.01 else "⚠ WARN (>1% non-Devanagari)"
        print(f"    Non-Devanagari content: {100*non_dev:.2f}% — {status}")

    # Write stats report
    stats = {
        "language": lang,
        "raw_lines": total_raw,
        "longest_raw_line_chars": max_seen_len,
        "truncated_lines": truncated_count,
        "rejected_script": rejected_script,
        "rejected_short": rejected_short,
        "exact_duplicates": exact_dup_count,
        "near_duplicates": near_dup_count,
        "final_lines": len(final_lines),
        "raw_size_mb": round(input_file.stat().st_size / (1024**2), 2),
        "clean_size_mb": round(final_size / (1024**2), 2),
    }
    stats_file = input_dir / "cleaning_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Stats saved to: {stats_file}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 1b: Clean extracted corpora")
    parser.add_argument("--lang", choices=LANGUAGES + ["all"], default="all",
                        help="Language to clean (default: all)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Only scan for pathological (very long) raw lines and exit, "
                             "without running the full cleaning pipeline")
    args = parser.parse_args()

    langs = LANGUAGES if args.lang == "all" else [args.lang]

    if args.diagnose:
        for lang in langs:
            input_dir = CLEAN_DIR / lang
            candidates = ["merged_raw.txt", "wiki_extracted.txt",
                          "cc100_extracted.txt", "l3cube_extracted.txt",
                          "gretil_extracted.txt"]
            input_file = None
            for name in candidates:
                f = input_dir / name
                if f.exists() and f.stat().st_size > 0:
                    input_file = f
                    break
            if input_file is None:
                print(f"  ERROR: No extracted data found for {lang} in {input_dir}")
                continue
            print(f"\n{'='*60}\n  DIAGNOSING: {lang.upper()}\n{'='*60}")
            find_pathological_lines(input_file)
        return

    for lang in langs:
        print(f"\n{'='*60}")
        print(f"  CLEANING: {lang.upper()}")
        print(f"{'='*60}")
        clean_corpus(lang)

    print("\n✓ Stage 1b (Cleaning) complete.")


if __name__ == "__main__":
    main()