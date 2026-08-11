"""
Stage 1a — Data Extraction
===========================
Extracts raw text from the downloaded sources:
  - Hindi:    MediaWiki XML dump → plain text
  - Marathi:  CC100 plain text + L3Cube plain text → merged
  - Sanskrit: MediaWiki XML dump + GRETIL TEI-XML → plain text

Usage:
    python -m pipeline.stage1_extract [--lang hindi|marathi|sanskrit|all]
"""

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

from pipeline.config import (
    RAW_SOURCES, CLEAN_DIR, CORPUS_TARGETS, LANGUAGES, ensure_dirs
)


# ─── MediaWiki XML Extraction ────────────────────────────────────────────────

def extract_wiki_xml(xml_path: Path, output_path: Path, target_bytes: int):
    """
    Stream-parse a MediaWiki dump XML and extract article text.
    Uses iterparse to handle large files without loading into memory.
    """
    print(f"  Extracting Wikipedia XML: {xml_path}")
    print(f"  Target: {target_bytes / (1024**2):.1f} MB")

    bytes_written = 0
    articles = 0
    ns = ""  # Will detect namespace from the root element

    with open(output_path, "w", encoding="utf-8") as out_f:
        context = ET.iterparse(str(xml_path), events=("start", "end"))

        # Detect namespace from root
        for event, elem in context:
            if event == "start":
                # Extract namespace from root tag like {http://...}mediawiki
                match = re.match(r"\{(.+?)\}", elem.tag)
                if match:
                    ns = match.group(1)
                break

        ns_prefix = f"{{{ns}}}" if ns else ""
        current_title = ""
        in_page = False

        for event, elem in context:
            if event == "start" and elem.tag == f"{ns_prefix}page":
                in_page = True

            elif event == "end" and elem.tag == f"{ns_prefix}title" and in_page:
                current_title = (elem.text or "").strip()

            elif event == "end" and elem.tag == f"{ns_prefix}text" and in_page:
                text = (elem.text or "").strip()
                if text and len(text) > 50:
                    # Strip MediaWiki markup (basic)
                    text = _strip_wiki_markup(text)
                    if text and len(text) > 50:
                        text_bytes = text.encode("utf-8")
                        out_f.write(text + "\n\n")
                        bytes_written += len(text_bytes) + 2
                        articles += 1

                        if articles % 5000 == 0:
                            print(f"    Articles: {articles}, "
                                  f"Written: {bytes_written / (1024**2):.1f} MB")

                        if bytes_written >= target_bytes:
                            break

            elif event == "end" and elem.tag == f"{ns_prefix}page":
                in_page = False
                elem.clear()  # Free memory

    print(f"  ✓ Extracted {articles} articles, {bytes_written / (1024**2):.1f} MB")
    return bytes_written


def _strip_wiki_markup(text: str) -> str:
    """Remove common MediaWiki markup patterns."""
    # Remove XML/HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove templates {{ ... }} (non-greedy, single-level)
    text = re.sub(r"\{\{[^}]*\}\}", " ", text)
    # Remove [[ File: ... ]], [[ Image: ... ]]
    text = re.sub(r"\[\[(File|Image|चित्र|फ़ाइल|संचिका):[^\]]*\]\]", " ", text, flags=re.IGNORECASE)
    # Convert [[link|text]] → text, [[link]] → link
    text = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", text)
    # Remove external links [http...]
    text = re.sub(r"\[https?://[^\]]*\]", " ", text)
    # Remove remaining [...]
    text = re.sub(r"\[[^\]]*\]", " ", text)
    # Remove ''' and '' (bold/italic)
    text = re.sub(r"'{2,}", "", text)
    # Remove headings == ... ==
    text = re.sub(r"={2,}[^=]+={2,}", " ", text)
    # Remove * and # (list markers)
    text = re.sub(r"^[*#:;]+", "", text, flags=re.MULTILINE)
    # Remove | (table separators)
    text = re.sub(r"\|", " ", text)
    # Remove multiple spaces/newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are just refs or categories
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip category / reference-only lines
        if re.match(r"^(श्रेणी|Category|वर्ग|विभाग)\s*:", line, re.IGNORECASE):
            continue
        if line.startswith("REDIRECT") or line.startswith("#REDIRECT") or line.startswith("#पुनर्प्रेषित"):
            continue
        lines.append(line)

    return "\n".join(lines).strip()


# ─── GRETIL TEI-XML Extraction ───────────────────────────────────────────────

def extract_gretil_tei(tei_dir: Path, output_path: Path, target_bytes: int):
    """
    Extract text from GRETIL TEI-XML and plaintext files.
    GRETIL stores Sanskrit texts in TEI-XML format with <body> containing
    the actual text, and also has plaintext transformations.
    """
    print(f"  Extracting GRETIL from: {tei_dir}")

    bytes_written = 0
    files_processed = 0

    # First try plaintext files (transformations/plaintext/)
    plaintext_dir = tei_dir / "transformations" / "plaintext"
    txt_files = []
    if plaintext_dir.exists():
        txt_files = sorted(plaintext_dir.glob("*.txt"))

    # Also collect XML files from the TEI dir
    xml_files = sorted(tei_dir.glob("*.xml"))

    with open(output_path, "w", encoding="utf-8") as out_f:
        # Process plaintext files first (cleaner)
        for txt_file in tqdm(txt_files, desc="GRETIL plaintext"):
            try:
                text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
                if text and len(text) > 100:
                    # Basic cleanup
                    text = _clean_gretil_text(text)
                    if text:
                        text_bytes = text.encode("utf-8")
                        out_f.write(text + "\n\n")
                        bytes_written += len(text_bytes) + 2
                        files_processed += 1

                        if bytes_written >= target_bytes:
                            break
            except Exception as e:
                print(f"    Warning: Could not read {txt_file.name}: {e}")
                continue

        # If we still need more, process XML files
        if bytes_written < target_bytes:
            for xml_file in tqdm(xml_files, desc="GRETIL XML"):
                try:
                    text = _extract_tei_body(xml_file)
                    if text and len(text) > 100:
                        text = _clean_gretil_text(text)
                        if text:
                            text_bytes = text.encode("utf-8")
                            out_f.write(text + "\n\n")
                            bytes_written += len(text_bytes) + 2
                            files_processed += 1

                            if bytes_written >= target_bytes:
                                break
                except Exception as e:
                    continue

    print(f"  ✓ Extracted {files_processed} GRETIL files, {bytes_written / (1024**2):.1f} MB")
    return bytes_written


def _extract_tei_body(xml_path: Path) -> str:
    """Extract text content from TEI-XML body."""
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        # Handle TEI namespace
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}

        # Try with namespace
        body = root.find(".//tei:body", ns)
        if body is None:
            # Try without namespace
            body = root.find(".//body")
        if body is None:
            # Try getting all text
            return ET.tostring(root, encoding="unicode", method="text")

        return ET.tostring(body, encoding="unicode", method="text")
    except ET.ParseError:
        return ""


def _clean_gretil_text(text: str) -> str:
    """Clean GRETIL-specific markup and metadata."""
    # Remove common GRETIL headers/footers
    lines = []
    skip_header = True
    for line in text.split("\n"):
        line = line.strip()
        # Skip header lines (usually metadata, encoding info, etc.)
        if skip_header:
            if re.match(r"^(##|<\?|<!|encoding|GRETIL|http|Input|This|Based|Göttingen)", line, re.IGNORECASE):
                continue
            if not line:
                continue
            skip_header = False

        # Skip reference markers and numbers-only lines
        if re.match(r"^[\d.]+$", line):
            continue
        if not line:
            continue

        lines.append(line)

    return "\n".join(lines).strip()


# ─── Plain Text Extraction (CC100, L3Cube) ──────────────────────────────────

def extract_plaintext(input_path: Path, output_path: Path, target_bytes: int):
    """
    Extract lines from a large plain-text file up to the target size.
    Used for CC100 Marathi and L3Cube corpus.
    """
    print(f"  Extracting plain text: {input_path}")
    print(f"  Target: {target_bytes / (1024**2):.1f} MB")

    bytes_written = 0
    lines_read = 0

    with open(input_path, "r", encoding="utf-8", errors="ignore") as in_f, \
         open(output_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue

            line_bytes = len(line.encode("utf-8"))
            out_f.write(line + "\n")
            bytes_written += line_bytes + 1
            lines_read += 1

            if lines_read % 500_000 == 0:
                print(f"    Lines: {lines_read:,}, Written: {bytes_written / (1024**2):.1f} MB")

            if bytes_written >= target_bytes:
                break

    print(f"  ✓ Extracted {lines_read:,} lines, {bytes_written / (1024**2):.1f} MB")
    return bytes_written


# ─── Merge Multiple Sources ─────────────────────────────────────────────────

def merge_sources(source_files: list, output_path: Path, target_bytes: int):
    """
    Merge multiple extracted source files into one, up to target size.

    IMPORTANT: gives each source its own slice of target_bytes
    (target_bytes // len(source_files)), rather than letting whichever
    source is listed first consume the entire budget. The previous
    version processed sources in list order and stopped the moment the
    *overall* target was hit, so a large first source (e.g. GRETIL for
    Sanskrit) could fill the whole quota before later sources (e.g.
    Wikipedia) were ever opened -- silently zeroing out their
    contribution to the merged corpus.
    """
    existing = [src for src in source_files if src.exists()]
    for src in source_files:
        if src not in existing:
            print(f"    Warning: {src} not found, skipping")

    if not existing:
        print(f"  ✓ Merged to 0.0 MB (no sources found)")
        return 0

    per_source_target = target_bytes // len(existing)
    print(f"  Merging {len(existing)} sources → {output_path.name} "
          f"(~{per_source_target / (1024**2):.1f} MB per source)")

    bytes_written = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for src in existing:
            src_bytes = 0
            with open(src, "r", encoding="utf-8", errors="ignore") as in_f:
                for line in in_f:
                    line_bytes = len(line.encode("utf-8"))
                    out_f.write(line)
                    bytes_written += line_bytes
                    src_bytes += line_bytes
                    if src_bytes >= per_source_target:
                        break
            print(f"    {src.name}: contributed {src_bytes / (1024**2):.1f} MB")

    print(f"  ✓ Merged to {bytes_written / (1024**2):.1f} MB")
    return bytes_written


# ─── Per-Language Extraction ────────────────────────────────────────────────

def extract_hindi():
    """Extract Hindi from Wikipedia XML dump."""
    ensure_dirs()
    out_dir = CLEAN_DIR / "hindi"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_SOURCES["hindi"]["wiki_xml"]

    if not raw_path.exists():
        print(f"  ERROR: Hindi wiki dump not found at {raw_path}")
        return

    # We want ~3x the target for post-cleaning shrinkage
    extract_target = min(CORPUS_TARGETS["hindi"] * 3,
                         os.path.getsize(raw_path))

    wiki_out = out_dir / "wiki_extracted.txt"
    extract_wiki_xml(raw_path, wiki_out, extract_target)

    # Hindi has only one source at this stage, so the extracted file is the raw pool
    print(f"  Hindi extraction complete → {wiki_out}")


def extract_marathi():
    """Extract Marathi from CC100 + L3Cube."""
    ensure_dirs()
    out_dir = CLEAN_DIR / "marathi"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract from each source (up to 150MB each, so we have ~300MB raw pool)
    per_source_target = CORPUS_TARGETS["marathi"] * 2  # 200 MB each

    cc100_out = out_dir / "cc100_extracted.txt"
    cc100_path = RAW_SOURCES["marathi"]["cc100"]
    if cc100_path.exists():
        extract_plaintext(cc100_path, cc100_out, per_source_target)
    else:
        print(f"  Warning: CC100 not found at {cc100_path}")

    l3cube_out = out_dir / "l3cube_extracted.txt"
    l3cube_path = RAW_SOURCES["marathi"]["l3cube"]
    if l3cube_path.exists():
        extract_plaintext(l3cube_path, l3cube_out, per_source_target)
    else:
        print(f"  Warning: L3Cube not found at {l3cube_path}")

    # Merge both sources
    sources = [f for f in [cc100_out, l3cube_out] if f.exists()]
    if sources:
        merged = out_dir / "merged_raw.txt"
        merge_sources(sources, merged, CORPUS_TARGETS["marathi"] * 3)
        print(f"  Marathi extraction complete → {merged}")


def extract_sanskrit():
    """Extract Sanskrit from Wikipedia + GRETIL."""
    ensure_dirs()
    out_dir = CLEAN_DIR / "sanskrit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # GRETIL
    gretil_out = out_dir / "gretil_extracted.txt"
    gretil_dir = RAW_SOURCES["sanskrit"]["gretil_dir"]
    if gretil_dir.exists():
        extract_gretil_tei(gretil_dir, gretil_out, CORPUS_TARGETS["sanskrit"] * 3)
    else:
        print(f"  Warning: GRETIL dir not found at {gretil_dir}")

    # Wikipedia
    wiki_out = out_dir / "wiki_extracted.txt"
    wiki_path = RAW_SOURCES["sanskrit"]["wiki_xml"]
    if wiki_path.exists():
        extract_wiki_xml(wiki_path, wiki_out, CORPUS_TARGETS["sanskrit"] * 3)
    else:
        print(f"  Warning: Sanskrit wiki not found at {wiki_path}")

    # Merge
    sources = [f for f in [gretil_out, wiki_out] if f.exists()]
    if sources:
        merged = out_dir / "merged_raw.txt"
        merge_sources(sources, merged, CORPUS_TARGETS["sanskrit"] * 3)
        print(f"  Sanskrit extraction complete → {merged}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

EXTRACTORS = {
    "hindi": extract_hindi,
    "marathi": extract_marathi,
    "sanskrit": extract_sanskrit,
}


def main():
    parser = argparse.ArgumentParser(description="Stage 1a: Extract raw text from downloaded sources")
    parser.add_argument("--lang", choices=LANGUAGES + ["all"], default="all",
                        help="Language to extract (default: all)")
    args = parser.parse_args()

    langs = LANGUAGES if args.lang == "all" else [args.lang]

    for lang in langs:
        print(f"\n{'='*60}")
        print(f"  EXTRACTING: {lang.upper()}")
        print(f"{'='*60}")
        EXTRACTORS[lang]()

    print("\n✓ Stage 1a (Extraction) complete.")


if __name__ == "__main__":
    main()