"""
Akshara <-> Private-Use-Area (PUA) codepoint remapping.
=========================================================
This is the fix for the Stage 2b bug where every akshara was joined with a
literal space (" ".join(segments)), which made SentencePiece's
split_by_whitespace=True treat every akshara as its own hard token boundary.
That protected aksharas from being split internally, but also made it
IMPOSSIBLE for BPE to ever merge two aksharas into one subword unit -- the
opposite of what a "Devanagari-aware" tokenizer is supposed to do.

The fix: instead of separating aksharas with spaces, collapse each distinct
akshara string into a single Private-Use-Area codepoint. A PUA codepoint is
atomic (one Unicode code point), so:
  - BPE can never split *inside* an akshara (there's nothing to split -- it's
    one symbol), which preserves the original grapheme-cluster protection.
  - BPE CAN merge multiple *adjacent* PUA codepoints into one token, because
    there's no separator between them anymore -- this is what lets the
    tokenizer learn genuine multi-akshara subword units.
  - Real whitespace (word boundaries) is left untouched as literal " ", so
    SentencePiece's word-boundary behavior is unaffected -- only in-word
    akshara-to-akshara merging changes.

The mapping is corpus-specific (built from the aksharas actually observed in
the training file) and is saved to disk alongside the tokenizer model, since
it's required to encode/decode text at inference time (see
devaware_tokenizer.py).
"""

import json
from pathlib import Path
from typing import Iterable, List


# ─── PUA code point pools ───────────────────────────────────────────────────
# BMP Private Use Area (6,400 code points) first, then the two supplementary
# private use planes (~65,534 code points each) if we ever need more than
# that -- in practice a single-language corpus has at most a few thousand
# distinct akshara strings, so we should never leave the BMP pool, but the
# supplementary planes are there as headroom rather than crashing.
_PUA_RANGES = [
    (0xE000, 0xF8FF),      # BMP Private Use Area
    (0xF0000, 0xFFFFD),    # Supplementary PUA-A
    (0x100000, 0x10FFFD),  # Supplementary PUA-B
]


def _pua_codepoints():
    for lo, hi in _PUA_RANGES:
        for cp in range(lo, hi + 1):
            yield cp


class AksharaPUAMap:
    """
    Bijective mapping between akshara strings (e.g. "क्ष", "ि", "मि") and
    single PUA code points. Deterministic given the same input akshara set
    (assigned in sorted order), so re-running Stage 2b on unchanged data
    reproduces the same mapping.
    """

    def __init__(self):
        self.akshara_to_pua: dict = {}
        self.pua_to_akshara: dict = {}

    # ─── Building ────────────────────────────────────────────────────────
    def build(self, aksharas: Iterable[str]) -> None:
        unique_sorted = sorted(set(aksharas))
        pua_iter = _pua_codepoints()
        for akshara in unique_sorted:
            try:
                cp = next(pua_iter)
            except StopIteration:
                raise RuntimeError(
                    f"Ran out of PUA code points while mapping akshara "
                    f"vocabulary (needed > "
                    f"{sum(hi - lo + 1 for lo, hi in _PUA_RANGES):,}). "
                    f"This should not happen for a single language corpus -- "
                    f"check for a bug upstream (e.g. bad segmentation "
                    f"producing near-unique 'aksharas' per line)."
                )
            ch = chr(cp)
            self.akshara_to_pua[akshara] = ch
            self.pua_to_akshara[ch] = akshara

    # ─── Persistence ─────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Store as codepoint ints (not raw chars) so the JSON is readable
        # and immune to any editor/tool mangling private-use characters.
        payload = {
            "akshara_to_pua_cp": {
                ak: ord(ch) for ak, ch in self.akshara_to_pua.items()
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "AksharaPUAMap":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        obj = cls()
        for ak, cp in payload["akshara_to_pua_cp"].items():
            ch = chr(cp)
            obj.akshara_to_pua[ak] = ch
            obj.pua_to_akshara[ch] = ak
        return obj

    @staticmethod
    def exists(path: Path) -> bool:
        return Path(path).exists()

    # ─── Encoding / decoding ─────────────────────────────────────────────
    def remap_segments(self, segments: List[str], space_marker: str = "▁") -> str:
        """
        Turn a list of segments (as produced by
        stage2b.segment_into_aksharas) into the PUA-remapped string that
        gets fed to SentencePiece training/inference.

        - Akshara segments known to the map -> single PUA char (no
          separator against neighboring aksharas, so BPE can merge them).
        - The internal space marker -> a literal " " (real word boundary,
          preserved so SentencePiece still splits words correctly).
        - Anything else (punctuation, digits, Latin letters, unmapped
          akshara-like strings at inference time on unseen data) -> passed
          through unchanged, one character at a time is fine since these
          were already emitted as single characters by segmentation.
        """
        out = []
        for seg in segments:
            if seg == space_marker:
                out.append(" ")
            elif seg in self.akshara_to_pua:
                out.append(self.akshara_to_pua[seg])
            else:
                # Unseen akshara (e.g. a conjunct that never appeared in
                # the training corpus) -- fall back to emitting it
                # literally. It will be handled by SentencePiece's
                # byte_fallback like any other OOV content, rather than
                # crashing. This should be rare; Stage 4 reports how often
                # it happens if you want to track it.
                out.append(seg)
        return "".join(out)

    def unmap_text(self, pua_text: str) -> str:
        """Inverse of remap_segments: PUA-encoded string -> original text."""
        out = []
        for ch in pua_text:
            out.append(self.pua_to_akshara.get(ch, ch))
        return "".join(out)