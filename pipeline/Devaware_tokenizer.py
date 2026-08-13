"""
DevAwareTokenizer — end-to-end wrapper around the Devanagari-aware BPE model.
==============================================================================
The underlying SentencePiece model (devaware_bpe_{lang}.model) was trained on
PUA-remapped text, NOT raw Devanagari text (see pua_remap.py for why). That
means you can't just call spm.SentencePieceProcessor().Load(...).EncodeAsPieces(text)
on raw text the way Stage 2a does for the baseline tokenizers -- the model has
never seen a literal "क" character, only its PUA stand-in.

This class hides that: it does segmentation -> PUA remap -> SentencePiece for
encoding, and the reverse for decoding, so callers can treat it like an
ordinary tokenizer.
"""

from pathlib import Path

import sentencepiece as spm

from pipeline.pua_remap import AksharaPUAMap
from pipeline.stage2b_devanagari_tokenizer import segment_into_aksharas


class DevAwareTokenizer:
    def __init__(self, spm_model_path, pua_map_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(spm_model_path))
        self.pua_map = AksharaPUAMap.load(pua_map_path)

    @classmethod
    def for_lang(cls, lang: str) -> "DevAwareTokenizer":
        from pipeline.config import TOKENIZER_DIR
        tok_dir = TOKENIZER_DIR / lang
        return cls(
            spm_model_path=tok_dir / f"devaware_bpe_{lang}.model",
            pua_map_path=tok_dir / f"akshara_pua_map_{lang}.json",
        )

    # ─── Encoding ────────────────────────────────────────────────────────
    def _to_pua(self, text: str) -> str:
        segments = segment_into_aksharas(text)
        return self.pua_map.remap_segments(segments)

    def encode_as_pieces(self, text: str) -> list:
        return self.sp.EncodeAsPieces(self._to_pua(text))

    def encode(self, text: str) -> list:
        return self.sp.EncodeAsIds(self._to_pua(text))

    # ─── Decoding ────────────────────────────────────────────────────────
    def decode_pieces(self, pieces: list) -> str:
        pua_text = self.sp.DecodePieces(pieces)
        return self.pua_map.unmap_text(pua_text)

    def decode(self, ids: list) -> str:
        pua_text = self.sp.DecodeIds(ids)
        return self.pua_map.unmap_text(pua_text)

    # ─── Misc ────────────────────────────────────────────────────────────
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    def piece_to_devanagari(self, piece: str) -> str:
        """Decode a single SentencePiece piece back to Devanagari text,
        for inspecting what a merged multi-akshara token actually spells."""
        return self.pua_map.unmap_text(piece.replace("▁", " "))

    def roundtrip_ok(self, text: str) -> bool:
        return self.decode(self.encode(text)) == text