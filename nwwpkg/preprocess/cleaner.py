# -*- coding: utf-8 -*-
"""
전처리(cleaner): 텍스트 정규화(normalization)
"""
import re
import unicodedata

_ZERO_WIDTH = re.compile(r"[\u200B-\u200D\uFEFF]")
_MULTI_SPACE = re.compile(r"\s+")
_QUOTES = {
    "“": "\"", "”": "\"", "‘": "'", "’": "'",
    "–": "-", "—": "-", "…": "...",
}

def normalize(text: str) -> str:
    if not text:
        return ""
    # 유니코드 정규화(NFC)
    text = unicodedata.normalize("NFC", text)
    # 제로폭 제거
    text = _ZERO_WIDTH.sub("", text)
    # 특수 따옴표 등 교정
    for src, tgt in _QUOTES.items():
        text = text.replace(src, tgt)
    # 공백 정리
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text
