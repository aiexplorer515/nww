# -*- coding: utf-8 -*-
"""
토크나이저(tokenizer): 문장 분리(sentence split)
- 한국어/영어 혼용 간단 규칙 기반
"""
import re

_SENT_SPLIT = re.compile(r"(?<=[\.!?])\s+|(?<=[다요임음음요\)])\s+(?=[A-Z가-힣0-9])")

def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    # 1차 분리
    parts = _SENT_SPLIT.split(text)
    # 노이즈 제거
    sents = [s.strip() for s in parts if s and len(s.strip()) > 2]
    return sents[:2000]  # 안전 상한
