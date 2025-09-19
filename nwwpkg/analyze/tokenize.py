# -*- coding: utf-8 -*-
from kiwipiepy import Kiwi
import re

_KIWI = Kiwi(num_workers=1)
_KEEP_POS = {"NNG","NNP","SL","SN"}  # 보통명사/고유명사/외래어/숫자
_STOP = set("""
기사|원문|입력|오후|사진|연합|연합뉴스|YTN|newsis|kmn|서비스|보내기|관련|본문|글자|수정|변화
""".strip().split("|"))

def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\u200b\ufeff]+", "", s)
    return s.strip()

def tokenize_ko(s: str):
    s = normalize_text(s)
    toks = []
    for w in _KIWI.tokenize(s):
        if w.tag in _KEEP_POS:
            t = w.form.lower()
            if len(t) < 2: 
                continue
            if t in _STOP:
                continue
            toks.append(t)
    return toks
