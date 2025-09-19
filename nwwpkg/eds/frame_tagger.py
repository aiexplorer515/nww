# -*- coding: utf-8 -*-
"""
프레임 태거(frame_tagger): ESD(전문가 사전) 기반 규칙 + 신뢰도 계산
"""
from __future__ import annotations
import re
from math import tanh

# (frame_name, weight, pattern)
_RULES = [
    ("Military", 0.9, r"(군대|전쟁|병력|포격|미사일|무력|교전|훈련)"),
    ("Economy",  0.7, r"(경제|무역|제재|환율|수출|수입|인플레이션|금리|실업)"),
    ("Diplomacy",0.6, r"(외교|협상|회담|합의|중재|대사|정상회담)"),
    ("Security", 0.75,r"(테러|폭탄|사이버|해킹|보안|침해|랜섬웨어)"),
    ("CivilUnrest",0.65,r"(시위|집회|폭동|파업|점거|충돌|진압)"),
    ("Health",   0.5, r"(감염|전염|질병|백신|팬데믹|확진|격리)"),
]

_COMPILED = [(name, w, re.compile(p)) for name, w, p in _RULES]

def tag(text: str) -> list[dict]:
    if not text:
        return [{"frame": "General", "matches": [], "score": 0.1, "confidence": 0.3}]
    results = []
    for name, w, pat in _COMPILED:
        matches = pat.findall(text)
        if matches:
            mcount = len(matches)
            # 간단한 점수/신뢰도: 발생량과 가중치 기반
            raw = w * (1.0 + 0.2 * (mcount - 1))
            score = min(1.0, raw)
            conf = 0.5 + 0.5 * tanh(0.5 * mcount)  # 0.5~1.0
            results.append({
                "frame": name,
                "matches": list(set(matches)),
                "count": mcount,
                "score": round(score, 3),
                "confidence": round(conf, 3),
            })
    if not results:
        results.append({"frame": "General", "matches": [], "count": 0, "score": 0.2, "confidence": 0.4})
    return results
