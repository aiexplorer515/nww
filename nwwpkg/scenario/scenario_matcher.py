# nwwpkg/scenario/scenario_matcher.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from math import sqrt
from typing import List, Dict
from nwwpkg.preprocess import embedder as _emb

# 간단 시나리오 코퍼스(설명문 기반)
_SCENARIOS = [
    {"id": 1, "title": "국경 분쟁 격화", "desc": "군사적 충돌, 병력 증강, 포격, 미사일 발사"},
    {"id": 2, "title": "경제 제재 강화", "desc": "제재 확대, 무역 제한, 환율 변동, 시장 불안"},
    {"id": 3, "title": "외교 협상 진전", "desc": "정상 회담, 합의, 중재, 외교 관계 개선"},
    {"id": 4, "title": "사이버 보안 위협", "desc": "해킹, 랜섬웨어, 데이터 침해, 보안 강화"},
]

def _mean(vecs: List[List[float]]) -> List[float]:
    if not vecs:
        return []
    n = len(vecs)
    m = [0.0]*len(vecs[0])
    for v in vecs:
        for i, x in enumerate(v):
            m[i] += x
    return [x/n for x in m]

def _cos(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = sqrt(sum(x*x for x in a)) or 1e-9
    nb = sqrt(sum(y*y for y in b)) or 1e-9
    return max(0.0, min(1.0, dot/(na*nb)))

def match(article_vecs: List[List[float]], top_k: int = 3) -> List[Dict]:
    """
    함수형 API: 기사 문장 임베딩 리스트 -> 시나리오 후보 상위 K
    """
    art = _mean(article_vecs)
    if not art:
        return []
    scen_vecs = _emb.embed([s["desc"] for s in _SCENARIOS])
    out = []
    for s, v in zip(_SCENARIOS, scen_vecs):
        sim = _cos(art, v)
        out.append({"scenario_id": s["id"], "title": s["title"], "similarity": round(sim, 3)})
    out.sort(key=lambda x: x["similarity"], reverse=True)
    return out[:top_k]

# ✅ 백워드 호환용 클래스 래퍼 (package __init__가 클래스를 import해도 동작)
class ScenarioMatcher:
    @staticmethod
    def match(article_vecs: List[List[float]], top_k: int = 3) -> List[Dict]:
        return match(article_vecs, top_k=top_k)
