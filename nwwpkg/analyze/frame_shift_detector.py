# -*- coding: utf-8 -*-
"""
프레임 시프트 감지: 최근 N 스냅샷 간 자카드 유사도 기반 변화 스코어
"""
from __future__ import annotations

def detect(history: list[list[dict]]) -> dict:
    """
    history: [frames_t1, frames_t2, ...] (각 원소는 frame dict 리스트)
    """
    if not history or len(history) < 2:
        return {"shift_detected": False, "shift_score": 0.0, "changed": []}

    def toset(frlist): return set(f.get("frame") for f in frlist)
    last = toset(history[-1]); prev = toset(history[-2])
    inter = len(last & prev)
    union = len(last | prev) or 1
    jacc = inter/union
    change = 1.0 - jacc
    changed = list((last ^ prev))
    return {"shift_detected": change >= 0.4, "shift_score": round(change, 3), "changed": changed}
