# -*- coding: utf-8 -*-
"""
Indicator Scorer: 프레임 → 지표 점수(0~1)
"""
from __future__ import annotations

_WEIGHTS = {
    "Military": 0.35,
    "Economy": 0.25,
    "Diplomacy": 0.15,
    "Security": 0.25,
    "CivilUnrest": 0.2,
    "Health": 0.2,
    "General": 0.05
}

def run(frames: list[dict]) -> float:
    if not frames:
        return 0.0
    s = 0.0
    for f in frames:
        w = _WEIGHTS.get(f.get("frame"), 0.1)
        s += w * float(f.get("score", 0.0))
    return min(1.0, round(s, 4))
