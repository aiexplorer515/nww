# -*- coding: utf-8 -*-
"""
Score Fusion(융합): 가중 평균 + 간단 캘리브레이션
"""
from __future__ import annotations

def combine(scores: list[float], weights: list[float] | None = None) -> float:
    if not scores:
        return 0.0
    if weights and len(weights) == len(scores):
        wsum = sum(weights) or 1.0
        val = sum(s * w for s, w in zip(scores, weights)) / wsum
    else:
        val = sum(scores) / len(scores)
    # 간이 캘리브레이션(살짝 보수적으로)
    calibrated = max(0.0, min(1.0, 0.95 * val))
    return float(round(calibrated, 4))
