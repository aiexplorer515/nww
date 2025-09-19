# -*- coding: utf-8 -*-
"""
DBN Inference(간이): 현재 프레임 vs 직전 프레임 변화율 기반 가점
"""
from __future__ import annotations

def _to_set(frames: list[dict]) -> set[str]:
    return set(f.get("frame") for f in frames if f.get("frame"))

def run(current_frames: list[dict], prev_frames: list[dict] | None = None) -> float:
    if not current_frames:
        return 0.0
    if not prev_frames:
        # 과거 정보가 없으면 낮은 가점
        return min(1.0, 0.1 * len(current_frames))

    cur = _to_set(current_frames)
    prv = _to_set(prev_frames)
    if not cur and not prv:
        return 0.0

    # 자카드 거리 기반 변화율
    inter = len(cur & prv)
    union = len(cur | prv) or 1
    jaccard = inter / union
    change = 1.0 - jaccard  # 변화가 클수록 1에 가까움
    score = 0.4 * change + 0.1 * len(cur - prv)  # 신규 프레임 보너스
    return float(round(min(1.0, score), 4))
