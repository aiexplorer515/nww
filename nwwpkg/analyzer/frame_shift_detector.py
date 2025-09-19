# -*- coding: utf-8 -*-
"""
frame_shift_detector.py
프레임 변화 탐지
"""

from typing import List, Dict
import pandas as pd

def run(events: List[Dict]) -> Dict:
    """
    사건 시계열 프레임 변화를 탐지
    :param events: [{ "time": str, "frame": str}, ...]
    :return: {"frames": [...], "transition": str}
    """
    frames = [ev.get("frame", "unknown") for ev in events]
    if not frames:
        return {"frames": [], "transition": "없음"}

    # 단순 시계열 변화를 감지 (첫 → 마지막)
    transition = f"{frames[0]} → {frames[-1]}" if len(frames) > 1 else frames[0]

    frame_counts = pd.Series(frames).value_counts().to_dict()

    return {
        "frames": frames,
        "transition": transition,
        "counts": frame_counts
    }
