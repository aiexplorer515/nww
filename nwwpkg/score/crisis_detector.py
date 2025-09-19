# -*- coding: utf-8 -*-
"""
crisis_detector.py
블록 단위 위험도 계산
"""

from typing import List, Dict

def run(blocks: List[Dict]) -> Dict:
    """
    블록 리스트 받아 위기 점수 계산
    """
    if not blocks:
        return {"score": 0.0}

    # 단순 평균 점수 예시
    score = sum([0.7 for _ in blocks]) / len(blocks)

    return {"score": round(score, 2), "status": "경고" if score > 0.5 else "안정"}
