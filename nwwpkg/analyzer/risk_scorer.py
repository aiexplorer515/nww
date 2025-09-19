# -*- coding: utf-8 -*-
"""
risk_scorer.py
위험 점수 계산
"""

from typing import Dict, List

def run(actor_report: List[Dict], network_report: Dict, frame_report: Dict) -> float:
    """
    단순 위험 점수 계산
    :param actor_report: [{"actor": str, "mentions": int}]
    :param network_report: {"centrality": {...}}
    :param frame_report: {"transition": str}
    :return: 위험 점수 (0~1)
    """
    num_actors = len(actor_report)
    avg_mentions = sum(a["mentions"] for a in actor_report) / num_actors if num_actors > 0 else 0
    max_centrality = max(network_report.get("centrality", {}).values(), default=0)

    # 프레임 변화 점수
    frame_score = 0.5 if "→" in frame_report.get("transition", "") else 0.2

    score = min(1.0, 0.3 * avg_mentions + 0.4 * max_centrality + 0.3 * frame_score)
    return round(score, 2)
