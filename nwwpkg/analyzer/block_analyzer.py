# -*- coding: utf-8 -*-
"""
block_analyzer.py
블록 단위 인물·네트워크·프레임·위험도 분석 종합
"""

from typing import Dict, Any
from nwwpkg.analyzer import actor_extractor, network_builder, frame_shift_detector, risk_scorer

def run(block: Dict[str, Any]) -> Dict[str, Any]:
    """
    블록 단위 분석 실행
    :param block: {"id": str, "events": [ {content, actors, frame, time...}, ... ] }
    :return: 분석 리포트 dict
    """
    block_id = block.get("id", "unknown")
    events = block.get("events", [])

    # 1️⃣ 인물 분석
    actor_report = actor_extractor.run(events)

    # 2️⃣ 네트워크 분석
    network_report = network_builder.run(events)

    # 3️⃣ 프레임 변화
    frame_report = frame_shift_detector.run(events)

    # 4️⃣ 위험 점수
    risk_score = risk_scorer.run(actor_report, network_report, frame_report)

    # 5️⃣ 종합 리포트
    report = {
        "block_id": block_id,
        "num_events": len(events),
        "actors": actor_report,
        "network": network_report,
        "frame_shift": frame_report,
        "risk_score": risk_score,
        "summary": f"블록 {block_id}: 주요 인물 {actor_report[0]['actor'] if actor_report else 'N/A'}, "
                   f"위험도 {risk_score:.2f}, 프레임 {frame_report['transition']}"
    }

    return report
