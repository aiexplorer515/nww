# -*- coding: utf-8 -*-
"""
영향 분석(impact): 프레임·점수 → 도메인별 영향 요약
"""
from __future__ import annotations

def run(frames: list[dict], fused_score: float) -> dict:
    impacts = set()
    fset = {f.get("frame") for f in frames}
    if "Military" in fset:
        impacts.add("군사적 긴장 고조 및 인접 지역 위험 증가")
    if "Economy" in fset:
        impacts.add("환율·금리 변동 및 교역 위축 가능성")
    if "Diplomacy" in fset:
        impacts.add("외교 라인 재조정/중재 강화")
    if "Security" in fset:
        impacts.add("사이버 보안 리스크 증대 및 대응 비용 증가")
    if "CivilUnrest" in fset:
        impacts.add("국내 사회 불안 및 치안 비용 상승")
    if "Health" in fset:
        impacts.add("보건 자원 수요 증가 및 이동 제한 위험")

    severity = "Low"
    if fused_score > 0.7: severity = "High"
    elif fused_score > 0.4: severity = "Medium"

    return {
        "severity": severity,
        "fused_score": fused_score,
        "impact_summary": sorted(list(impacts))
    }
