# -*- coding: utf-8 -*-
"""
LLM Judge(설명 생성 대체): 점수/프레임 기반 규칙적 설명문 생성
- 실제 배포 시 LLM API로 교체 가능
"""
from __future__ import annotations

def explain(scores: dict, frames: list[dict] | None = None) -> str:
    lines = []
    fused = scores.get("fused", 0.0)
    ind = scores.get("indicator", 0.0)
    dbn = scores.get("dbn", 0.0)

    def level(x):
        return "높음" if x>0.7 else ("중간" if x>0.4 else "낮음")

    lines.append(f"- 종합 점수(fused): {fused:.2f} ({level(fused)})")
    lines.append(f"- 지표 점수(indicator): {ind:.2f} ({level(ind)})")
    lines.append(f"- 시계열 점수(dbn): {dbn:.2f} ({level(dbn)})")

    if frames:
        top = sorted(frames, key=lambda f: f.get("score",0), reverse=True)[:3]
        tops = ", ".join(f"{t['frame']}({t.get('score',0):.2f})" for t in top)
        lines.append(f"- 주요 프레임: {tops}")

    if fused>0.7:
        lines.append("=> 단기 주의 필요. 추가 증거와 출처 교차검증 권고.")
    elif fused>0.4:
        lines.append("=> 경계 단계. 동향 모니터링 및 리스크 완화책 준비 권고.")
    else:
        lines.append("=> 저위험. 추적 모니터링 수준 유지.")

    return "\n".join(lines)
