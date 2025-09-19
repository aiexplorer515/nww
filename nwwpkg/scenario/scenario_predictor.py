# nwwpkg/scenario/scenario_predictor.py
# -*- coding: utf-8 -*-
from __future__ import annotations

def generate(text: str) -> dict:
    if not text:
        return {"scenario": "정보 부족으로 관망", "confidence": 0.3}
    t = text
    if any(k in t for k in ["군대", "전쟁", "포격", "미사일"]):
        return {"scenario": "단기적 군사적 긴장 고조 및 국지 충돌 가능성", "confidence": 0.7}
    if any(k in t for k in ["경제", "무역", "제재", "환율"]):
        return {"scenario": "경제 제재/무역 갈등 심화와 시장 변동성 확대", "confidence": 0.65}
    if any(k in t for k in ["외교", "협상", "회담", "합의"]):
        return {"scenario": "외교적 협상 재개 및 단계적 긴장 완화", "confidence": 0.6}
    if any(k in t for k in ["해킹", "랜섬웨어", "보안", "침해"]):
        return {"scenario": "사이버 공격 대비 강화 및 추가 침해 위험", "confidence": 0.6}
    return {"scenario": "상황 관망 및 정보 수집 지속", "confidence": 0.45}

# ✅ 백워드 호환용 클래스 래퍼
class ScenarioPredictor:
    @staticmethod
    def generate(text: str) -> dict:
        return generate(text)
