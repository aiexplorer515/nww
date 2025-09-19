# -*- coding: utf-8 -*-
"""
최종 경보 결정(decider): 점수 → 등급/이모지
"""
from __future__ import annotations

def decide(final_score: float) -> str:
    if final_score > 0.7:
        return "🚨 High Risk"
    if final_score > 0.4:
        return "⚠️ Medium Risk"
    return "✅ Low Risk"
