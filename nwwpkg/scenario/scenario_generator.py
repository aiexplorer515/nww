# -*- coding: utf-8 -*-
"""
scenario_generator.py
블록 기반 위험 시나리오 자동 생성
"""

from typing import Dict

def run(block_id: str) -> Dict:
    """
    블록 ID를 받아 시나리오 생성
    """
    # GPT 연동 자리 (현재는 템플릿)
    return {
        "block_id": block_id,
        "scenario": f"블록 {block_id} 기반: 군사적 긴장 고조 → 외교 협상 결렬 가능성",
        "confidence": 0.6
    }
