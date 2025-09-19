"""
Gating - 위기 지표 기반 필터링 모듈
입력: kyw_sum.jsonl
출력: gated.jsonl
"""

import os
import json
import logging
from typing import Dict, List
import re

logger = logging.getLogger(__name__)

class Gating:
    def __init__(self):
        # 위기 신호 체크리스트 (데모용)
        # weight = 중요도 (0~1)
        self.indicators = {
            "military": {
                "troop": 0.4, "army": 0.4, "missile": 0.5, "artillery": 0.5,
                "war": 0.6, "border": 0.3, "drill": 0.3
            },
            "diplomacy": {
                "negotiation": 0.4, "agreement": 0.5, "treaty": 0.5,
                "sanction": 0.6, "conflict": 0.5
            },
            "economy": {
                "inflation": 0.5, "market": 0.4, "currency": 0.5,
                "oil": 0.4, "energy": 0.5, "trade": 0.6
            }
        }
        self.threshold = 0.3  # 필터링 최소 점수

    def run(self, input_path: str, output_path: str):
        """입력 파일 → 필터링 → 출력 파일"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            for line in fin:
                try:
                    article = json.loads(line.strip())
                    gated = self._gate_article(article)
                    if gated:  # threshold 이상인 경우만 출력
                        fout.write(json.dumps(gated, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.error(f"[Gating] Failed: {e}")

        logger.info(f"[Gating] Completed. → {output_path}")

    def _gate_article(self, article: Dict) -> Dict:
        """단일 기사 게이팅"""
        text = (article.get("text") or "").lower()
        tokens = article.get("kw", [])
        frames = article.get("frames", [])

        scores = {}
        total_score = 0.0

        # 1) 키워드 기반 지표 점수화
        for domain, indicator_dict in self.indicators.items():
            score = 0.0
            for word, weight in indicator_dict.items():
                if word in text or word in tokens:
                    score += weight
            if domain in frames:
                score += 0.2  # 프레임 매칭 보너스
            if score > 0:
                scores[domain] = round(min(score, 1.0), 3)
                total_score += score

        # 2) 필터링: threshold 이상일 때만 유지
        if total_score >= self.threshold:
            article.update({
                "indicator_scores": scores,
                "indicator_total": round(total_score, 3),
                "gated": True
            })
            return article
        else:
            return None



