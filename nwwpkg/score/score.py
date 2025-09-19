"""
NWW Scoring Package - IS + DBN + LLM Judge + Fusion
입력: gated.jsonl
출력: scores.jsonl
"""

import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# -------------------------------
# 1️⃣ Indicator Scoring (IS)
# -------------------------------
class ScoreIS:
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "military": 0.4,
            "diplomacy": 0.3,
            "economy": 0.3
        }

    def run(self, input_path: str, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            for line in fin:
                try:
                    article = json.loads(line.strip())
                    scored = self._score_article(article)
                    fout.write(json.dumps(scored, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.error(f"[ScoreIS] Failed: {e}")
        logger.info(f"[ScoreIS] Completed → {output_path}")

    def _score_article(self, article: Dict) -> Dict:
        ind_scores = article.get("indicator_scores", {})
        weighted_sum, total_weight = 0.0, 0.0

        for domain, score in ind_scores.items():
            w = self.weights.get(domain, 0.2)
            weighted_sum += score * w
            total_weight += w

        base_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        article["score_IS"] = round(base_score, 3)
        return article


# -------------------------------
# 2️⃣ DBN 기반 시계열 점수
# -------------------------------
class ScoreDBN:
    def __init__(self, decay: float = 0.8):
        self.decay = decay  # 과거 신호 반영 가중치

    def run(self, input_path: str, output_path: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"[ScoreDBN] Input not found: {input_path}")

        articles = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    articles.append(json.loads(line.strip()))
                except:
                    continue

        scored = self._apply_dbn(articles)

        with open(output_path, "w", encoding="utf-8") as fout:
            for a in scored:
                fout.write(json.dumps(a, ensure_ascii=False) + "\n")

        logger.info(f"[ScoreDBN] Completed → {output_path}")

    def _apply_dbn(self, articles: List[Dict]) -> List[Dict]:
        prev_score = 0.0
        for a in articles:
            is_score = a.get("score_IS", 0.0)
            dbn_score = (self.decay * prev_score) + ((1 - self.decay) * is_score)
            a["score_DBN"] = round(dbn_score, 3)
            prev_score = dbn_score
        return articles


# -------------------------------
# 3️⃣ LLM Judge (판정 기반 보정)
# -------------------------------
class LLMJudge:
    def __init__(self, factor: float = 0.1):
        self.factor = factor  # 보정 강도

    def run(self, input_path: str, output_path: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"[LLMJudge] Input not found: {input_path}")

        results = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    article = json.loads(line.strip())
                    scored = self._judge(article)
                    results.append(scored)
                except:
                    continue

        with open(output_path, "w", encoding="utf-8") as fout:
            for r in results:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")

        logger.info(f"[LLMJudge] Completed → {output_path}")

    def _judge(self, article: Dict) -> Dict:
        dbn_score = article.get("score_DBN", 0.0)
        # 간단한 규칙: 군사 기사면 보정치 +
        if "military" in article.get("indicator_scores", {}):
            adj = self.factor
        else:
            adj = 0.0
        article["score_LLM"] = round(min(1.0, dbn_score + adj), 3)
        return article


# -------------------------------
# 4️⃣ Fusion (최종 스코어 합성)
# -------------------------------
class FusionCalibration:
    def run(self, input_path: str, output_path: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"[Fusion] Input not found: {input_path}")

        fused = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    a = json.loads(line.strip())
                    fused.append(self._fuse(a))
                except:
                    continue

        with open(output_path, "w", encoding="utf-8") as fout:
            for r in fused:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")

        logger.info(f"[Fusion] Completed → {output_path}")

    def _fuse(self, article: Dict) -> Dict:
        s_is = article.get("score_IS", 0.0)
        s_dbn = article.get("score_DBN", 0.0)
        s_llm = article.get("score_LLM", 0.0)

        final_score = (0.4 * s_is) + (0.3 * s_dbn) + (0.3 * s_llm)
        article["score_final"] = round(final_score, 3)
        return article
