"""
Tagger - 뉴스 기사 분석 모듈
articles.norm.jsonl → kyw_sum.jsonl
"""

import os
import json
import logging
from typing import Dict, List
import re

import nltk
from nltk.corpus import stopwords
from collections import Counter

# NLTK 자원 준비
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

logger = logging.getLogger(__name__)


class Tagger:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        # 간단한 프레임 사전 (데모용)
        self.frame_dict = {
            "military": ["troop", "army", "missile", "artillery", "war"],
            "diplomacy": ["negotiation", "treaty", "agreement", "diplomatic"],
            "economy": ["sanction", "trade", "economy", "inflation"]
        }

    def run(self, input_path: str, output_path: str):
        """입력 파일 → 분석 → 출력 파일"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            for line in fin:
                try:
                    article = json.loads(line.strip())
                    tagged = self._analyze_article(article)
                    fout.write(json.dumps(tagged, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.error(f"[Tagger] Failed: {e}")

        logger.info(f"[Tagger] Completed. → {output_path}")

    def _analyze_article(self, article: Dict) -> Dict:
        """단일 기사 분석"""
        text = article.get("text", "")
        if not text:
            return article

        # 1) 키워드 추출
        tokens = self._tokenize(text)
        keywords = self._extract_keywords(tokens)

        # 2) 요약 (문장 앞부분 기반 간단 요약)
        summary = self._summarize(text)

        # 3) 엔티티 추출 (간단 NER 모사)
        actors = self._extract_entities(text)

        # 4) 프레임 태깅
        frames = self._tag_frames(tokens)

        # 결과 병합
        article.update({
            "kw": keywords,
            "summary": summary,
            "actors": actors,
            "frames": frames
        })
        return article

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"\b[a-zA-Z가-힣]+\b", text.lower())
        return [t for t in tokens if t not in self.stop_words and len(t) > 2]

    def _extract_keywords(self, tokens: List[str], topn: int = 5) -> List[str]:
        counter = Counter(tokens)
        return [w for w, _ in counter.most_common(topn)]

    def _summarize(self, text: str, max_sent: int = 2) -> str:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        return " ".join(sentences[:max_sent])

    def _extract_entities(self, text: str) -> List[str]:
        # 아주 단순한 엔티티 추출 (대문자로 시작하는 단어)
        return list(set(re.findall(r"\b[A-Z][a-zA-Z]+\b", text)))

    def _tag_frames(self, tokens: List[str]) -> List[str]:
        matched_frames = []
        for frame, vocab in self.frame_dict.items():
            if any(tok in vocab for tok in tokens):
                matched_frames.append(frame)
        return matched_frames
