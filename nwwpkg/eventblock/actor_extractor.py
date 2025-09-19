# -*- coding: utf-8 -*-
"""
actor_extractor.py
실제 기사 텍스트에서 인물(Actor) 추출 및 집계
"""

from collections import Counter
from typing import List, Dict
from transformers import pipeline

# ✅ 한국어/다국어 지원 가능한 NER 파이프라인 로드
# (최초 실행 시 모델 다운로드 필요: ~400MB)
ner = pipeline("ner", model="bert-base-multilingual-cased", aggregation_strategy="simple")

# ✅ alias 매핑 (이후 DB/JSON으로 확장 가능)
ALIAS_MAP = {
    "김 위원장": "김정은",
    "정은": "김정은",
    "문 대통령": "문재인",
    "윤 대통령": "윤석열",
}

def normalize_name(name: str) -> str:
    """별칭을 표준 이름으로 정규화"""
    return ALIAS_MAP.get(name, name)

def extract_actors(text: str) -> List[str]:
    """NER 기반 인물 추출"""
    if not text.strip():
        return []

    entities = ner(text)
    actors = []
    for ent in entities:
        if ent["entity_group"] == "PER":  # Person
            name = normalize_name(ent["word"])
            actors.append(name)

    return actors

def run(events: List[Dict]) -> List[Dict]:
    """
    블록 내 모든 사건에서 인물 추출 후 집계
    :param events: 사건 리스트
    :return: [{actor, mentions}]
    """
    all_actors = []
    for evt in events:
        text = evt.get("content", "")
        all_actors.extend(extract_actors(text))

    counter = Counter(all_actors)
    results = [{"actor": a, "mentions": m} for a, m in counter.most_common(10)]
    return results
