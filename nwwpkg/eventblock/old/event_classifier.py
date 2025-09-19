# -*- coding: utf-8 -*-
"""
event_classifier.py
실데이터 기반 사건 분류기
- NER(인물·국가 추출)
- 프레임 태깅 (군사/외교/경제/사회)
"""

from typing import List, Dict, Any
import re
import spacy

# spaCy 한국어/영어 모델 (사전 설치 필요: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None  # spaCy 미사용 환경에서는 fallback

# 프레임 태깅 규칙 사전
FRAME_RULES = {
    "군사": ["미사일", "발사", "훈련", "군사", "무기", "전쟁", "병력"],
    "외교": ["회담", "협상", "결렬", "제재", "외교", "조약", "합의"],
    "경제": ["주가", "무역", "관세", "수출", "수입", "금융", "환율", "원유"],
    "사회": ["시위", "선거", "언론", "여론", "정치", "시민", "인권"],
}


def extract_actors(text: str) -> List[str]:
    """텍스트에서 인물/국가 엔티티 추출"""
    actors = set()

    # 1️⃣ spaCy 기반 NER
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "GPE", "ORG"]:  # 인물, 지명, 조직
                actors.add(ent.text)

    # 2️⃣ 정규식 기반 (한글 이름/국가 fallback)
    patterns = [
        r"[가-힣]{2,3}",       # 한국어 이름 후보
        r"(북한|한국|미국|중국|일본|러시아)",  # 주요 국가
    ]
    for p in patterns:
        for m in re.findall(p, text):
            actors.add(m)

    return list(actors)


def tag_frame(text: str) -> str:
    """텍스트에서 프레임 태깅"""
    for frame, keywords in FRAME_RULES.items():
        if any(kw in text for kw in keywords):
            return frame
    return "기타"


def run(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    기사 리스트 입력 → 사건 리스트 반환
    :param articles: [{"id": str, "text": str, "time": str}, ...]
    :return: 사건 리스트 [{"id", "text", "time", "actors", "frame"}]
    """
    events = []
    for art in articles:
        text = art.get("text", "")
        actors = extract_actors(text)
        frame = tag_frame(text)

        event = {
            "id": art.get("id"),
            "text": text,
            "time": art.get("time"),
            "actors": actors,
            "frame": frame,
        }
        events.append(event)

    return events


if __name__ == "__main__":
    # 샘플 실행
    sample_articles = [
        {"id": "a1", "text": "북한이 미사일 발사를 강행했다.", "time": "2025-09-17"},
        {"id": "a2", "text": "한국과 미국이 공동 군사 훈련을 실시했다.", "time": "2025-09-17"},
        {"id": "a3", "text": "한중 무역 협상이 결렬되었다.", "time": "2025-09-16"},
        {"id": "a4", "text": "서울에서 대규모 시위가 발생했다.", "time": "2025-09-15"},
    ]
    events = run(sample_articles)
    from pprint import pprint
    pprint(events)
