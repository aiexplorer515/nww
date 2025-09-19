# -*- coding: utf-8 -*-
"""
event_classifier.py
- 입력: data/bundles/{bundle}/entities.jsonl
- 출력: data/bundles/{bundle}/events.jsonl
- 역할: 전처리된 엔티티 기반 사건 구조화 + 프레임 태깅
"""

from typing import Dict, Any, List
import json
from pathlib import Path

# 프레임 태깅 규칙 사전
FRAME_RULES = {
    "군사": ["미사일", "발사", "훈련", "군사", "무기", "전쟁", "병력"],
    "외교": ["회담", "협상", "결렬", "제재", "외교", "조약", "합의"],
    "경제": ["주가", "무역", "관세", "수출", "수입", "금융", "환율", "원유"],
    "사회": ["시위", "선거", "언론", "여론", "정치", "시민", "인권"],
}


def tag_frame(text: str) -> str:
    """텍스트에서 프레임 태깅"""
    for frame, keywords in FRAME_RULES.items():
        if any(kw in text for kw in keywords):
            return frame
    return "기타"


def run(bundle_id: str, root: Path = Path("data")) -> List[Dict[str, Any]]:
    """
    entities.jsonl → events.jsonl 변환
    """
    fin = root / "bundles" / bundle_id / "entities.jsonl"
    fout = root / "bundles" / bundle_id / "events.jsonl"

    events = []
    with open(fin, "r", encoding="utf-8") as f:
        for line in f:
            art = json.loads(line)
            text = art.get("text", "")

            # actors: entities에서 통합 (persons + orgs + gpes)
            entities = art.get("entities", {})
            actors = entities.get("persons", []) + entities.get("orgs", []) + entities.get("gpes", [])

            event = {
                "id": art.get("id"),
                "time": art.get("time"),
                "text": text,
                "actors": actors,
                "frame": tag_frame(text),
            }
            events.append(event)

    with open(fout, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    return events


if __name__ == "__main__":
    # 예시 실행
    bundle = "b01"
    data = run(bundle)
    from pprint import pprint
    pprint(data[:3])
