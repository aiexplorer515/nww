# tools/relation_generator.py
"""
Relation Generator
------------------
events.jsonl → relations.jsonl 자동 변환기
1. 이벤트 쌍을 비교
2. Actor 공유, 시간·공간 근접, 도메인 차이 등을 조건으로 관계 생성
"""

import json
from pathlib import Path
from itertools import combinations
from datetime import datetime

DATA_DIR = Path("data")
EVENT_FILE = DATA_DIR / "events.jsonl"
REL_FILE = DATA_DIR / "relations.jsonl"

def days_diff(d1, d2):
    d1, d2 = datetime.fromisoformat(d1), datetime.fromisoformat(d2)
    return abs((d1 - d2).days)

def load_events(file):
    events = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            events.append(json.loads(line))
    return events

def generate_relations(events, max_days=5):
    relations = []
    for e1, e2 in combinations(events, 2):
        reasons = []

        # Actor 공유
        if set(e1["actors"]) & set(e2["actors"]):
            reasons.append("공통 Actor")

        # 시간·공간 근접
        if days_diff(e1["date"], e2["date"]) <= max_days and e1["location"] == e2["location"]:
            reasons.append("시간·공간 근접")

        # 도메인 다름 (겉보기 다름에도 연결)
        if e1["domain"] != e2["domain"] and reasons:
            reasons.append("이종 도메인 연결")

        if reasons:
            relations.append({
                "from": e1["event_id"],
                "to": e2["event_id"],
                "reasons": reasons
            })
    return relations

def save_relations(relations, file):
    with open(file, "w", encoding="utf-8") as f:
        for rel in relations:
            f.write(json.dumps(rel, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    events = load_events(EVENT_FILE)
    relations = generate_relations(events)
    save_relations(relations, REL_FILE)
    print(f"[INFO] {len(relations)} relations saved to {REL_FILE}")
