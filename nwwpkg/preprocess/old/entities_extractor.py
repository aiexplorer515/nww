# -*- coding: utf-8 -*-
"""
entities_extractor.py
기사 본문에서 인물(Person), 국가(GPE), 조직(ORG) 엔티티 추출
- 입력: data/bundles/{bundle}/clean.jsonl
- 출력: data/bundles/{bundle}/entities.jsonl
"""

from typing import Dict, Any, List
import json
from pathlib import Path
import re
import spacy
import pandas as pd


# 1️⃣ spaCy NER 모델 로드 (영어 중심, 필요 시 한국어 모델 교체 가능)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    텍스트에서 인물·국가·조직 엔티티 추출
    :param text: 기사 본문
    :return: {"persons": [...], "orgs": [...], "gpes": [...]}
    """
    persons, orgs, gpes = set(), set(), set()

    # 1️⃣ spaCy 기반 추출
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                persons.add(ent.text)
            elif ent.label_ == "ORG":
                orgs.add(ent.text)
            elif ent.label_ == "GPE":
                gpes.add(ent.text)

    # 2️⃣ 한글 패턴 기반 fallback
    patterns = {
        "persons": [r"[가-힣]{2,3}"],  # 한국어 이름 후보 (예: 김정은, 윤석열)
        "gpes": [r"(북한|한국|대한민국|미국|중국|일본|러시아|유럽연합)"],
        "orgs": [r"(국방부|외교부|청와대|백악관|UN|NATO|EU)"],
    }
    for label, pats in patterns.items():
        for p in pats:
            for m in re.findall(p, text):
                if label == "persons":
                    persons.add(m)
                elif label == "gpes":
                    gpes.add(m)
                elif label == "orgs":
                    orgs.add(m)

    return {
        "persons": list(persons),
        "orgs": list(orgs),
        "gpes": list(gpes),
    }

def run_text(text: str):
    """단일 텍스트에서 엔티티 추출"""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_})
    return entities


def run_file(fin: str, fout: str) -> pd.DataFrame:
    rows = []
    with open(fin, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line.strip())
                if not isinstance(row, dict):
                    continue
            except json.JSONDecodeError:
                continue

            text = row.get("normalized") or row.get("text") or ""
            ents = run(text)  # 기존 run() 함수로 NER 수행
            row["entities"] = ents
            rows.append(row)

    # 저장
    with open(fout, "w", encoding="utf-8") as fw:
        for r in rows:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")

    return pd.DataFrame(rows)


def run(bundle_id: str, root: Path = Path("data")) -> List[Dict[str, Any]]:
    """
    clean.jsonl → entities.jsonl 변환
    """
    fin = root / "bundles" / bundle_id / "clean.jsonl"
    fout = root / "bundles" / bundle_id / "entities.jsonl"

    results = []
    with open(fin, "r", encoding="utf-8") as f:
        for line in f:
            art = json.loads(line)
            text = art.get("text", "")
            ents = extract_entities(text)
            enriched = {
                "id": art.get("id"),
                "time": art.get("time"),
                "text": text,
                "entities": ents,
            }
            results.append(enriched)

    with open(fout, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results


if __name__ == "__main__":
    # 예시 실행
    bundle = "b01"
    data = run(bundle)
    from pprint import pprint
    pprint(data[:3])
