# -*- coding: utf-8 -*-
"""
actor_extractor.py
사건 리스트에서 인물 정보 추출
"""

from typing import List, Dict
from collections import Counter

def run(events: List[Dict]) -> List[Dict]:
    """
    인물 추출 및 등장 빈도 집계
    :param events: [{ "id": str, "text": str, "actors": [str, ...]}, ...]
    :return: [{"actor": str, "mentions": int}]
    """
    actors = []
    for ev in events:
        actors.extend(ev.get("actors", []))

    counter = Counter(actors)
    return [{"actor": a, "mentions": c} for a, c in counter.items()]
