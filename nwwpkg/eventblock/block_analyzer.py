# -*- coding: utf-8 -*-
"""
block_analyzer.py
블록 단위 인물·네트워크·프레임·위험 분석 및 리포트 생성
- 입력: data/bundles/{bundle}/blocks.jsonl
- 출력: 분석 리포트 dict
"""

from typing import Dict, Any, List
import json
from pathlib import Path
import networkx as nx
from collections import Counter
import numpy as np


def analyze_actors(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """블록 내부 인물 빈도 분석"""
    counter = Counter()
    for ev in events:
        for a in ev.get("actors", []):
            counter[a] += 1

    return [{"actor": a, "mentions": c} for a, c in counter.most_common()]


def analyze_network(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """블록 내부 인물 네트워크 분석"""
    G = nx.Graph()
    for ev in events:
        actors = ev.get("actors", [])
        for i in range(len(actors)):
            for j in range(i + 1, len(actors)):
                G.add_edge(actors[i], actors[j])

    if len(G) == 0:
        return {"nodes": [], "edges": [], "centrality": {}}

    centrality = nx.degree_centrality(G)
    nodes = list(G.nodes())
    edges = [{"source": u, "target": v} for u, v in G.edges()]

    return {
        "nodes": nodes,
        "edges": edges,
        "centrality": centrality,
    }


def analyze_frames(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """프레임 변화 분석 (시간순 프레임 추적)"""
    frames_by_time = []
    for ev in sorted(events, key=lambda e: e.get("time", "")):
        frames_by_time.append({"time": ev["time"], "frame": ev.get("frame", "기타")})

    transitions = []
    for i in range(1, len(frames_by_time)):
        if frames_by_time[i - 1]["frame"] != frames_by_time[i]["frame"]:
            transitions.append(
                f"{frames_by_time[i - 1]['frame']} → {frames_by_time[i]['frame']}"
            )

    return {
        "frames_by_time": frames_by_time,
        "transitions": transitions,
    }


def score_risk(actors: List[Dict[str, Any]], frames: Dict[str, Any]) -> float:
    """위험 점수 계산 (단순 버전: 인물 수 + 프레임 전환 가중치)"""
    num_actors = len(actors)
    num_transitions = len(frames.get("transitions", []))
    score = np.tanh(0.3 * num_actors + 0.7 * num_transitions)  # 0~1 정규화
    return round(float(score), 2)


def run(block: Dict[str, Any]) -> Dict[str, Any]:
    """
    단일 블록 분석 실행
    :param block: {"block_id": str, "events": [...]}
    :return: 분석 리포트 dict
    """
    block_id = block.get("block_id", "unknown")
    events = block.get("events", [])

    actor_report = analyze_actors(events)
    network_report = analyze_network(events)
    frame_report = analyze_frames(events)
    risk_score = score_risk(actor_report, frame_report)

    report = {
        "block_id": block_id,
        "num_events": len(events),
        "actors": actor_report,
        "network": network_report,
        "frame_report": frame_report,
        "risk_score": risk_score,
        "summary": f"블록 {block_id}: 인물 {len(actor_report)}명, "
                   f"프레임 전환 {len(frame_report['transitions'])}회, "
                   f"위험도 {risk_score:.2f}"
    }
    return report


def run_all(bundle_id: str, root: Path = Path("data")) -> List[Dict[str, Any]]:
    """
    blocks.jsonl → 모든 블록 분석
    """
    fin = root / "bundles" / bundle_id / "blocks.jsonl"
    reports = []
    with open(fin, "r", encoding="utf-8") as f:
        for line in f:
            block = json.loads(line)
            reports.append(run(block))
    return reports


if __name__ == "__main__":
    # 예시 실행
    bundle = "b01"
    results = run_all(bundle)
    from pprint import pprint
    pprint(results[:2])
