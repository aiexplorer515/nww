# -*- coding: utf-8 -*-
"""
network_builder.py
인물 네트워크 생성 및 지표 계산
"""

from typing import List, Dict
import networkx as nx

def run(events: List[Dict]) -> Dict:
    """
    인물 네트워크 그래프 생성
    :param events: [{ "actors": [str, ...]}, ...]
    :return: {"nodes": [...], "edges": [...], "centrality": {...}}
    """
    G = nx.Graph()

    for ev in events:
        actors = ev.get("actors", [])
        for i in range(len(actors)):
            for j in range(i + 1, len(actors)):
                G.add_edge(actors[i], actors[j])

    centrality = nx.degree_centrality(G)

    return {
        "nodes": list(G.nodes),
        "edges": [{"source": u, "target": v} for u, v in G.edges],
        "centrality": centrality
    }
