# -*- coding: utf-8 -*-
"""
network_builder.py
인물 네트워크 생성
"""

import networkx as nx
from typing import List, Dict
from itertools import combinations

def run(events: List[Dict]) -> Dict:
    G = nx.Graph()
    for evt in events:
        actors = evt.get("actors", [])
        for a in actors:
            G.add_node(a)
        for a, b in combinations(actors, 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

    density = nx.density(G)
    centrality = nx.degree_centrality(G) if G.nodes else {}
    central_actor = max(centrality, key=centrality.get) if centrality else None

    return {
        "n_nodes": len(G.nodes),
        "n_edges": len(G.edges),
        "density": round(density, 3),
        "central_actor": central_actor,
    }
