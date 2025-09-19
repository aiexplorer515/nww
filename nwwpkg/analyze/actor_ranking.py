# nwwpkg/analysis/actor_ranking.py
"""
Actor Ranking Module
--------------------
목적: 사건(Event)과 인물(Actor)의 연결망에서 중심성을 계산하여
중요 인물/사건을 자동 추출한다.
"""

import json
import networkx as nx
from pathlib import Path

class ActorRanking:
    def __init__(self, input_file: Path):
        self.input_file = input_file
        self.graph = nx.Graph()
        self.results = {}

    def load_events(self):
        events = []
        with open(self.input_file, encoding="utf-8") as f:
            for line in f:
                events.append(json.loads(line))
        return events

    def build_graph(self, events):
        for e in events:
            eid = e["event_id"]
            for actor in e["actors"]:
                self.graph.add_edge(eid, actor)

    def compute_centrality(self):
        deg_cent = nx.degree_centrality(self.graph)
        btw_cent = nx.betweenness_centrality(self.graph)
        eig_cent = nx.eigenvector_centrality(self.graph)

        results = []
        for node in self.graph.nodes():
            results.append({
                "node": node,
                "type": "Actor" if not node.startswith("E") else "Event",
                "degree": deg_cent[node],
                "betweenness": btw_cent[node],
                "eigenvector": eig_cent[node]
            })

        self.results = sorted(results, key=lambda x: x["degree"], reverse=True)

    def save_results(self, output_file: Path):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

    def run(self, output_file: Path):
        events = self.load_events()
        self.build_graph(events)
        self.compute_centrality()
        self.save_results(output_file)

# 실행 예시
if __name__ == "__main__":
    actor_ranker = ActorRanking(Path("data/events.jsonl"))
    actor_ranker.run(Path("data/actor_ranking.json"))
