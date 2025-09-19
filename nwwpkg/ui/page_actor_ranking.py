# nwwpkg/ui/page_analyze.py
import streamlit as st
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

def page_analyze(root: Path):
    st.header("🔎 Analyze – 관계 분석")

    edge_file = root / "data/relations.jsonl"
    if not edge_file.exists():
        st.warning("관계 데이터(relations.jsonl)가 없습니다.")
        return

    # === 데이터 로드 ===
    edges = []
    with open(edge_file, encoding="utf-8") as f:
        for line in f:
            edges.append(json.loads(line))
    df = pd.DataFrame(edges)

    # === 탭 구조 ===
    tab1, tab2 = st.tabs(["📋 Table", "🌐 Graph"])

    with tab1:
        st.subheader("관계 테이블")
        st.dataframe(df)

    with tab2:
        st.subheader("네트워크 그래프")
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_edge(row["from"], row["to"], reason=",".join(row["reasons"]))

        plt.figure(figsize=(6, 4))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True,
                node_color="skyblue", node_size=1200,
                edge_color="gray", font_size=8)
        edge_labels = nx.get_edge_attributes(G, "reason")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        st.pyplot(plt)
