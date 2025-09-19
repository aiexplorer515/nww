# -*- coding: utf-8 -*-
"""
draw_network_pyvis.py
PyVis 기반 네트워크 시각화 (frame_shift 반영 버전)
"""

from pyvis.network import Network
import math

def draw_network_pyvis(block: dict, report: dict) -> str:
    """
    PyVis 네트워크 시각화 HTML 반환
    :param block: {"events": [...]}
    :param report: block_analyzer.run() 결과 (network + frame_shift 포함)
    :return: HTML 문자열
    """
    network_data = report.get("network", {})
    frame_data = report.get("frame_shift", {})
    nodes = network_data.get("nodes", [])
    edges = network_data.get("edges", [])
    centrality = network_data.get("centrality", {})

    net = Network(height="500px", width="100%", notebook=False, bgcolor="#ffffff", font_color="black")
    net.force_atlas_2based()

    # 중심성 값 정규화
    max_centrality = max(centrality.values(), default=0)
    min_centrality = min(centrality.values(), default=0)
    diff = max_centrality - min_centrality if max_centrality != min_centrality else 1

    # 1️⃣ 노드 추가 (중심성 기반 크기/색상)
    for node in nodes:
        c_val = centrality.get(node, 0)
        norm_val = (c_val - min_centrality) / diff  # 0~1로 스케일링
        size = 15 + norm_val * 25
        color_intensity = int(255 - norm_val * 200)
        color = f"rgb(255,{color_intensity},{color_intensity})"

        net.add_node(
            node,
            label=node,
            title=f"{node}<br>중심성: {c_val:.2f}",
            size=size,
            color=color
        )

    # 2️⃣ 엣지 추가 (프레임 기반 스타일 반영)
    for edge in edges:
        source, target = edge["source"], edge["target"]

        # 기본값
        color = "gray"
        dashes = True
        width = 1

        # 블록 이벤트에서 프레임을 확인
        related_events = [
            ev for ev in block.get("events", [])
            if source in ev.get("actors", []) and target in ev.get("actors", [])
        ]
        if related_events:
            frame = related_events[-1].get("frame", "unknown")  # 최신 사건 기준
            if frame == "긴장":
                color, dashes, width = "gray", True, 1
            elif frame == "협상":
                color, dashes, width = "blue", False, 2
            elif frame == "충돌":
                color, dashes, width = "red", False, 3

        net.add_edge(source, target, color=color, width=width, dashes=dashes)

    return net.generate_html()
