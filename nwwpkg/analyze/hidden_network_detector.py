# -*- coding: utf-8 -*-
"""
숨은 네트워크(간이): 문장 내 공출현(co-occurrence) 기반 인물/조직 연결
- 고급: spaCy/KoNLPy/NER 모델 대체 가능
"""
from __future__ import annotations
import re
from collections import defaultdict
# nwwpkg/analyze/hidden_network_detector.py

# 기본 불용어(필요시 확장)
_STOP = {
    "기사원문","입력","오후","사진","연합뉴스","YTN","newsis","kmn","서비스","보내기","변경하기","사용하기",
    "관련","대한","본문","글자","수정","변환","그러나","그리고","대한민국","정부","경제","외교","보도"
}

# 간단 엔티티 후보(예: 한글 2~6자, 영문 고유명사)
_RE_KO = re.compile(r"[가-힣]{2,6}")
_RE_EN = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b")

_KO_CAND = re.compile(r"[가-힣]{2,4}")
_EN_PROPER = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b")


def _extract_entities(text: str) -> list[str]:
    tokens = []
    if not isinstance(text, str):
        return tokens
    tokens += _RE_KO.findall(text)
    tokens += _RE_EN.findall(text)
    # 숫자 포함, 길이 1, 공백 제거
    tokens = [t.strip() for t in tokens if t and not any(ch.isdigit() for ch in t)]
    return tokens

def detect_from_sentences(sentences: list[str], min_count: int = 2, stopwords: set[str] | None = None) -> dict:
    stopwords = stopwords or _STOP
    nodes, edges = defaultdict(int), defaultdict(int)
    for s in sentences or []:
        ents = [e for e in set(_extract_entities(s)) if e not in stopwords]
        for e in ents: nodes[e] += 1
        ent_list = sorted(ents)
        for i in range(len(ent_list)):
            for j in range(i+1, len(ent_list)):
                edges[(ent_list[i], ent_list[j])] += 1
    nodes = {k: v for k, v in nodes.items() if v >= min_count}
    edges = {f"{a}—{b}": c for (a, b), c in edges.items()
             if c >= min_count and a in nodes and b in nodes}
    return {"nodes": nodes, "edges": edges}

def detect(graph_or_sentences, **kwargs) -> dict:
    if isinstance(graph_or_sentences, list):
        return detect_from_sentences(graph_or_sentences, **kwargs)
    if isinstance(graph_or_sentences, dict) and "sentences" in graph_or_sentences:
        return detect_from_sentences(graph_or_sentences["sentences"], **kwargs)
    return {"nodes": {}, "edges": {}}


def _extract_entities(sent: str) -> set[str]:
    ko = set(x for x in _KO_CAND.findall(sent) if x not in _STOP)
    en = set(x for x in _EN_PROPER.findall(sent) if x not in _STOP)
    return ko | en

def detect_from_sentences(sentences: list[str], min_count: int = 2) -> dict:
    nodes = defaultdict(int)
    edges = defaultdict(int)
    for s in sentences:
        ents = sorted(list(_extract_entities(s)))
        for e in ents:
            nodes[e] += 1
        # 동시 등장 쌍 카운트
        for i in range(len(ents)):
            for j in range(i+1, len(ents)):
                pair = (ents[i], ents[j])
                edges[pair] += 1
    # 필터링
    nodes = {k:v for k,v in nodes.items() if v >= min_count}
    edges = {f"{a}—{b}":c for (a,b),c in edges.items() if c >= min_count and a in nodes and b in nodes}
    return {"nodes": nodes, "edges": edges}

# 하위호환 API
def detect(graph_or_sentences) -> dict:
    if isinstance(graph_or_sentences, dict) and "sentences" in graph_or_sentences:
        return detect_from_sentences(graph_or_sentences["sentences"])
    if isinstance(graph_or_sentences, list):  # sentences
        return detect_from_sentences(graph_or_sentences)
    return {"nodes": {}, "edges": {}}
