# -*- coding: utf-8 -*-
import json, math, itertools, collections
from pathlib import Path
from .tokenize import tokenize_ko

def build_cooc(fin: str, fout: str, min_count=3, window=10, top_nodes=200):
    docs = []
    for line in Path(fin).open(encoding="utf-8"):
        r = json.loads(line)
        toks = tokenize_ko(r.get("clean_text") or r.get("text") or "")
        if toks: docs.append(toks)

    tf = collections.Counter(w for d in docs for w in d)
    # 노드 필터
    nodes = {w:c for w,c in tf.items() if c >= min_count}
    vocab = {w:i for i,(w,_) in enumerate(sorted(nodes.items(), key=lambda x:-x[1])[:top_nodes])}

    # 동시출현(슬라이딩 윈도우)
    co = collections.Counter()
    for d in docs:
        idx = [vocab[w] for w in d if w in vocab]
        for i in range(len(idx)):
            wi = idx[i]
            for j in range(i+1, min(i+1+window, len(idx))):
                wj = idx[j]
                if wi==wj: continue
                a,b = (wi,wj) if wi<wj else (wj,wi)
                co[(a,b)] += 1

    N = sum(nodes.values())
    edges = []
    for (i,j), cij in co.items():
        wi = list(vocab.keys())[list(vocab.values()).index(i)]
        wj = list(vocab.keys())[list(vocab.values()).index(j)]
        pi, pj = nodes[wi]/N, nodes[wj]/N
        pij = cij/N
        # NPMI
        pmi = math.log( (pij + 1e-12) / (pi*pj + 1e-12) )
        npmi = pmi / (-math.log(pij + 1e-12))
        if cij >= min_count and npmi > 0:
            edges.append({"source": wi, "target": wj, "weight": round(npmi,4), "co": int(cij)})
    out = {
        "nodes": {w:int(nodes[w]) for w in vocab.keys()},
        "edges": edges
    }
    Path(fout).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
