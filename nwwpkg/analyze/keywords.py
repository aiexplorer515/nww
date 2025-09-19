# -*- coding: utf-8 -*-
import json, collections
from pathlib import Path
from .tokenize import tokenize_ko
from soynlp.word import WordExtractor  # 간단한 결합도 기반 phrase

def build_keywords(fin: str, fout_jsonl: str, topn=50):
    toks_all = []
    rows = []
    for line in Path(fin).open(encoding="utf-8"):
        r = json.loads(line)
        t = r.get("clean_text") or r.get("text") or ""
        toks = tokenize_ko(t); toks_all.append(toks)
        rows.append({"id": r.get("id"), "tokens": toks})
    # 빈도
    cnt = collections.Counter(w for toks in toks_all for w in toks)
    top = cnt.most_common(topn)
    Path(fout_jsonl).write_text(
        "\n".join(json.dumps({"word":w, "freq":c}, ensure_ascii=False) for w,c in top),
        encoding="utf-8"
    )
    return rows, cnt

def mine_phrases(fin: str, topk=50):
    # soynlp 결합도 기반 bi/tri-gram 후보
    corpus = [ (json.loads(l).get("clean_text") or "") for l in Path(fin).open(encoding="utf-8") ]
    we = WordExtractor(min_frequency=2)
    we.train(corpus)
    word_scores = we.extract()
    # cohesion_score 상위
    cands = sorted(word_scores.items(), key=lambda x: -x[1].cohesion_forward)[:topk]
    return [{"phrase": w, "cohesion": float(s.cohesion_forward)} for w, s in cands]
