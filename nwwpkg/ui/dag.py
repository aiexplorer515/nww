# nwwpkg/ui/dag.py
import streamlit as st
from graphviz import Digraph
from pathlib import Path
import json

NODES = [
  ("raw","Raw"),("clean","Clean"),("dedup","Dedup"),
  ("kyw","Keyword"),("kboost","LexBoost"),("frame","Frame"),
  ("score","Score"),("alert","Alert")
]
EDGES = [("raw","clean"),("clean","dedup"),("dedup","kyw"),("kyw","kboost"),("kboost","frame"),("frame","score"),("score","alert")]

def _meta(bundle:str, key:str):
    p = Path(f"data/bundles/{bundle}")
    fn = {
        "raw":"raw.jsonl","clean":"clean.jsonl","dedup":"clean.dedup.jsonl",
        "kyw":"kyw.jsonl","kboost":"kyw_boosted.jsonl","frame":"frames.jsonl",
        "score":"scores.jsonl","alert":"alerts.jsonl",
    }[key]
    f = p/fn
    if not f.exists(): return "❌"
    try:
        n = sum(1 for _ in f.open(encoding="utf-8"))
    except: n = "?"
    ts = f.stat().st_mtime
    return f"✅ n={n}"

def dag_view(bundle:str):
    g = Digraph(engine="dot")
    for k,label in NODES:
        g.node(k, f"{label}\n{_meta(bundle,k)}", shape="box", style="rounded")
    for a,b in EDGES:
        g.edge(a,b)
    st.graphviz_chart(g, use_container_width=True)
