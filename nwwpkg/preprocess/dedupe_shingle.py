# nwwpkg/prep/dedupe_shingle.py
import json, re, sys
from collections import defaultdict

def shingles(text, k=5):
    toks = re.findall(r"[A-Za-z0-9]+|[가-힣]+", text.lower())
    return {" ".join(toks[i:i+k]) for i in range(max(0, len(toks)-k+1))}

def jaccard(a:set, b:set)->float:
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter/union if union else 0.0

def run(fin:str, fout:str, j_thr=0.85):
    rows, kept, sigs = [], [], []
    with open(fin, encoding="utf-8") as f:
        for line in f: rows.append(json.loads(line))
    for r in rows:
        s = shingles(r.get("clean_text",""))
        dup = any(jaccard(s, s2) >= j_thr for s2 in sigs)
        if not dup:
            kept.append(r); sigs.append(s)
    with open(fout, "w", encoding="utf-8") as w:
        for r in kept: w.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    fin  = sys.argv[sys.argv.index("--in")+1]
    fout = sys.argv[sys.argv.index("--out")+1]
    thr  = float(sys.argv[sys.argv.index("--thr")+1]) if "--thr" in sys.argv else 0.85
    run(fin, fout, thr)
