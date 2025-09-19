# nwwpkg/prep/qc_report.py
import json, sys, statistics as st

def run(fin:str, fout:str):
    L = []
    with open(fin, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            L.append((r.get("id"), r.get("num_chars",0), r.get("num_sents",0)))
    n = len(L)
    chars = [x[1] for x in L]; sents = [x[2] for x in L]
    report = {
        "docs": n,
        "chars_median": int(st.median(chars)) if n else 0,
        "sents_median": int(st.median(sents)) if n else 0,
        "short_docs_pct": round(100*sum(c<200 for c in chars)/n, 1) if n else 0.0
    }
    with open(fout, "w", encoding="utf-8") as w:
        w.write(json.dumps(report, ensure_ascii=False, indent=2))

if __name__=="__main__":
    fin  = sys.argv[sys.argv.index("--in")+1]
    fout = sys.argv[sys.argv.index("--out")+1]
    run(fin, fout)
