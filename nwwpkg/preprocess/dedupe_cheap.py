import json, re, sys
from pathlib import Path

def norm_title(t:str)->str:
    t = re.sub(r"\s+", " ", (t or "").lower()).strip()
    t = re.sub(r"[^a-z0-9가-힣 ]","", t)
    return t

def lead_sig(text:str, k=40)->str:
    toks = re.findall(r"[A-Za-z0-9]+|[가-힣]+", (text or "").lower())
    return " ".join(toks[:k])

def run(fin:str, fout:str):
    seen=set(); kept=0
    with open(fin, encoding="utf-8") as f, open(fout,"w",encoding="utf-8") as w:
        for line in f:
            r=json.loads(line)
            sig = norm_title(r.get("title","")) + "|" + lead_sig(r.get("clean_text",""), 40)
            if sig in seen: 
                continue
            seen.add(sig); w.write(json.dumps(r, ensure_ascii=False)+"\n"); kept+=1
    print(f"kept={kept}, dup={len(seen)-kept}")

if __name__=="__main__":
    fin  = sys.argv[sys.argv.index("--in")+1]
    fout = sys.argv[sys.argv.index("--out")+1]
    run(fin, fout)
