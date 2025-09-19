import json, re, collections
from pathlib import Path
root=Path("data/b01"); src=root/"clean.jsonl"; outp=root/"network.json"
try:
    import kiwipiepy as kp
    kiwi=kp.Kiwi(num_workers=1); KEEP={"NNG","NNP","SL","SN"}
    STOP=set("기사|원문|입력|오전|오후|사진|연합뉴스|뉴스1|YTN|KBS|SBS|MBC|JTBC|newsis|서비스|보내기|관련|본문|글자|수정|배포|금지|재배포|저작권|기자|특파원|통신원|AFP|AP|로이터".split("|"))
    def toks(s):
        s=re.sub(r"\s+"," ", s or "").strip()
        return [w.form.lower() for w in kiwi.tokenize(s) if w.tag in KEEP and len(w.form)>=2 and w.form not in STOP]
except Exception:
    def toks(s):
        s=re.sub(r"[^가-힣a-zA-Z0-9 ]+"," ", s or "").lower()
        return [t for t in s.split() if len(t)>=2]

docs=[]
for line in (src.open(encoding="utf-8")):
    r=json.loads(line); docs.append(toks(r.get("clean_text") or r.get("text") or ""))

tf=collections.Counter(w for d in docs for w in d)
vocab=[w for w,c in tf.items() if c>=1][:300]
idx={w:i for i,w in enumerate(vocab)}
co=collections.Counter(); W=12
for d in docs:
    ids=[idx[w] for w in d if w in idx]
    for i in range(len(ids)):
        for j in range(i+1, min(i+1+W, len(ids))):
            a,b=sorted((ids[i],ids[j])); 
            if a!=b: co[(a,b)]+=1
edges=[{"source":vocab[i],"target":vocab[j],"weight":float(c),"co":int(c)} for (i,j),c in co.items() if c>=1]
(outp).write_text(json.dumps({"nodes": {w:int(tf[w]) for w in vocab}, "edges": edges}, ensure_ascii=False, indent=2), encoding="utf-8")
print("network ->", outp, "nodes:", len(vocab), "edges:", len(edges))
