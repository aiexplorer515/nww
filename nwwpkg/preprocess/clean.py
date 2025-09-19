# nwwpkg/prep/clean.py
import json, re, unicodedata, sys
from pathlib import Path

_ABBR = r"(?:Mr|Dr|No|U\.S|U\.N|Rev|etc)\."
# Fixed: Use a simpler approach without variable-width look-behind
_SENT_SPLIT = re.compile(r"(?<=[\.\?\!。？！…])\s+")

BPAT = [
    r"All rights reserved.*$", r"Subscribe to .*newsletter.*$",
    r"무단전재.*금지.*$", r"저작권자.*무단.*금지.*$", r"Copyright \d{4}.*$"
]

def normalize(t:str)->str:
    t = unicodedata.normalize("NFKC", t or "")
    t = t.replace("\u00A0"," ").replace("\t"," ")
    t = re.sub(r"[‘’´`]", "'", t)
    t = re.sub(r"[“”]", '"', t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def drop_boilerplate(t:str)->str:
    for p in BPAT: t = re.sub(p, " ", t, flags=re.I|re.M)
    return t
    

def sent_tokenize(t:str)->list[str]:
    # First, protect abbreviations by temporarily replacing them
    abbr_map = {}
    abbr_count = 0
    for match in re.finditer(_ABBR, t):
        placeholder = f"__ABBR_{abbr_count}__"
        abbr_map[placeholder] = match.group()
        t = t.replace(match.group(), placeholder)
        abbr_count += 1
    
    # Split sentences
    sentences = [s.strip() for s in _SENT_SPLIT.split(t) if s.strip()]
    
    # Restore abbreviations
    for placeholder, original in abbr_map.items():
        for i, sent in enumerate(sentences):
            sentences[i] = sent.replace(placeholder, original)
    
    return sentences

def run(fin:str, fout:str):
    out=[]
    with open(fin, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            title = normalize(r.get("title",""))
            body  = normalize(r.get("body") or r.get("text") or "")
            merged = drop_boilerplate((title + ". " + body).strip() if body else title)
            sents = sent_tokenize(merged)

            # 짧은·잡음 문장 제거(5자 미만 삭제), 중복문장 제거
            dedup, seen = [], set()
            for s in sents:
                if len(s) < 5: continue
                key = re.sub(r"\W+","", s.lower())
                if key not in seen: seen.add(key); dedup.append(s)

            r["clean_text"] = " ".join(dedup)
            r["num_sents"]  = len(dedup)
            r["num_chars"]  = len(r["clean_text"])
            out.append(r)

    with open(fout, "w", encoding="utf-8") as w:
        for r in out: w.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    fin  = sys.argv[sys.argv.index("--in")+1]
    fout = sys.argv[sys.argv.index("--out")+1]
    run(fin, fout)
