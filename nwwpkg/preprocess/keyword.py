# nwwpkg/prep/keyword.py
import json, math, re, sys
from collections import Counter, defaultdict

STOP_EN = set("the a an and or of to in for on with by from is are was were be been being not".split())
STOP_KO = set("그리고 그러나 또한 또한은 또한를 및 등의 등에 에서 으로 에게 대한 또는 또".split())

def tokenize_basic(txt: str) -> list[str]:
    # 한/영 혼용 안전 토큰화 (기호 최소 보존)
    toks = re.findall(r"[A-Za-z0-9]+|[가-힣]+|[%\/\-\&]", txt)
    return [t.lower() for t in toks]

def ngrams(tokens: list[str], nmin=1, nmax=3):
    for n in range(nmin, nmax+1):
        for i in range(len(tokens)-n+1):
            yield " ".join(tokens[i:i+n])

def is_stop_ngram(g: str) -> bool:
    parts = g.split()
    if all(p in STOP_EN for p in parts) or all(p in STOP_KO for p in parts):
        return True
    # 한 글자 한글/영문 단독 토큰 배제
    if len(g) <= 1: return True
    return False

def compute_tfidf(docs_tokens: list[list[str]], nmin=1, nmax=3):
    # 문서단위 n-gram 빈도
    doc_ngrams = []
    df = Counter()
    for toks in docs_tokens:
        c = Counter(g for g in ngrams(toks, nmin, nmax) if not is_stop_ngram(g))
        doc_ngrams.append(c)
        for g in c.keys(): df[g] += 1
    N = len(docs_tokens)
    tfidf_docs = []
    for c in doc_ngrams:
        scores = {}
        for g, tf in c.items():
            idf = math.log((N + 1) / (df[g] + 0.5))
            scores[g] = tf * idf
        tfidf_docs.append(scores)
    return tfidf_docs

def yake_like_features(txt: str, cand: str) -> float:
    # 매우 경량화된 위치/길이 보너스
    pos = txt.lower().find(cand)
    pos_score = 1.0 if pos == -1 else 1.5 if pos < max(10, len(txt)*0.05) else 1.0
    len_bonus = 1.2 if len(cand.split())>=2 else 1.0
    return pos_score * len_bonus

def rank_keywords(clean_texts: list[str], topk=15):
    tokenized = [tokenize_basic(t) for t in clean_texts]
    tfidf_docs = compute_tfidf(tokenized, 1, 3)
    results = []
    for idx, txt in enumerate(clean_texts):
        base = tfidf_docs[idx]
        scored = {g: score * yake_like_features(txt, g) for g, score in base.items()}
        # 상위 N 정렬
        items = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:topk*3]
        # 중복 줄이기(서로 포함되는 n-gram 중 긴 것 우선)
        chosen, seen = [], set()
        for g, s in items:
            root = g.replace(" ", "")
            if any(root in x for x in seen): 
                continue
            seen.add(root)
            chosen.append((g, float(s)))
            if len(chosen) >= topk: break
        results.append(chosen)
    return results

def run(fin: str, fout: str, topk=15):
    rows, texts = [], []
    with open(fin, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); rows.append(r)
            texts.append(r.get("clean_text","") or "")
    ranked = rank_keywords(texts, topk=topk)
    with open(fout, "w", encoding="utf-8") as w:
        for r, kws in zip(rows, ranked):
            r["keywords"] = [{"text": k, "score": s, "src": "tfidf+yake-lite"} for k, s in kws]
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    args = sys.argv
    fin  = args[args.index("--in")+1]
    fout = args[args.index("--out")+1]
    topk = int(args[args.index("--topk")+1]) if "--topk" in args else 15
    run(fin, fout, topk)
