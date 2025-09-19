# tools/frame_map_auto.py
import json, sys, csv, difflib, pathlib
from collections import Counter
# tools/frame_map_auto.py (추가)
from pathlib import Path

def resolve_vocab_path(p: str) -> str:
    cand = [
        p,
        "config/frames.master.json",
        "configs/frames.master.json",
        "frames.master.json",
    ]
    for c in cand:
        if Path(c).exists():
            return c
    raise FileNotFoundError(f"frames.master.json not found. tried: {cand}")

def load_vocab(p):
    p = resolve_vocab_path(p)   # ← 추가
    return [x.strip() for x in json.load(open(p, encoding="utf-8"))]

def iter_frames(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            j=json.loads(line)
            fs = j.get("frames") or j.get("frame") or []
            for fr in fs:
                if fr: 
                    yield str(fr).strip()

def main(fin, vocab_json, fout_csv, cutoff=0.6):
    vocab = load_vocab(vocab_json)
    seen = Counter(iter_frames(fin))
    rows = [("from","to","count","match_score")]
    for src, cnt in seen.items():
        tgt = difflib.get_close_matches(src, vocab, n=1, cutoff=cutoff)
        if tgt:
            score = difflib.SequenceMatcher(None, src, tgt[0]).ratio()
            rows.append((src, tgt[0], cnt, f"{score:.2f}"))
        else:
            rows.append((src, "other", cnt, "0.00"))
    with open(fout_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

if __name__ == "__main__":
    fin, vocab, fout = sys.argv[1], sys.argv[2], sys.argv[3]
    pathlib.Path(fout).parent.mkdir(parents=True, exist_ok=True)
    main(fin, vocab, fout)
