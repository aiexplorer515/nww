import json
import os

def normalize(articles, out_dir="data/bundles/sample"):
    """Normalize text (lowercase, deduplication, etc.)"""
    norm_articles = []
    seen = set()
    for art in articles:
        text = art["text"].lower().strip()
        h = hash(text)
        if h not in seen:
            seen.add(h)
            art["text"] = text
            norm_articles.append(art)
    out_path = os.path.join(out_dir, "articles.norm.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for a in norm_articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    return norm_articles

def extract_features(articles, out_dir="data/bundles/sample"):
    """Dummy feature extraction: keywords, summary, entities"""
    enriched = []
    for art in articles:
        enriched.append({
            "id": art["id"],
            "kw": art["text"].split()[:5],
            "summary": art["text"][:100],
            "actors": [],
            "frames": []
        })
    out_path = os.path.join(out_dir, "kyw_sum.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for a in enriched:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    return enriched
