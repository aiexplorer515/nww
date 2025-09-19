import json
import os

def score(articles, out_dir="data/bundles/sample"):
    """Indicator-based scoring (dummy weights)"""
    results = []
    for art in articles:
        score_val = min(len(art.get("kw", [])) * 0.1, 1.0)
        results.append({
            "id": art["id"],
            "stage": "IS",
            "score": score_val,
            "detail": {"kw_count": len(art.get("kw", []))}
        })
    out_path = os.path.join(out_dir, "scores.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return results
