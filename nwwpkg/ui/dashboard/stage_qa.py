import streamlit as st
import json
import statistics as stats
from pathlib import Path

def render_stage_qa(bundle_path: Path):
    """
    🧪 Stage QA (품질 점검)
    """
    qc_path = bundle_path / "prep.qc.json"
    if qc_path.exists():
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
    else:
        vals_c, vals_s = [], []
        p = bundle_path / "clean.jsonl"
        if p.exists():
            with p.open(encoding="utf-8-sig") as f:
                for i, line in enumerate(f):
                    if i > 2000: break
                    try:
                        r = json.loads(line)
                        t = r.get("clean_text") or r.get("body") or r.get("text") or ""
                        vals_c.append(len(str(t)))
                        vals_s.append(int(r.get("num_sents") or 1))
                    except Exception:
                        continue
        qc = {
            "chars_median": int(stats.median(vals_c)) if vals_c else 0,
            "sents_median": int(stats.median(vals_s)) if vals_s else 0,
            "short_docs_pct": round(100*sum(1 for x in vals_c if x < 200)/len(vals_c), 1) if vals_c else 100.0
        }

    c1, c2, c3 = st.columns(3)
    c1.metric("Chars(med)", qc["chars_median"])
    c2.metric("Sents(med)", qc["sents_median"])
    c3.metric("Short Docs%", f'{qc["short_docs_pct"]}%')

    g1 = "✅" if qc["chars_median"] >= 800 else ("⚠️" if qc["chars_median"] >= 600 else "❌")
    g2 = "✅" if qc["sents_median"] >= 8 else ("⚠️" if qc["sents_median"] >= 6 else "❌")
    g3 = "✅" if qc["short_docs_pct"] <= 20 else ("⚠️" if qc["short_docs_pct"] <= 30 else "❌")
    st.caption(f"Gate: 본문 {g1} / 문장수 {g2} / 짧은문서 {g3}")
