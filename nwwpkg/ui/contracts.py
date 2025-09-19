import json
from pathlib import Path
import streamlit as st
import pandas as pd

SCHEMAS = {
  "frames.jsonl": ["id", "frame.label", "frame.conf", "frame.scores"],
  "alerts.jsonl": ["id","level","fused_score","ci","reason","evidence","links","t_on"]
}

def _peek(path: Path, n=3):
    if not path.exists(): return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i>=n: break
            try: rows.append(json.loads(line))
            except: pass
    return rows

def contract_view(bundle: str):
    st.subheader("📜 Data Contracts")
    base = Path(f"data/bundles/{bundle}")
    for fname, fields in SCHEMAS.items():
        f = base/fname
        st.markdown(f"**{fname}** — required fields: `{', '.join(fields)}`")
        rows = _peek(f)
        if rows:
            df = pd.json_normalize(rows)
            st.dataframe(df[ [c for c in df.columns if any(c.startswith(k) for k in fields)] ],
                         use_container_width=True, hide_index=True)
        else:
            st.caption("샘플 없음")
        st.divider()
