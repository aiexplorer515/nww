from pathlib import Path, PurePath
import json, streamlit as st

def contracts_tile(bundle: str):
    p = Path(f"data/bundles/{bundle}/contract.report.json")
    st.subheader("📜 Contracts Compliance")
    if not p.exists():
        st.warning("contract.report.json 없음 — schema_audit를 먼저 실행")
        return
    rep = json.loads(p.read_text(encoding="utf-8"))
    fails = [r for r in rep if r["rate"]<0.99]
    st.metric("Files", len(rep))
    st.metric("Passing", len(rep)-len(fails))
    if fails:
        st.error(f"❌ Fails: {len(fails)}")
        for r in fails[:5]:
            st.caption(f"{PurePath(r['file']).name}: {r['rate']:.2%}")
    else:
        st.success("✅ All contracts ≥ 99%")
