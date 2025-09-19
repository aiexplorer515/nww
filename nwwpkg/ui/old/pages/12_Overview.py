import streamlit as st
import pandas as pd
import plotly.express as px
from nwwpkg.utils.loader import load_jsonl
from nwwpkg.utils.adapters import normalize_dataframe, safe_float

st.title("🌍 Crisis Overview")

bundle_id = st.session_state.get("bundle_id", "sample")
path = f"data/bundles/{bundle_id}/overview.jsonl"

records = load_jsonl(path)
if not records:
    st.warning("⚠️ 개요 데이터 없음")
else:
    df = pd.DataFrame(records)
    df = normalize_dataframe(df, {
        "date": ["date", "time"],
        "alerts": ["alerts", "count"],
        "avg_score": ["avg_score", "risk"]
    })
    df["avg_score"] = df["avg_score"].apply(lambda x: safe_float(x, 0))

    st.metric("기사 수", len(df))
    st.metric("평균 위험도", f"{df['avg_score'].mean():.2f}")

    fig = px.line(df, x="date", y="alerts", title="최근 알림 추세")
    st.plotly_chart(fig, use_container_width=True)
