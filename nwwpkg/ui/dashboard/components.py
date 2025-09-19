# nwwpkg/dashboard/components.py
# NWW Dashboard Components 모듈
import streamlit as st
import pandas as pd
import json
import plotly.express as px
from pathlib import Path
import statistics as stats
from collections import Counter
from io import StringIO

# === 공용 유틸 ===
def _count_lines(path: Path) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def load_jsonl(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_json(path, lines=True, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()

# === 1. Pipeline DAG ===
def render_pipeline_dag(bundle_path: Path):
    STAGES = [
        ("ingest", "ingest.jsonl"),
        ("clean", "clean.jsonl"),
        ("dedup", ("clean.dedup.jsonl", "clean.cheap.jsonl")),
        ("keyword", "kyw.jsonl"),
        ("kboost", "kyw_boosted.jsonl"),
        ("frame", "frames.jsonl"),
        ("score", "scores.jsonl"),
        ("alert", "alerts.jsonl"),
    ]
    cols = st.columns(len(STAGES))
    for i, (name, fns) in enumerate(STAGES):
        if isinstance(fns, tuple):
            exist = [(bundle_path/x) for x in fns if (bundle_path/x).exists()]
            n = max((_count_lines(p) for p in exist), default=0)
        else:
            p = bundle_path / fns
            n = _count_lines(p)
        icon = "✅" if n > 0 else "❌"
        cols[i].metric(label=name, value=n, help=f"{icon} {name}")

# === 2. Stage QA ===
def render_stage_qa(bundle_path: Path):
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
                    r = json.loads(line)
                    t = r.get("clean_text") or r.get("body") or r.get("text") or ""
                    vals_c.append(len(str(t)))
                    vals_s.append(int(r.get("num_sents") or 1))
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

# === 3. Alerts ===
def render_alerts(df_alerts: pd.DataFrame, bundle_id: str):
    st.subheader("📡 Alerts (위험 경보)")
    if df_alerts.empty:
        st.info("⚠️ Alerts 데이터가 없습니다.")
        return
    
    # 기본 카드 (Top 5 Alerts)
    for _, r in df_alerts.head(5).iterrows():
        lvl = str(r.get("alert_level", "watch")).upper()
        sc  = float(r.get("fused_score", 0.0))
        rsn = r.get("reason") or "(no reason)"
        st.markdown(f"**[{lvl}]** {rsn} · score={sc:.2f} · {r.get('date','')}")

    # Alert Level 분포
    if "alert_level" in df_alerts.columns:
        lvl = df_alerts["alert_level"].value_counts().reset_index()
        lvl.columns = ["level", "count"]
        fig_bar = px.bar(lvl, x="level", y="count", text="count", title="🚨 경보 등급 분포")
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

# === 4. Choropleth ===
def render_choropleth(df_alerts: pd.DataFrame):
    st.subheader("🗺️ Choropleth (경보 위치 분포)")
    if df_alerts.empty or "region" not in df_alerts.columns:
        st.info("⚠️ 지역 정보가 부족합니다.")
        return
    
    agg = df_alerts["region"].value_counts().reset_index()
    agg.columns = ["region", "count"]
    fig = px.choropleth(
        agg,
        locations="region",
        locationmode="country names",
        color="count",
        color_continuous_scale="Reds",
        title="국가/지역별 경보 분포"
    )
    st.plotly_chart(fig, use_container_width=True)

# === 5. Risk Blocks ===
def render_risk_blocks(df_blocks: pd.DataFrame, bundle_id: str):
    st.subheader("🔥 Risk Blocks")
    if df_blocks.empty or "fused_score" not in df_blocks.columns:
        st.info("⚠️ Risk Blocks 데이터 없음")
        return
    top_blk = (df_blocks.groupby("block")["fused_score"]
               .mean().reset_index()
               .sort_values("fused_score", ascending=False)
               .head(10))
    fig = px.bar(top_blk, x="block", y="fused_score", text="fused_score",
                 title="블록별 평균 위험도 (Top 10)")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# === 6. Scenarios ===
def render_scenarios(df_scen: pd.DataFrame, bundle_id: str):
    st.subheader("🧭 위험 시나리오")
    if df_scen.empty or "fused_score" not in df_scen.columns:
        st.info("⚠️ Scenarios 데이터 없음")
        return
    tmp = df_scen.copy()
    scen_col = "scenario_predicted" if "scenario_predicted" in tmp.columns else "scenario_matched"
    if scen_col not in tmp.columns:
        st.info("⚠️ 시나리오 컬럼이 없음")
        return
    tmp["scenario_name"] = tmp[scen_col].astype(str)
    top_scen = (tmp.groupby("scenario_name")["fused_score"]
                .mean().reset_index()
                .sort_values("fused_score", ascending=False)
                .head(20))
    st.dataframe(top_scen.rename(columns={"scenario_name":"시나리오", "fused_score":"평균 위험도"}),
                 use_container_width=True)

# === 7. Entities ===
def render_entities(df_ana: pd.DataFrame):
    st.subheader("🏷️ 주요 인물/조직")
    if df_ana.empty or "network" not in df_ana.columns:
        st.info("⚠️ Analyze 데이터에 network 없음")
        return
    counter = Counter()
    for g in df_ana["network"].dropna():
        try:
            nodes = (g or {}).get("nodes", {})
            counter.update({k: int(v) for k, v in nodes.items()})
        except Exception:
            continue
    ent_df = pd.DataFrame(counter.most_common(20), columns=["entity","count"])
    fig = px.bar(ent_df, x="entity", y="count", text="count",
                 title="Top-20 Entities")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# === 8. Sources ===
def render_sources(df_ingest: pd.DataFrame):
    st.subheader("📰 언론사/출처 분포")
    if df_ingest.empty or "source" not in df_ingest.columns:
        st.info("⚠️ Ingest 데이터에 source 없음")
        return
    src = df_ingest["source"].fillna("Manual").value_counts().reset_index()
    src.columns = ["source", "count"]
    fig = px.bar(src.head(30), x="source", y="count", text="count",
                 title="언론사/도메인 분포 (Top 30)")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
