"""
NWW Streamlit Dashboard - Complete Automation Package UI (Final)
- Landing (Crisis Overview)
- Ingest (Real news URL -> full pipeline, append & accumulate)
- Scoring / Timeline / Blocks / Scenarios / Artifacts / Ledger / Overview
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------------------
# Path & Logging
# -------------------------------------------------------------------
# Ensure we can import nwwpkg.* when running as "streamlit run nwwpkg/ui/app_total.py"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NWW-UI")

# -------------------------------------------------------------------
# Import pipeline modules (new path first, then fallback to old API)
# -------------------------------------------------------------------
try:
    # New structure
    from nwwpkg.ingest.extractor import Extractor  # optional (we also support direct URL fetching here)
    from nwwpkg.preprocess.normalizer import Normalizer
    from nwwpkg.analyze.tagger import Tagger
    from nwwpkg.rules.gating import Gating

    from nwwpkg.score.score_is import ScoreIS
    from nwwpkg.score.score_dbn import ScoreDBN
    from nwwpkg.score.score_llm import ScoreLLM as LLMJudge  # name align

    from nwwpkg.fusion.calibration import FusionCalibration
    from nwwpkg.eds.block_matching import BlockMatcher as EDSBlockMatcher
    from nwwpkg.scenario.scenario_builder import ScenarioBuilder

    from nwwpkg.ops.alert_decider import AlertDecider
    from nwwpkg.ops.ledger import Ledger as AuditLedger

    from nwwpkg.eventblock.aggregator import EventBlockAggregator
    NEW_IMPORTS = True
except Exception as e_new:
    logger.warning(f"[Compat] Falling back to legacy imports: {e_new}")
    try:
        # Legacy structure via API-stable __init__.py (if you added them)
        from nwwpkg.ingest import Extractor
        from nwwpkg.preprocess import Normalizer
        from nwwpkg.analyze import Tagger
        from nwwpkg.rules import Gating
        from nwwpkg.score import ScoreIS, ScoreDBN, LLMJudge
        from nwwpkg.fusion import FusionCalibration
        from nwwpkg.eds import EDSBlockMatcher
        from nwwpkg.scenario import ScenarioBuilder
        from nwwpkg.decider import AlertDecider
        from nwwpkg.ledger import AuditLedger
        from nwwpkg.eventblock import EventBlockAggregator
        NEW_IMPORTS = False
    except Exception as e_old:
        logger.error(f"[ImportError] Cannot import pipeline modules: {e_old}")
        # We will still allow URL ingest/append path to work; other steps will be skipped.

# Optional libs for URL fetching
try:
    import requests
    from bs4 import BeautifulSoup
    try:
        from readability import Document
        HAVE_READABILITY = True
    except Exception:
        HAVE_READABILITY = False
    HAVE_REQUESTS = True
except Exception:
    HAVE_REQUESTS = False
    HAVE_READABILITY = False


# =============================================================================
# Streamlit App
# =============================================================================
def main():
    st.set_page_config(
        page_title="🌍 NWW Dashboard",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Session State
    if "bundle_dir" not in st.session_state:
        st.session_state.bundle_dir = "data/bundles/sample"
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = {}
    if "config" not in st.session_state:
        st.session_state.config = {
            "threshold": 0.7,
            "ema_alpha": 0.3,
            "hysteresis": 0.1
        }

    # Sidebar
    render_sidebar()

    st.title("🌍 NWW - News World Watch")
    st.caption("실제 뉴스 URL을 수동 입력 → 전체 파이프라인 실행 → 누적 저장 → 통계")

    # Tabs
    landing, ingest_tab, scoring_tab, timeline_tab, blocks_tab, scenarios_tab, artifacts_tab, ledger_tab, overview_tab = st.tabs([
        "🦉 랜딩", "📥 Ingest", "🎯 Scoring", "⏰ Timeline",
        "🧱 Blocks", "📋 Scenarios", "📦 Artifacts", "📝 Ledger", "📊 Overview"
    ])

    with landing:
        render_landing_tab()

    with ingest_tab:
        render_ingest_tab()

    with scoring_tab:
        render_scoring_tab()

    with timeline_tab:
        render_timeline_tab()

    with blocks_tab:
        render_blocks_tab()

    with scenarios_tab:
        render_scenarios_tab()

    with artifacts_tab:
        render_artifacts_tab()

    with ledger_tab:
        render_ledger_tab()

    with overview_tab:
        render_overview_tab()


# =============================================================================
# Sidebar
# =============================================================================
def render_sidebar():
    st.sidebar.title("⚙️ Configuration")

    bundle_dir = st.sidebar.text_input(
        "Bundle Directory",
        value=st.session_state.bundle_dir,
        help="Path to the data bundle directory"
    )
    st.session_state.bundle_dir = bundle_dir

    st.sidebar.subheader("🔧 Thresholds")
    st.session_state.config["threshold"] = st.sidebar.slider("Alert Threshold", 0.0, 1.0, st.session_state.config["threshold"], 0.05)
    st.session_state.config["ema_alpha"] = st.sidebar.slider("EMA Alpha", 0.0, 1.0, st.session_state.config["ema_alpha"], 0.05)
    st.session_state.config["hysteresis"] = st.sidebar.slider("Hysteresis", 0.0, 0.5, st.session_state.config["hysteresis"], 0.01)

    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Run All (No URL / existing files)"):
        with st.spinner("Running full pipeline on existing files..."):
            run_pipeline_chain_none_url(st.session_state.bundle_dir)
            st.success("✅ Done")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Status")
    for k, v in st.session_state.processing_status.items():
        if v == "completed":
            st.sidebar.success(f"✅ {k}")
        elif v == "running":
            st.sidebar.info(f"🔄 {k}")
        elif v == "error":
            st.sidebar.error(f"❌ {k}")
        else:
            st.sidebar.write(f"⏸️ {k}")


# =============================================================================
# Landing
# =============================================================================
def render_landing_tab():
    st.header("🦉 Crisis Overview (Aggregated)")

    alerts_df = get_jsonl_df(os.path.join(st.session_state.bundle_dir, "alerts.jsonl"))
    alerts_df = ensure_alerts_schema(alerts_df)

    if alerts_df.empty:
        st.info("⚠️ 아직 alerts.jsonl 데이터가 없습니다. Ingest 탭에서 URL을 입력해 파이프라인을 실행하세요.")
        return

    # KPI cards
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Active Alerts", int((alerts_df["state"] == "ALERT").sum()))
    with col_b:
        st.metric("Avg Score", f"{alerts_df['score'].mean():.2f}")
    with col_c:
        st.metric("Regions", alerts_df["region"].nunique())
    with col_d:
        st.metric("Domains", alerts_df["domain"].nunique())

    # Region / Domain charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌍 지역별 위기 현황")
        reg = alerts_df.groupby("region")["score"].count().reset_index(name="Active Alerts")
        if not reg.empty:
            fig1 = px.bar(reg, x="region", y="Active Alerts", color="region", title="지역별 활성 알림")
            st.plotly_chart(fig1, width="stretch")
        else:
            st.info("지역 데이터가 없습니다.")

    with col2:
        st.subheader("⚔️ 분야별 위기 현황")
        dom = alerts_df.groupby("domain")["score"].count().reset_index(name="Count")
        if not dom.empty:
            fig2 = px.pie(dom, values="Count", names="domain", title="분야별 알림 비중")
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("분야 데이터가 없습니다.")

    # Timeline (hourly)
    st.subheader("🕒 최근 24시간 알림 추이")
    alerts_df["ts"] = pd.to_datetime(alerts_df["ts"], errors="coerce")
    recent = alerts_df.dropna(subset=["ts"]).copy()
    if not recent.empty:
        recent["hour"] = recent["ts"].dt.floor("h")
        by_hour = recent.groupby("hour")["score"].count().reset_index(name="alerts")
        fig3 = px.line(by_hour, x="hour", y="alerts", title="시간대별 알림 수")
        st.plotly_chart(fig3, width="stretch")
    else:
        st.info("타임라인 데이터를 만들 수 없습니다 (ts 없음).")

    # Recent Alerts Table
    st.subheader("🗂 최근 알림 (최신 30)")
    cols = ["ts", "id", "title", "region", "domain", "score", "ci_low", "ci_high", "intl_flag", "state"]
    show = alerts_df.sort_values("ts", ascending=False).head(30)
    st.dataframe(show[cols], width="stretch")


# =============================================================================
# Ingest (URL -> pipeline)
# =============================================================================
def render_ingest_tab():
    st.header("📥 Ingest - 실제 뉴스 URL 입력")

    url = st.text_input("뉴스 기사 URL", placeholder="https://...")
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("▶️ URL 수집 & 전체 파이프라인 실행", type="primary"):
            if not url:
                st.warning("URL을 입력하세요.")
            else:
                with st.spinner("Fetching + Running pipeline..."):
                    ok = run_pipeline_chain_for_url(url, st.session_state.bundle_dir)
                    if ok:
                        st.success("✅ 처리 완료 (누적 저장됨)")
                    else:
                        st.error("❌ 처리 중 일부 단계 실패. 로그를 확인하세요.")
    with col_btn2:
        if st.button("🔁 누적 상태 새로고침"):
            st.rerun()

    st.markdown("---")
    st.subheader("📄 현재 기사 누적 현황")
    art = get_jsonl_df(os.path.join(st.session_state.bundle_dir, "articles.jsonl"))
    st.write(f"총 기사 수: **{len(art)}**")
    if not art.empty:
        cols = ["id", "ts", "title", "domain", "region", "source", "lang"]
        for c in cols:
            if c not in art.columns:
                art[c] = None
        st.dataframe(art[cols].tail(30).sort_values("ts", ascending=False), width="stretch")


# =============================================================================
# Scoring / Timeline / Blocks / Scenarios / Artifacts / Ledger / Overview
# =============================================================================
def render_scoring_tab():
    st.header("🎯 Scoring")
    scores = get_jsonl_df(os.path.join(st.session_state.bundle_dir, "scores.jsonl"))
    if scores.empty:
        st.info("scores.jsonl 없음. URL 처리 후 생성됩니다.")
        return
    if "score" not in scores.columns:
        scores["score"] = 0.0
    if "stage" not in scores.columns:
        scores["stage"] = "FUSION"

    st.subheader("📊 Score Distribution")
    fig = px.histogram(scores, x="score", nbins=30, title="Score Distribution")
    st.plotly_chart(fig, width="stretch")

    st.subheader("📈 Scores by Stage")
    try:
        agg = scores.groupby("stage")["score"].agg(["mean", "std", "count"]).reset_index()
        st.dataframe(agg, width="stretch")
    except Exception as e:
        st.write(f"그룹화 불가: {e}")


def render_timeline_tab():
    st.header("⏰ Timeline")
    alerts_df = ensure_alerts_schema(get_jsonl_df(os.path.join(st.session_state.bundle_dir, "alerts.jsonl")))
    if alerts_df.empty:
        st.info("alerts.jsonl 없음")
        return
    alerts_df["ts"] = pd.to_datetime(alerts_df["ts"], errors="coerce")
    t = alerts_df.dropna(subset=["ts"]).copy()
    t["hour"] = t["ts"].dt.floor("h")
    by_hour = t.groupby(["hour", "domain"])["score"].count().reset_index(name="alerts")
    if by_hour.empty:
        st.info("타임라인 데이터 없음")
        return
    fig = px.line(by_hour, x="hour", y="alerts", color="domain", title="도메인별 시간대 알림 추이")
    st.plotly_chart(fig, width="stretch")


def render_blocks_tab():
    st.header("🧱 EDS Blocks")
    blocks = get_jsonl_df(os.path.join(st.session_state.bundle_dir, "blocks.jsonl"))
    if blocks.empty:
        st.info("blocks.jsonl 없음 (EDS 매칭 전)")
        return
    st.dataframe(blocks.tail(50), width="stretch")


def render_scenarios_tab():
    st.header("📋 Scenarios")
    sc = get_jsonl_df(os.path.join(st.session_state.bundle_dir, "scenarios.jsonl"))
    if sc.empty:
        st.info("scenarios.jsonl 없음")
        return
    st.dataframe(sc.tail(50), width="stretch")


def render_artifacts_tab():
    st.header("📦 Artifacts & Export")
    st.write("필요 시 JSON/ZIP/STIX 내보내기 기능 연결 (향후). 현재는 파일 누적만 확인합니다.")
    bundle = st.session_state.bundle_dir
    files = [
        "articles.jsonl", "articles.norm.jsonl", "kyw_sum.jsonl", "gated.jsonl",
        "scores.jsonl", "fused_scores.jsonl", "blocks.jsonl", "scenarios.jsonl",
        "alerts.jsonl", "event_blocks.jsonl", "ledger.jsonl"
    ]
    rows = []
    for f in files:
        p = os.path.join(bundle, f)
        rows.append({
            "file": f, "exists": os.path.exists(p),
            "size(bytes)": os.path.getsize(p) if os.path.exists(p) else 0
        })
    st.dataframe(pd.DataFrame(rows), width="stretch")


def render_ledger_tab():
    st.header("📝 Audit Ledger")
    lg = get_jsonl_df(os.path.join(st.session_state.bundle_dir, "ledger.jsonl"))
    if lg.empty:
        st.info("ledger.jsonl 없음")
        return
    st.dataframe(lg.tail(200), width="stretch")


def render_overview_tab():
    st.header("📊 Overview (Pipeline Steps)")
    steps = [
        ("Ingest", "extract & parse articles"),
        ("Normalize", "clean & normalize text"),
        ("Analyze", "keywords, summary, entities"),
        ("Gate", "indicator-based gating"),
        ("Score IS/DBN/LLM", "multi-model scoring"),
        ("Fusion", "calibration & conformal"),
        ("Blocks", "EDS block matching"),
        ("Scenarios", "scenario building"),
        ("Alerts", "final alert decision"),
        ("Ledger", "audit trail"),
    ]
    for s, d in steps:
        state = st.session_state.processing_status.get(s, "pending")
        if state == "completed":
            st.success(f"✅ {s}: {d}")
        elif state == "running":
            st.info(f"🔄 {s}: {d}")
        elif state == "error":
            st.error(f"❌ {s}: {d}")
        else:
            st.write(f"⏸️ {s}: {d}")


# =============================================================================
# Pipeline Orchestration
# =============================================================================
def run_pipeline_chain_for_url(url: str, bundle_dir: str) -> bool:
    """
    1) Fetch URL -> append to articles.jsonl
    2) Normalize -> Analyze -> Gating -> Scoring(IS/DBN/LLM)
    3) Fusion -> EDS -> Scenario -> Alerts -> Event Blocks -> Ledger
    """
    ok = True
    _mark("Ingest", "running")
    try:
        append_article_from_url(url, bundle_dir)
        _mark("Ingest", "completed")
    except Exception as e:
        logger.exception(e)
        _mark("Ingest", "error")
        ok = False

    ok = run_pipeline_chain_none_url(bundle_dir) and ok
    return ok


def run_pipeline_chain_none_url(bundle_dir: str) -> bool:
    """
    Run the rest of pipeline on existing files (no fresh URL ingest here).
    """
    results = []

    # Normalize
    results.append(_safe_step("Normalize", _step_normalize, bundle_dir))

    # Analyze
    results.append(_safe_step("Analyze", _step_analyze, bundle_dir))

    # Gating
    results.append(_safe_step("Gate", _step_gating, bundle_dir))

    # Scoring
    results.append(_safe_step("Score IS", _step_score_is, bundle_dir))
    results.append(_safe_step("Score DBN", _step_score_dbn, bundle_dir))
    results.append(_safe_step("Score LLM", _step_score_llm, bundle_dir))

    # Fusion
    results.append(_safe_step("Fusion", _step_fusion, bundle_dir))

    # Blocks
    results.append(_safe_step("Blocks", _step_blocks, bundle_dir))

    # Scenarios
    results.append(_safe_step("Scenarios", _step_scenarios, bundle_dir))

    # Alerts
    results.append(_safe_step("Alerts", _step_alerts, bundle_dir))

    # Event Blocks
    results.append(_safe_step("Event Blocks", _step_event_blocks, bundle_dir))

    # Ledger
    results.append(_safe_step("Ledger", _step_ledger, bundle_dir))

    return all(results)


def _safe_step(name: str, fn, bundle_dir: str) -> bool:
    _mark(name, "running")
    try:
        fn(bundle_dir)
        _mark(name, "completed")
        return True
    except Exception as e:
        logger.exception(e)
        _mark(name, "error")
        return False


def _mark(step: str, state: str):
    st.session_state.processing_status[step] = state


# ---- Individual steps (robust to missing modules) ---------------------------
def _step_normalize(bundle_dir: str):
    src = os.path.join(bundle_dir, "articles.jsonl")
    dst = os.path.join(bundle_dir, "articles.norm.jsonl")
    log = os.path.join(bundle_dir, "logs", "normalize.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    if not os.path.exists(src):
        raise FileNotFoundError("articles.jsonl not found")
    if 'Normalizer' in globals():
        Normalizer().run(src, dst, log)
    else:
        # minimal passthrough
        _copy_jsonl(src, dst)


def _step_analyze(bundle_dir: str):
    src = os.path.join(bundle_dir, "articles.norm.jsonl")
    dst = os.path.join(bundle_dir, "kyw_sum.jsonl")
    if not os.path.exists(src):
        raise FileNotFoundError("articles.norm.jsonl not found")
    if 'Tagger' in globals():
        Tagger().run(src, dst)
    else:
        # minimal placeholder
        passthrough_enrich(src, dst, fields={"kw": [], "summary": "", "actors": [], "frames": []})


def _step_gating(bundle_dir: str):
    src = os.path.join(bundle_dir, "kyw_sum.jsonl")
    dst = os.path.join(bundle_dir, "gated.jsonl")
    weights = os.path.join("config", "weights.yaml")
    if not os.path.exists(src):
        raise FileNotFoundError("kyw_sum.jsonl not found")
    if 'Gating' in globals():
        Gating().run(src, dst, weights if os.path.exists(weights) else None)
    else:
        _copy_jsonl(src, dst)


def _step_score_is(bundle_dir: str):
    src = os.path.join(bundle_dir, "gated.jsonl")
    dst = os.path.join(bundle_dir, "scores.jsonl")
    weights = os.path.join("config", "weights.yaml")
    if not os.path.exists(src):
        raise FileNotFoundError("gated.jsonl not found")
    if 'ScoreIS' in globals():
        ScoreIS().run(src, dst, weights if os.path.exists(weights) else None)
    else:
        passthrough_enrich(src, dst, fields={"stage": "IS", "score": 0.3})


def _step_score_dbn(bundle_dir: str):
    dst = os.path.join(bundle_dir, "scores.jsonl")
    if 'ScoreDBN' in globals():
        ScoreDBN().run(bundle_dir, dst)
    else:
        # no-op if missing, keep file as is
        if not os.path.exists(dst):
            open(dst, "a", encoding="utf-8").close()


def _step_score_llm(bundle_dir: str):
    dst = os.path.join(bundle_dir, "scores.jsonl")
    if 'LLMJudge' in globals():
        try:
            LLMJudge().run(bundle_dir, dst)
        except Exception as e:
            logger.warning(f"LLM judge skipped: {e}")
    else:
        # no-op
        if not os.path.exists(dst):
            open(dst, "a", encoding="utf-8").close()


def _step_fusion(bundle_dir: str):
    src = os.path.join(bundle_dir, "scores.jsonl")
    dst = os.path.join(bundle_dir, "fused_scores.jsonl")
    if not os.path.exists(src):
        raise FileNotFoundError("scores.jsonl not found")
    if 'FusionCalibration' in globals():
        FusionCalibration().run(src, dst)
    else:
        _copy_jsonl(src, dst)


def _step_blocks(bundle_dir: str):
    dst = os.path.join(bundle_dir, "blocks.jsonl")
    if 'EDSBlockMatcher' in globals():
        EDSBlockMatcher().run(bundle_dir, dst)
    else:
        # no-op create empty
        open(dst, "a", encoding="utf-8").close()


def _step_scenarios(bundle_dir: str):
    dst = os.path.join(bundle_dir, "scenarios.jsonl")
    if 'ScenarioBuilder' in globals():
        ScenarioBuilder().run(bundle_dir, dst)
    else:
        open(dst, "a", encoding="utf-8").close()


def _step_alerts(bundle_dir: str):
    dst = os.path.join(bundle_dir, "alerts.jsonl")
    if 'AlertDecider' in globals():
        AlertDecider().run(bundle_dir, dst)
    else:
        # simple thresh on fused_scores as fallback
        fused = get_jsonl_df(os.path.join(bundle_dir, "fused_scores.jsonl"))
        fused = ensure_alerts_schema_from_scores(fused)
        if not fused.empty:
            append_jsonl_rows(dst, fused.to_dict(orient="records"))


def _step_event_blocks(bundle_dir: str):
    dst = os.path.join(bundle_dir, "event_blocks.jsonl")
    if 'EventBlockAggregator' in globals():
        EventBlockAggregator().run(bundle_dir, dst)
    else:
        open(dst, "a", encoding="utf-8").close()


def _step_ledger(bundle_dir: str):
    dst = os.path.join(bundle_dir, "ledger.jsonl")
    if 'AuditLedger' in globals():
        AuditLedger().run(bundle_dir, dst)
    else:
        # append simple ledger entry
        append_jsonl_rows(dst, [{"ts": now_iso(), "event": "pipeline_end", "bundle": bundle_dir}])


# =============================================================================
# URL → Article append
# =============================================================================
def append_article_from_url(url: str, bundle_dir: str):
    """
    Fetch URL, extract title/content/basic meta and append to articles.jsonl
    Fallbacks are built-in if readability/requests are missing.
    """
    out = os.path.join(bundle_dir, "articles.jsonl")
    os.makedirs(bundle_dir, exist_ok=True)

    title, text = None, None
    source = url
    lang = None

    if HAVE_REQUESTS:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        html = r.text

        if HAVE_READABILITY:
            doc = Document(html)
            title = safe_strip(doc.short_title())
            content_html = doc.summary(html_partial=True)
            text = html_to_text(content_html)
        else:
            # BeautifulSoup fallback
            text = html_to_text(html)
            if not title:
                soup = BeautifulSoup(html, "html.parser")
                if soup.title:
                    title = safe_strip(soup.title.text)
    else:
        # as last resort, store URL only
        text = f"(fetch disabled) {url}"
        title = url

    article = {
        "id": gen_id(url),
        "ts": now_iso(),
        "title": title or "(no title)",
        "text": text or "",
        "domain": None,
        "region": None,
        "source": source,
        "lang": lang or "unknown"
    }
    append_jsonl_rows(out, [article])


# =============================================================================
# Data helpers
# =============================================================================
def get_jsonl_df(path: str) -> pd.DataFrame:
    rows = []
    if not os.path.exists(path):
        return pd.DataFrame([])
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return pd.DataFrame([])
    return pd.DataFrame(rows)


def append_jsonl_rows(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ensure_alerts_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Standard columns for landing charts
    need = {
        "id": "",
        "ts": now_iso(),
        "title": "",
        "region": "Unknown",
        "domain": "General",
        "score": 0.0,
        "ci_low": 0.0,
        "ci_high": 1.0,
        "intl_flag": False,
        "state": "ALERT",
    }
    if df is None or df.empty:
        return pd.DataFrame([need]).iloc[0:0]
    for k, v in need.items():
        if k not in df.columns:
            df[k] = v
    # coerce types where needed
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    return df


def ensure_alerts_schema_from_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal fallback: convert fused/scores rows to alerts rows (thresholded).
    """
    if scores is None or scores.empty:
        return pd.DataFrame([])
    if "score" not in scores.columns:
        return pd.DataFrame([])
    thr = st.session_state.config.get("threshold", 0.7)
    rows = []
    for _, r in scores.iterrows():
        s = float(r.get("score", 0.0))
        if s >= thr:
            rows.append({
                "id": r.get("id", gen_id(str(_))),
                "ts": r.get("ts", now_iso()),
                "title": r.get("title", ""),
                "region": r.get("region", "Unknown"),
                "domain": r.get("domain", "General"),
                "score": s,
                "ci_low": r.get("ci_low", max(0.0, s - 0.15)),
                "ci_high": r.get("ci_high", min(1.0, s + 0.15)),
                "intl_flag": r.get("intl_flag", False),
                "state": "ALERT"
            })
    return pd.DataFrame(rows)


def _copy_jsonl(src: str, dst: str):
    data = get_jsonl_df(src)
    append_jsonl_rows(dst, data.to_dict(orient="records"))


def passthrough_enrich(src: str, dst: str, fields: Dict[str, Any]):
    data = get_jsonl_df(src)
    if data.empty:
        append_jsonl_rows(dst, [])
        return
    for k, v in fields.items():
        if k not in data.columns:
            data[k] = v
    append_jsonl_rows(dst, data.to_dict(orient="records"))


def html_to_text(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        # Remove scripts/styles
        for t in soup(["script", "style", "noscript"]):
            t.extract()
        return "\n".join([safe_strip(x) for x in soup.get_text("\n").splitlines() if safe_strip(x)])
    except Exception:
        return html


def safe_strip(s: Optional[str]) -> str:
    return s.strip() if isinstance(s, str) else ""


def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def gen_id(seed: str) -> str:
    import hashlib
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


# =============================================================================
# Entrypoint
# =============================================================================
if __name__ == "__main__":
    main()
