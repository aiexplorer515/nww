# app_main.py (실제 호출 버전: Ingest → Normalize → Analyze → Gate → Scoring → Fusion → Blocks → Scenarios → Alerts)
from __future__ import annotations
import os
import io
import re
import json
from datetime import datetime
from collections import Counter

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
import plotly.express as px
import matplotlib.pyplot as plt
from importlib.util import spec_from_file_location, module_from_spec
import inspect
from pyvis.network import Network
import tempfile
from wordcloud import WordCloud
# 파일 상단 어디든(가능하면 import 바로 아래)
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)   # .env 값으로만 채우고 기존 env는 보존
except Exception:
    pass



# --------------------------
# 프로젝트 모듈
# --------------------------
from nwwpkg.ingest import news_collector
from nwwpkg.preprocess import cleaner, tokenizer, embedder
from nwwpkg.eds import frame_tagger
from nwwpkg.score import indicator_scorer, dbn_inference
from nwwpkg.fusion import fuse
from nwwpkg.scenario import scenario_matcher, scenario_predictor
from nwwpkg.analyze import impact_analyzer, frame_shift_detector, hidden_network_detector
from nwwpkg.judge import llm_judge
from nwwpkg.decider import alert_decider
from nwwpkg.ledger import recorder, viewer
from nwwpkg.utils.io import load_jsonl# from nwwpkg.ui.components.page_wrappers import (
from nwwpkg.eventblock import event_classifier, block_accumulator
from nwwpkg.score import crisis_detector
from nwwpkg.scenario import scenario_generator#     render_fusion, render_blocks, render_scenarios, render_ledger
from nwwpkg.network.network_analyze import draw_network
from nwwpkg.analyzer import block_analyzer
# )
from nwwpkg.ui.components.gate_indicator import render_gate_indicator
from nwwpkg.ui.components.scoring_page import render_scoring
from nwwpkg.ui.components.alerts_page import render_alerts as render_alerts_config
from nwwpkg.ui.components.eventblocks_page import render_eventblocks
from nwwpkg.ui.components.risk_snapshot import render_risk_snapshot
from nwwpkg.ui.components.risk_page import render_risk_page
from nwwpkg.ui.components.frames_eval import render_frames_eval
from nwwpkg.ui.components.analytics_basic import render_keywords_and_sentiment
from nwwpkg.ui.page_actor_ranking import page_analyze

# Dashboard 모듈 import
from nwwpkg.ui.dashboard.pipeline_dag import render_pipeline_dag
from nwwpkg.ui.dashboard.stage_qa import render_stage_qa
from nwwpkg.ui.dashboard.alerts import render_alert as render_alert_dashboard, render_alerts_timeseries_inline
from nwwpkg.ui.dashboard.choropleth import render_choropleth
from nwwpkg.ui.dashboard.risk_blocks import render_risk_blocks
from nwwpkg.ui.dashboard.scenarios import render_scenarios
from nwwpkg.ui.dashboard.entities import render_entities
from nwwpkg.ui.dashboard.sources import render_sources


# --- Korean font resolver (Windows/Mac/Linux 지원) ---
import platform, os
from matplotlib import font_manager, rcParams

def get_korean_font_path(custom_path: str | None = None) -> str | None:
    if custom_path and os.path.exists(custom_path):
        return custom_path

    candidates = []
    if platform.system() == "Windows":
        candidates += [
            r"C:\Windows\Fonts\malgun.ttf",            # 맑은고딕
            r"C:\Windows\Fonts\malgunbd.ttf",
            r"C:\Windows\Fonts\NanumGothic.ttf",       # 나눔고딕(있는 경우)
        ]
    elif platform.system() == "Darwin":  # macOS
        candidates += [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/Library/Fonts/AppleGothic.ttf",
        ]
    else:  # Linux
        candidates += [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]

    for p in candidates:
        if os.path.exists(p):
            return p
    return None



PAGES_OFF_DIR = Path(__file__).resolve().parent / "_pages_off"

def _load_module_from_file(stem: str):
    """
    stem: 'fusion' -> _pages_off/fusion.py
    return: module or None
    """
    mod_path = PAGES_OFF_DIR / f"{stem}.py"
    if not mod_path.exists():
        return None
    spec = spec_from_file_location(f"_pages_off.{stem}", str(mod_path))
    if not spec or not spec.loader:
        return None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _call_page(stem: str, func: str, root: Path):
    mod = _load_module_from_file(stem)
    if mod is None:
        st.info(f"`_pages_off/{stem}.py` 를 찾을 수 없습니다.")
        return

    fn = getattr(mod, func, None)
    if not callable(fn):
        st.info(f"`_pages_off/{stem}.py`에 `{func}` 함수가 없습니다.")
        return

    # 시그니처 검사 (안 되면 그냥 ROOT 넣어 호출)
    try:
        sig = inspect.signature(fn)
    except Exception:
        try:
            return fn(root)
        except TypeError:
            return fn(os.getenv("NWW_BUNDLE", root.name))

    params = list(sig.parameters.values())
    bundle_id = os.getenv("NWW_BUNDLE", root.name)

    if len(params) == 1:
        p = params[0]
        # 주석(Annotation) 안전 추출
        try:
            ann = p.annotation
            # ann 이 문자열 / typing 객체 / class 일 수 있음 → 문자열로 통일
            ann_s = (getattr(ann, "__name__", None) or str(ann) or "").lower()
        except Exception:
            ann_s = ""

        name_s = (p.name or "").lower()

        # ROOT(Path)가 더 자연스러운 케이스 감지
        wants_path = (
            "root" in name_s or
            "path" in name_s or
            "dir"  in name_s or
            "path" in ann_s or         # '<class pathlib.Path>' / 'Path' / 'pathlib.Path'
            "pathlib.path" in ann_s
        )

        if wants_path:
            try:
                return fn(root)
            except TypeError:
                return fn(bundle_id)
        else:
            try:
                return fn(bundle_id)
            except TypeError:
                return fn(root)
    else:
        # 인자 2개 이상이면 우선 ROOT 시도 → 실패하면 bundle_id
        try:
            return fn(root)
        except TypeError:
            return fn(bundle_id)


def ensure_matplotlib_korean(font_path: str | None):
    """matplotlib에 폰트를 등록 (타이틀/라벨 한글 깨짐 방지)"""
    if not font_path:
        return
    try:
        font_manager.fontManager.addfont(font_path)
        fp = font_manager.FontProperties(fname=font_path)
        rcParams["font.family"] = fp.get_name()
    except Exception:
        pass
# ===== Korean WordCloud Utilities =====

# --------------------------
# 공통 유틸
# --------------------------
DEFAULT_BUNDLE = "b01"
ROOT = Path(os.getenv("NWW_DATA_HOME","data")) / os.getenv("NWW_BUNDLE","b01")

def bundle_id_from_root(root: Path) -> str:
    # data/<bundle> 구조일 때 마지막 폴더명이 bundle_id
    return root.name

def load_jsonl(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            content = f.read().strip()
            if not content:
                return pd.DataFrame()
            lines = [json.loads(line) for line in content.split('\n') if line.strip()]
        return pd.DataFrame(lines)
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
        return pd.DataFrame()

def save_jsonl(df, path):
    import pandas as pd, numpy as np, json, math, datetime as dt

    # NaN/NaT → None
    df = df.where(pd.notnull(df), None)

    def _default(o):
        # pandas.Timestamp / datetime/date → ISO 문자열
        if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
            # NaT 보호
            try:
                return None if pd.isna(o) else o.isoformat()
            except Exception:
                return None
        # numpy 스칼라 → 파이썬 스칼라
        if isinstance(o, np.generic):
            return o.item()
        return str(o)  # 최후 보루

    with open(path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False, default=_default, allow_nan=False) + "\n")

def load_json_safe(p: Path, default=None):
    if not p.exists(): return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default

def read_jsonl_safe(p: Path) -> pd.DataFrame:
    if not p.exists(): return pd.DataFrame()
    rows=[]
    with p.open(encoding="utf-8-sig") as f:
        for line in f:
            s=line.strip()
            if not s: continue
            try: rows.append(json.loads(s))
            except: pass
    return pd.DataFrame(rows)

def _frames_pretty(x):
    # 다양한 포맷 방어적 처리
    def pick_score(d):
        for k in ("score","conf","p","prob","confidence"):
            if k in d: 
                try: return float(d[k])
                except: pass
        return 0.0
    if isinstance(x, list):
        x_sorted = sorted(x, key=lambda d: pick_score(d), reverse=True)
        return "; ".join(f"{d.get('label') or d.get('frame')}({pick_score(d):.2f})" for d in x_sorted[:5])
    if isinstance(x, dict):
        return f"{x.get('label') or x.get('frame')}({pick_score(x):.2f})"
    return None

# --- add: text column fallback ---
TEXT_CANDIDATES = [
    "text", "clean_text", "body", "content", "article", "message",
    "raw_text", "doc", "desc", "summary"
]

# def ensure_text_column(df):
#     """데이터프레임에 df['text']를 보장한다. 있으면 그대로, 없으면 후보 컬럼을 복사/생성."""
#     import pandas as pd
#     # 1) 후보 중 있는 것 선택
#     for c in TEXT_CANDIDATES:
#         if c in df.columns:
#             if c != "text":
#                 df["text"] = df[c].astype(str)
#             return "text"
#     # 2) title+body 결합 폴백
#     if "title" in df.columns and "body" in df.columns:
#         df["text"] = df["title"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
#         return "text"
#     # 3) 문자열(객체) dtype 첫 컬럼 폴백
#     obj_cols = [c for c in df.columns if df[c].dtype == "O"]
#     if obj_cols:
#         df["text"] = df[obj_cols[0]].astype(str)
#         return "text"
#     # 4) 실패
#     return None
# # --- end add ---

# === [ADD] common helpers ===
from pathlib import Path
import json, pandas as pd

# app_main.py 상단 import 근처
import os
from pathlib import Path

def _bundle_path(bundle: str) -> Path:
    base = Path(os.getenv("NWW_DATA_HOME", "data")) / bundle
    legacy = Path("data/bundles") / bundle
    return base if base.exists() else legacy


def _to_kst_naive(x):
    return (pd.to_datetime(x, errors="coerce", utc=True)
              .dt.tz_convert("Asia/Seoul")
              .dt.tz_localize(None))
# === [/ADD] ===

import os, pandas as pd

def _to_local_naive(s, output_tz: str = "Asia/Seoul"):
    """
    어떤 형태(문자열/naive/aware)든 받아서 KST naive(datetime64[ns])로 변환.
    - 모든 값을 먼저 UTC 기준 timezone-aware 로 통일(pd.to_datetime(..., utc=True))
    - 그다음 원하는 시간대로 tz-convert → tz 정보 제거(naive)
    """
    import pandas as pd

    # 1) 문자열/파이썬 datetime/타입 섞여도 UTC aware 로 통일
    dt = pd.to_datetime(s, errors="coerce", utc=True)

    # 2) KST로 변환 후 tz 제거(naive)
    #   (모두 NaT이어도 에러 없이 NaT 반환)
    dt_local_naive = dt.dt.tz_convert(output_tz).dt.tz_localize(None)

    return dt_local_naive


# >>> START: frames pretty
def _frames_pretty(x):
    # x: [{"label":"긴장상승","conf":0.81}, ...] 혹은 dict/None
    try:
        if isinstance(x, list):
            return ", ".join(f"{d.get('label')}({float(d.get('conf',0)):.2f})" for d in x[:5])
        if isinstance(x, dict):
            return f"{x.get('label')}({float(x.get('conf',0)):.2f})"
    except Exception:
        pass
    return None
# df_sample["frames_pretty"] = df_sample["frames"].apply(_frames_pretty)
# st.dataframe(df_sample[["url","title","frames_pretty", ...]])
# <<< END: frames pretty


# --- Streamlit rerun 호환 래퍼 ---
# =========================
# Helpers: rerun / fused backfill / quick pipeline
# =========================
import hashlib

def _safe_rerun():
    """Streamlit rerun: st.rerun() 우선, 없으면 experimental_rerun()"""
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()  # 일부 버전만 존재
        except Exception:
            st.toast("🔁 페이지를 새로고침(F5) 해주세요.")

def _content_id(row):
    base = str(row.get("url") or row.get("normalized") or row.get("text") or "")
    return hashlib.sha1(base.encode("utf-8", "ignore")).hexdigest() if base else None

def _safe_two_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def _ensure_fused(df: pd.DataFrame, bundle_id: str) -> pd.DataFrame:
    """df에 fused_score가 없을 때 생성 규칙: indicator+dbn → score → 0.0"""
    if "fused_score" in df.columns:
        return df
    out = df.copy()
    if "indicator_score" in out.columns and "dbn_score" in out.columns:
        out["fused_score"] = out.apply(
            lambda r: fuse.combine([float(r.get("indicator_score", 0.0)),
                                    float(r.get("dbn_score", 0.0))]),
            axis=1
        )
        return out
    if "score" in out.columns:
        out["fused_score"] = out["score"].astype(float)
        return out
    out["fused_score"] = 0.0
    return out

def backfill_fused_score(df_blocks: pd.DataFrame, bundle_id: str) -> pd.DataFrame:
    """
    blocks.jsonl 등에 fused_score가 없을 때 복구:
    1) fusion.jsonl → url / content_id 병합
    2) scoring.jsonl → url / content_id 병합
    3) 그래도 없으면 indicator+dbn 재계산 or 0.0
    """
    if not isinstance(df_blocks, pd.DataFrame) or df_blocks.empty:
        return df_blocks
    if "fused_score" in df_blocks.columns:
        return df_blocks

    out = df_blocks.copy()
    if "content_id" not in out.columns:
        out["content_id"] = out.apply(_content_id, axis=1)

    for ref_name in ["fusion.jsonl", "scoring.jsonl"]:
        ref = load_jsonl(f"data/{bundle_id}/{ref_name}")
        if ref.empty:
            continue
        ref = _ensure_fused(ref, bundle_id)

        # url 병합
        common = _safe_two_cols(ref, ["url", "fused_score"])
        if {"url", "fused_score"}.issubset(common) and "url" in out.columns:
            merged = out.merge(ref[common], on="url", how="left", suffixes=("", "_f"))
            if "fused_score_f" in merged.columns:
                merged["fused_score"] = merged["fused_score"].fillna(merged["fused_score_f"])
                merged = merged.drop(columns=[c for c in ["fused_score_f"] if c in merged.columns])
            if "fused_score" in merged.columns and merged["fused_score"].notna().any():
                out = merged

        # content_id 병합
        if "content_id" not in ref.columns:
            ref = ref.copy(); ref["content_id"] = ref.apply(_content_id, axis=1)
        common = _safe_two_cols(ref, ["content_id", "fused_score"])
        if {"content_id", "fused_score"}.issubset(common):
            merged = out.merge(ref[common], on="content_id", how="left", suffixes=("", "_f"))
            if "fused_score_f" in merged.columns:
                merged["fused_score"] = merged["fused_score"].fillna(merged["fused_score_f"])
                merged = merged.drop(columns=[c for c in ["fused_score_f"] if c in merged.columns])
            if "fused_score" in merged.columns and merged["fused_score"].notna().any():
                out = merged

    out = _ensure_fused(out, bundle_id)
    return out

def _quick_pipeline(bundle_id: str) -> bool:
    """
    ingest → normalize → analyze → gate → scoring → blocks → scenarios → alerts
    빠른 일괄 처리(MVP). 파일을 단계별로 덮어씀.
    """
    # 1) ingest → normalize
    df_ing = load_jsonl(f"data/{bundle_id}/ingest.jsonl")
    if df_ing.empty:
        return False

    df_norm = df_ing.copy()
    df_norm["normalized"] = df_norm["text"].astype(str).apply(cleaner.normalize)
    df_norm["sentences"] = df_norm["normalized"].apply(tokenizer.split_sentences)
    save_jsonl(df_norm, f"data/{bundle_id}/normalize.jsonl")

    # 2) analyze
    df_ana = df_norm.copy()
    df_ana["frames"] = df_ana["normalized"].apply(frame_tagger.tag)
    save_jsonl(df_ana, f"data/{bundle_id}/analyze.jsonl")

    # 3) gate
    df_gate = df_ana.copy()
    df_gate["indicator_score"] = df_gate["frames"].apply(indicator_scorer.run)
    save_jsonl(df_gate, f"data/{bundle_id}/gate.jsonl")

    # 4) scoring
    ind_scores, dbn_scores, fused_scores = [], [], []
    prev_frames = None
    for _, r in df_gate.iterrows():
        fr = r.get("frames", [])
        ind = float(r.get("indicator_score", 0.0))
        dbn = dbn_inference.run(fr, prev_frames=prev_frames)
        fused = fuse.combine([ind, dbn])
        ind_scores.append(ind); dbn_scores.append(dbn); fused_scores.append(fused)
        prev_frames = fr

    df_score = df_gate.copy()
    df_score["dbn_score"]   = dbn_scores
    df_score["fused_score"] = fused_scores
    save_jsonl(df_score, f"data/{bundle_id}/scoring.jsonl")

    # 5) blocks
    def _primary(fs):
        if not fs: return "General"
        best = sorted(fs, key=lambda f: f.get("score", 0), reverse=True)[0]
        return best.get("frame", "General")
    df_blk = df_score.copy()
    df_blk["block"] = df_blk["frames"].apply(_primary)
    keep = [c for c in ["url","date","normalized","frames","indicator_score","dbn_score","fused_score"] if c in df_blk.columns]
    out_blk = df_blk[keep + ["block"]].copy()
    save_jsonl(out_blk, f"data/{bundle_id}/blocks.jsonl")

    # 6) scenarios
    matched, predicted = [], []
    for _, r in out_blk.iterrows():
        sents = r.get("sentences", [])
        if not sents and isinstance(r.get("normalized",""), str):
            sents = tokenizer.split_sentences(r["normalized"])
        vecs = embedder.embed(sents)
        matched.append(scenario_matcher.match(vecs, top_k=3))
        predicted.append(scenario_predictor.generate(r.get("normalized","")))
    df_scen = out_blk.copy()
    df_scen["scenario_matched"]   = matched
    df_scen["scenario_predicted"] = predicted
    save_jsonl(df_scen, f"data/{bundle_id}/scenarios.jsonl")

    # 7) alerts
    df_alert = df_scen.copy()
    df_alert["decision"] = df_alert["fused_score"].apply(alert_decider.decide)
    def _to_level(dec: str) -> str:
        if "High" in dec: return "High"
        if "Medium" in dec: return "Medium"
        return "Low"
    df_alert["alert_level"] = df_alert["decision"].apply(_to_level)
    save_jsonl(df_alert, f"data/{bundle_id}/alerts.jsonl")
    return True

# ===== Plotly 팔레트 유틸 =====
import itertools
import plotly.express as px

# 명목형(discrete) 팔레트들 순환
_PALLETS = [
    px.colors.qualitative.Set1,
    px.colors.qualitative.Set2,
    px.colors.qualitative.Set3,
    px.colors.qualitative.Bold,
    px.colors.qualitative.Dark24,
    px.colors.qualitative.Pastel,
    px.colors.qualitative.D3,
    px.colors.qualitative.T10,
    px.colors.qualitative.G10,
]
_pal_iter = itertools.cycle(_PALLETS)

def next_palette():
    """차트마다 다른 색 팔레트(qualitative)"""
    return next(_pal_iter)

# 연속(continuous) 팔레트들 순환 (지도/heatmap 등)
_CSEQS = [
    px.colors.sequential.Viridis,
    px.colors.sequential.Plasma,
    px.colors.sequential.Magma,
    px.colors.sequential.Cividis,
    px.colors.sequential.Blues,
    px.colors.sequential.Greens,
    px.colors.sequential.Reds,
    px.colors.sequential.Oranges,
    px.colors.sequential.Turbo,
]
_cseq_iter = itertools.cycle(_CSEQS)

def next_cscale():
    """차트마다 다른 연속 팔레트"""
    return next(_cseq_iter)

# ===== Alerts Time-series Helper =====
def _alert_timeseries(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    df_alerts -> (date, alert_level) 시계열 집계
    freq: "D"(일), "W"(주), "M"(월)
    return: long-form [date, alert_level, count]
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "alert_level", "count"])
    need = {"date", "alert_level"}
    if not need.issubset(df.columns):
        return pd.DataFrame(columns=["date", "alert_level", "count"])

    out = df.copy()
    try:
        out["date"] = pd.to_datetime(out["date"])
    except Exception:
        # date가 문자열이 아니거나 파싱 실패 → 빈 프레임 반환
        return pd.DataFrame(columns=["date", "alert_level", "count"])

    # 타임존 제거 및 주기별 버킷팅
    # out["date"] = out["date"].dt.tz_localize(None, nonexistent="shift_forward", ambiguous="NaT", errors="ignore")
    out["date"] = (
        pd.to_datetime(out["date"], errors="coerce", utc=True)   # tz-aware UTC
            .dt.tz_convert("Asia/Seoul")                           # KST로 변환 (DST 모호성 없음)
            .dt.tz_localize(None)                                  # tz 제거 → naive KST
    )
    out["date"] = out["date"].dt.to_period(freq).dt.to_timestamp()

    # 경보 레벨 정규화
    lvl_map = {"HIGH": "High", "high": "High", "H": "High",
               "MEDIUM": "Medium", "medium": "Medium", "M": "Medium",
               "LOW": "Low", "low": "Low", "L": "Low"}
    out["alert_level"] = out["alert_level"].astype(str).map(lambda x: lvl_map.get(x, x))
    order = ["High", "Medium", "Low"]
    out["alert_level"] = pd.Categorical(out["alert_level"], categories=order, ordered=True)

    grp = out.groupby(["date", "alert_level"], observed=True).size().reset_index(name="count")
    grp = grp.sort_values(["date", "alert_level"])
    return grp

def _to_wide(ts_long: pd.DataFrame) -> pd.DataFrame:
    """long-form(timeseries) → wide index=date, columns(level)"""
    if ts_long.empty:
        return pd.DataFrame(columns=["date","High","Medium","Low"])
    wide = ts_long.pivot(index="date", columns="alert_level", values="count").fillna(0)
    # 레벨 없을 수 있으니 보장
    for c in ["High","Medium","Low"]:
        if c not in wide.columns:
            wide[c] = 0
    wide = wide[["High","Medium","Low"]]
    wide = wide.sort_index()
    return wide

def reset_index_as_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index()
    # 이미 'date' 있으면 그대로, 없고 'index'만 있으면 date로
    if "date" not in out.columns and "index" in out.columns:
        out = out.rename(columns={"index":"date"})
    # 혹시 모를 중복 방지
    if "date" in out.columns and getattr(out.columns, "duplicated", lambda: False)().any():
        # 첫 번째 date만 남기고 나머지 date* 컬럼으로 변경
        seen = False
        new_cols = []
        for c in out.columns:
            if c == "date":
                if not seen:
                    new_cols.append(c); seen = True
                else:
                    new_cols.append("date_idx")
            else:
                new_cols.append(c)
        out.columns = new_cols
    return out

# --- hidden_network_detector 호출 호환 레이어 ---
def _postfilter_graph(graph: dict, min_count: int, stopwords: set[str]) -> dict:
    """구버전 detect() 반환 그래프에 대해 stopwords 제거 + min_count 임계 적용"""
    if not isinstance(graph, dict):
        return {"nodes": {}, "edges": {}}

    nodes = graph.get("nodes", {}) or {}
    edges = graph.get("edges", {}) or {}

    # 노드: 불용어 제거 + 임계 적용
    nodes2 = {k: int(v) for k, v in nodes.items()
              if isinstance(v, (int, float)) and v >= min_count and k not in stopwords}

    # 엣지: 노드 생존 여부 + 임계 적용
    edges2 = {}
    for key, c in (edges.items() if isinstance(edges, dict) else []):
        if not isinstance(c, (int, float)) or c < min_count:
            continue
        if isinstance(key, tuple):
            a, b = key
            edge_key = f"{a}—{b}"
        else:
            parts = str(key).split("—")
            if len(parts) != 2:
                continue
            a, b = parts
            edge_key = key
        if a in nodes2 and b in nodes2:
            edges2[edge_key] = int(c)

    return {"nodes": nodes2, "edges": edges2}

def _detect_network_compat(sentences, min_count: int, stopwords: set[str]):
    """
    신버전: detect(sentences, min_count=..., stopwords=...)
    구버전: detect(sentences)만 지원 → 반환 그래프를 후처리로 필터링
    """
    try:
        return hidden_network_detector.detect(sentences, min_count=min_count, stopwords=stopwords)
    except TypeError:
        g = hidden_network_detector.detect(sentences)
        return _postfilter_graph(g, min_count, stopwords)


# ===== Plotly 팔레트 유틸 =====
import itertools
import plotly.express as px

_PALLETS = [
    px.colors.qualitative.Set1, px.colors.qualitative.Set2, px.colors.qualitative.Set3,
    px.colors.qualitative.Bold, px.colors.qualitative.Dark24, px.colors.qualitative.Pastel,
    px.colors.qualitative.D3, px.colors.qualitative.T10, px.colors.qualitative.G10,
]
_pal_iter = itertools.cycle(_PALLETS)
def next_palette(): return next(_pal_iter)

_CSEQS = [
    px.colors.sequential.Viridis, px.colors.sequential.Plasma, px.colors.sequential.Magma,
    px.colors.sequential.Cividis, px.colors.sequential.Blues, px.colors.sequential.Greens,
    px.colors.sequential.Reds, px.colors.sequential.Oranges, px.colors.sequential.Turbo,
]
_cseq_iter = itertools.cycle(_CSEQS)
def next_cscale(): return next(_cseq_iter)

# ===== GeoJSON & 행정코드 유틸 =====
import json, os

def load_geojson(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def _detect_sig_props(geo: dict):
    """GeoJSON의 properties에서 코드/이름 키 자동 탐지"""
    if not geo or "features" not in geo or not geo["features"]:
        return None, None
    props = geo["features"][0].get("properties", {})
    # 흔한 키 패턴
    candidates_code = ["SIG_CD", "adm_cd", "ADM_CD"]
    candidates_name = ["SIG_KOR_NM", "sig_kor_nm", "ADM_NM", "adm_nm", "EMD_KOR_NM"]
    code_key = next((k for k in candidates_code if k in props), None)
    name_key = next((k for k in candidates_name if k in props), None)
    return code_key, name_key

def attach_sig_cd_from_name(df: pd.DataFrame, geo: dict, name_col: str) -> pd.DataFrame:
    """df[name_col] (예: '서울특별시 강남구')를 GeoJSON의 시군구 코드로 매핑해 df['sig_cd'] 생성"""
    code_key, name_key = _detect_sig_props(geo)
    if not (code_key and name_key) or name_col not in df.columns:
        return df
    # GeoJSON name->code 사전
    pairs = [(f["properties"][name_key], f["properties"][code_key]) for f in geo["features"]]
    name2code = {str(n): str(c) for n, c in pairs if n and c}
    out = df.copy()
    out["sig_cd"] = out[name_col].astype(str).map(name2code)
    return out

def ensure_sig_cd(df: pd.DataFrame, geo: dict) -> pd.DataFrame:
    """df에 sig_cd가 없으면 region/sigungu에서 유추"""
    if "sig_cd" in df.columns:
        return df
    # 우선순위: sigungu(시군구 전체명) → region(권역명은 매칭 낮음)
    if "sigungu" in df.columns:
        return attach_sig_cd_from_name(df, geo, "sigungu")
    if "region" in df.columns:
        # region에 시군구가 직접 들어올 가능성이 낮음. 그래도 시도.
        return attach_sig_cd_from_name(df, geo, "region")
    return df



# ==========================
# 1) Ingest
# ==========================
# ---------------- Ingest 탭 ----------------
def page_ingest(bundle_id="sample"):
    st.header("📰 Ingest – 기사 수집")

    url = st.text_input("기사 URL 입력")
    text_input = st.text_area("기사 원문 입력(선택)", height=200)

    if st.button("기사 저장"):
        if not url and not text_input:
            st.warning("⚠️ URL 또는 기사 원문 입력 필요")
            return

        # 실제 기사 수집
        source = "Manual"; title = None; published = None
        text = text_input or ""
        if url:
            ing = news_collector.collect(url)
            text = text or ing.get("text", "")
            source = ing.get("source") or news_collector.get_domain(url)
            title = ing.get("title"); published = ing.get("published")

        df_new = pd.DataFrame([{
            "url": url or None,
            "text": text,
            "date": datetime.today().strftime("%Y-%m-%d"),
            "source": source,
            "title": title,
            "published": published
        }])
        recorder.save(bundle_id, "ingest.jsonl", df_new, dedup_on="url")
        st.success("✅ 기사 저장 완료")

    # ===========================
    # 📑 저장된 기사 목록 (+삭제)
    # ===========================
    st.subheader("📑 저장된 기사 목록")
    ingest_path = f"data/{bundle_id}/ingest.jsonl"
    df = load_jsonl(ingest_path)

    if df.empty:
        st.info("아직 저장된 기사가 없습니다.")
        return

    # 안정적 삭제를 위한 내부 rowid 보강
    df = df.copy()
    df["__rowid"] = df.index

    # 표시용 테이블 구성
    view_cols = [c for c in ["__rowid", "date", "source", "title", "url", "published", "text"] if c in df.columns]
    df_view = df[view_cols].copy()
    df_view["삭제"] = False  # 체크박스 컬럼

    edited = st.data_editor(
        df_view,
        key=f"ingest_editor_{bundle_id}",
        hide_index=True,
        width='stretch',
        column_config={
            "__rowid": st.column_config.NumberColumn("rowid", help="내부 식별자", disabled=True),
            "삭제": st.column_config.CheckboxColumn("삭제", help="삭제할 행 선택"),
            "text": st.column_config.TextColumn("text", disabled=True)
        }
    )

    c1, c2, c3 = st.columns([1,1,1])

    # 🗑 선택 행 삭제
    if c1.button("🗑 선택 행 삭제"):
        to_del = edited.loc[edited["삭제"] == True, "__rowid"].tolist()
        if not to_del:
            st.info("선택된 행이 없습니다.")
        else:
            new_df = df[~df["__rowid"].isin(to_del)].drop(columns="__rowid")
            save_jsonl(new_df, ingest_path)
            st.success(f"✅ {len(to_del)}건 삭제 완료")
            _safe_rerun()

    # 🧯 URL 기준 중복 제거(최신만 유지)
    if c2.button("🧯 URL 중복 제거"):
        if "url" in df.columns:
            before = len(df)
            # 최신 날짜를 남기도록 정렬 후 drop_duplicates
            tmp = df.sort_values(by=["date"], ascending=True) if "date" in df.columns else df
            new_df = tmp.drop_duplicates(subset=["url"], keep="last").drop(columns="__rowid", errors="ignore")
            save_jsonl(new_df, ingest_path)
            st.success(f"✅ 중복 제거: {before - len(new_df)}건 정리")
            _safe_rerun()
        else:
            st.warning("URL 컬럼이 없어 중복 제거를 수행할 수 없습니다.")

    # 🧨 전체 삭제(초기화)
    if c3.button("🧨 전체 삭제(초기화)"):
        empty = pd.DataFrame(columns=[c for c in df.columns if c != "__rowid"])
        save_jsonl(empty, ingest_path)
        st.success("✅ 전체 삭제 완료")
        _safe_rerun()

    # 간단 분포 차트
    if "source" in df.columns:
        counts = df["source"].fillna("Manual").value_counts().reset_index()
        counts.columns = ["source", "count"]
        fig = px.bar(counts, x="source", y="count", text="count",
                     labels={"source": "언론사", "count": "기사 수"},
                     title="언론사별 기사 분포")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width='stretch')


# ==========================
# 2) Normalize (전처리)
# ==========================
from nwwpkg.preprocess import cleaner, tokenizer, entities_extractor
from nwwpkg.utils.io import (
    load_jsonl,
    save_jsonl,
    count_lines,
    ensure_text_column,
)


def page_normalize(bundle_id: str):
    st.header("🔤 Normalize – 텍스트 정규화 & 엔티티 추출")

    # 데이터 로드
    ingest_path = Path(f"data/{bundle_id}/ingest.jsonl")
    norm_path = Path(f"data/{bundle_id}/normalize.jsonl")
    ent_path = Path(f"data/{bundle_id}/entities.jsonl")

    df = load_jsonl(ingest_path)
    if df.empty:
        st.info("⚠️ 먼저 Ingest 단계에서 기사를 입력하세요.")
        return

    # 본문 텍스트 컬럼 확인
    col = ensure_text_column(df)
    if col is None:
        st.warning("본문 텍스트 컬럼(text/clean_text/body)을 찾을 수 없습니다.")
        return

    # 실행 버튼
    if st.button("⚡ 정규화 + 엔티티 추출 실행"):
        # 정규화
        df["normalized"] = df[col].astype(str).apply(cleaner.normalize)
        df["sentences"] = df["normalized"].apply(tokenizer.split_sentences)

        # 저장
        save_jsonl(norm_path, df.to_dict("records"))

        # 엔티티 추출
        ents = []
        for _, row in df.iterrows():
            text = row.get("normalized") or row.get(col, "")
            entities = entities_extractor.run(text)
            ents.append({
                "source_id": row.get("id", None),
                "text": text,
                "entities": entities
            })
        save_jsonl(ent_path, ents)

        st.success("✅ 정규화 및 엔티티 추출 완료")

    # 📑 정규화 데이터 미리보기
    if norm_path.exists():
        norm_df = load_jsonl(norm_path)
        st.subheader("📑 원문 vs 정규화 결과")
        if "normalized" in norm_df.columns and col in norm_df.columns:
            st.dataframe(norm_df[[col, "normalized"]], use_container_width=True)
        else:
            st.warning("⚠️ 표시할 본문/정규화 컬럼이 없습니다. (text/clean_text/body/normalized 확인 필요)")

        # 문장 수 분포
        if "sentences" in norm_df.columns:
            norm_df["sent_count"] = norm_df["sentences"].apply(len)
            fig = px.histogram(norm_df, x="sent_count", nbins=20, title="문장 수 분포")
            st.plotly_chart(fig, use_container_width=True)

        # 다운로드
        st.subheader("📥 정규화 데이터 다운로드")
        csv_buffer = io.StringIO()
        norm_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        st.download_button("⬇️ CSV 다운로드", data=csv_buffer.getvalue(),
                           file_name=f"{bundle_id}_normalized.csv", mime="text/csv")

        jsonl_buffer = io.StringIO()
        for _, row in norm_df.iterrows():
            jsonl_buffer.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        st.download_button("⬇️ JSONL 다운로드", data=jsonl_buffer.getvalue(),
                           file_name=f"{bundle_id}_normalized.jsonl", mime="application/json")

    # 🧑‍🤝‍🧑 엔티티 추출 결과
    if ent_path.exists():
        ents_df = load_jsonl(ent_path)

        # dict/list → 문자열 변환 (표시용)
        if "entities" in ents_df.columns:
            ents_df["entities"] = ents_df["entities"].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)
            )

        st.subheader("🧑‍🤝‍🧑 엔티티 추출 결과")
        st.dataframe(ents_df, use_container_width=True)
        # st.json(ents_df.to_dict("records")[:3])  # 상위 3개 미리보기

        # 다운로드
        ent_buffer = io.StringIO()
        for _, row in ents_df.iterrows():
            ent_buffer.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        st.download_button("⬇️ 엔티티 JSONL 다운로드", data=ent_buffer.getvalue(),
                           file_name=f"{bundle_id}_entities.jsonl", mime="application/json")



# ==========================
# 3) Analyze (프레임/네트워크/워드클라우드)
# ==========================
def extract_keywords(texts, topn=20):
    words = " ".join(texts).split()
    counter = Counter(words)
    return pd.DataFrame(counter.most_common(topn), columns=["word", "freq"])

def page_analyze(bundle_id: str):
    """
    🔍 Analyze – 프레임/네트워크/키워드/워드클라우드(+감정 예시, 글로벌 네트워크)
    - frame_tagger.tag() : 기사별 프레임 태깅
    - hidden_network_detector.detect(sentences) : 기사별 & 전체 공출현 네트워크
    - 한글 워드클라우드: 폰트 자동탐색+검증, 한글 토큰만 추출
    - 결과 저장: data/{bundle_id}/analyze.jsonl
    """
    import os, platform, re
    from collections import Counter
    from PIL import ImageFont
    from matplotlib import font_manager, rcParams
    from wordcloud import WordCloud
    import plotly.express as px
    import matplotlib.pyplot as plt

    # ------------------------
    # 내부 유틸 (이 함수 안에서만 사용)
    # ------------------------
    KO_FONT_CANDIDATES = {
        "Windows": [
            r"C:\Windows\Fonts\malgun.ttf",
            r"C:\Windows\Fonts\malgunbd.ttf",
            r"C:\Windows\Fonts\NanumGothic.ttf",
        ],
        "Darwin": [  # macOS
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/Library/Fonts/AppleGothic.ttf",
        ],
        "Linux": [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ],
    }

    def _font_supports_korean(font_path: str) -> bool:
        try:
            f = ImageFont.truetype(font_path, 24)
            return f.getlength("한글테스트") > 0
        except Exception:
            return False

    def resolve_korean_font(custom_path: str | None = None) -> str | None:
        # 1) 사용자가 세션에 넣어둔 경로 우선
        if custom_path and os.path.exists(custom_path) and _font_supports_korean(custom_path):
            return custom_path
        # 2) OS 후보군 순회
        for p in KO_FONT_CANDIDATES.get(platform.system(), []):
            if os.path.exists(p) and _font_supports_korean(p):
                return p
        return None

    def set_matplotlib_korean(font_path: str | None):
        if not font_path:
            return
        try:
            font_manager.fontManager.addfont(font_path)
            fp = font_manager.FontProperties(fname=font_path)
            rcParams["font.family"] = fp.get_name()
        except Exception:
            pass

    KO_TOKEN = re.compile(r"[가-힣]{2,}")  # 2글자 이상 한글만

    def korean_tokens(texts: list[str], stopwords: set[str] | None = None) -> list[str]:
        stopwords = stopwords or set()
        toks = []
        for t in texts:
            if not isinstance(t, str):
                continue
            words = KO_TOKEN.findall(t)
            words = [w for w in words if len(w) >= 2 and w not in stopwords]
            toks.extend(words)
        return toks

    # ------------------------
    # 데이터 로드
    # ------------------------
    st.header("🔍 Analyze – 프레임/네트워크/키워드")
    df = load_jsonl(f"data/{bundle_id}/normalize.jsonl")
    if df.empty:
        st.info("⚠️ 먼저 Normalize 단계를 실행하세요.")
        return

    # 텍스트 컬럼 확보
    if "normalized" not in df.columns:
        df["normalized"] = df.get("text", "").astype(str)

    # 문장 컬럼 보강
    if "sentences" not in df.columns:
        df["sentences"] = df["normalized"].apply(tokenizer.split_sentences)

    # ------------------------
    # 프레임 태깅 (기사별)
    # ------------------------
    df["frames"] = df["normalized"].apply(frame_tagger.tag)

    # ------------------------
    # 네트워크 탐지 (기사별 + 전체(Global))
    # ------------------------
    st.subheader("🕸️ 네트워크 탐지 옵션")
    min_count = st.slider("최소 등장 빈도(min_count)", 1, 20, 5, 1)
    sw_text = st.text_area(
        "불용어(쉼표로 구분, 추가)",
        value="기사원문,입력,오후,사진,연합뉴스,YTN,newsis,kmn,서비스,보내기,변경하기,사용하기,관련,대한,본문,글자,수정,변환",
        height=80
    )
    custom_sw = {w.strip() for w in sw_text.split(",") if w.strip()}

    # ---- 기사별 네트워크 계산 (옵션 적용) ----
    df["network"] = df["sentences"].apply(lambda s: _detect_network_compat(s, min_count, custom_sw))
    df["net_nodes"] = df["network"].apply(lambda g: len((g or {}).get("nodes", {})))
    df["net_edges"] = df["network"].apply(lambda g: len((g or {}).get("edges", {})))

    # 글로벌 네트워크
    all_sents = [sent for sents in df["sentences"] for sent in (sents or [])]
    global_net = _detect_network_compat(all_sents, min_count, custom_sw)
    # st.subheader("🕸️ 글로벌 네트워크 프리뷰")
    # st.json(global_net)

    # # (프리뷰 표도 원하면)
    # top_nodes = sorted(global_net["nodes"].items(), key=lambda x: x[1], reverse=True)[:20]
    # st.dataframe(pd.DataFrame(top_nodes, columns=["node","count"]))
    # ------------------------
    # 키워드 Top-N (한글 전용)
    # ------------------------
    stopwords = {"기자", "연합뉴스", "뉴스", "단독", "속보", "종합", "사진"}
    toks = korean_tokens(df["normalized"].fillna("").tolist(), stopwords=stopwords)
    
    # ------------------------
    # 감정 분포 (예시 데이터 유지)
    # ------------------------
    

    # ------------------------
    # 워드클라우드 (한글 폰트 자동탐색 + 검증)
    # ------------------------
    st.subheader("☁️ 워드클라우드 (한글)")
    # 세션에 폰트 경로 저장해둔 경우 우선 사용 (없으면 자동탐색)
    custom_font = st.session_state.get("kofont") if "kofont" in st.session_state else ""
    font_path = resolve_korean_font(custom_font or None)

    if not font_path:
        st.error("한글 지원 폰트를 찾지 못했습니다. 예) Windows: C:\\Windows\\Fonts\\malgun.ttf")
    else:
        set_matplotlib_korean(font_path)
        corpus = " ".join(toks)
        if corpus.strip():
            wc = WordCloud(
                width=1000,
                height=500,
                background_color="white",
                font_path=font_path,
                prefer_horizontal=1.0,
                collocations=False
            ).generate(corpus)
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.imshow(wc, interpolation="bilinear")
            ax2.axis("off")
            st.caption(f"font: {font_path}")
            st.pyplot(fig2, clear_figure=True)
        else:
            st.warning("유효한 한글 토큰이 없어 워드클라우드를 생성할 수 없습니다.")

    # ------------------------
    # 프리뷰(기사별/글로벌 네트워크) & 저장
    # ------------------------
    # >>> START: network real
    import pandas as pd
    import networkx as nx
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 한글 폰트 (Windows 기준: Malgun Gothic)
    matplotlib.rc("font", family="Malgun Gothic")
    matplotlib.rcParams["axes.unicode_minus"] = False

    netp = _bundle_path(bundle_id) / "network.json"
    net = load_json_safe(netp, default=None)
    st.subheader("🌐 글로벌 네트워크 프리뷰")

    if net:
        nodes = net.get("nodes", {})
        edges = net.get("edges", [])
        c_nodes, c_edges = len(nodes), len(edges)
        c_nonzero = sum(1 for e in edges if e.get("weight", 0) > 0)

        # 기본 메트릭
        col1, col2, col3 = st.columns(3)
        col1.metric("노드 수", c_nodes)
        col2.metric("엣지 수", c_edges)
        col3.metric("양의 가중치 엣지", c_nonzero)

        if len(edges) > 0:
            df_e = pd.DataFrame(edges)
            if "weight" in df_e.columns and "co" in df_e.columns:
                df_e = df_e.sort_values(["weight", "co"], ascending=False).head(50)
            else:
                df_e = df_e.head(50)
            st.dataframe(df_e, use_container_width=True, height=360)

            # 네트워크 그래프 생성
            G = nx.Graph()
            for e in df_e.to_dict("records"):
                if "source" in e and "target" in e:  # 방어적 체크
                    G.add_edge(e["source"], e["target"], weight=e.get("weight", 1))

            # ⚠️ 여기서 반드시 확인
            if G.number_of_nodes() > 0:
                degrees = dict(G.degree())
                weights = [G[u][v].get("weight", 1) for u, v in G.edges()]

                plt.figure(figsize=(6, 4))
                pos = nx.kamada_kawai_layout(G)

                nx.draw_networkx_nodes(
                    G, pos,
                    node_color="skyblue",
                    node_size=[degrees.get(n, 1) * 120 for n in G.nodes()],
                    alpha=0.8
                )
                nx.draw_networkx_edges(
                    G, pos,
                    edge_color="gray",
                    width=[w * 0.15 for w in weights],
                    alpha=0.5
                )

                # 라벨 한글 깨짐 방지
                import matplotlib.font_manager as fm
                font_path = "C:/Windows/Fonts/malgun.ttf"
                fontprop = fm.FontProperties(fname=font_path)

                labels = {n: n for n in G.nodes() if not str(n).isdigit()}
                nx.draw_networkx_labels(
                    G, pos, labels,
                    font_size=9,
                    font_family=fontprop.get_name()
                )

                plt.axis("off")
                st.pyplot(plt, clear_figure=True)
            else:
                st.info("그래프를 그릴 노드가 없습니다.")
        else:
            st.info("엣지 데이터가 없습니다. min_count/window 파라미터를 낮춰 재생성하세요.")
    else:
        st.info("network.json이 없습니다. 네트워크 생성 스크립트를 먼저 실행하세요.")


    # 📑 프레임 표(Frames Table)

    st.subheader("📑 프레임 표 (실데이터)")
    p_frames = _bundle_path(bundle_id) / "frames.jsonl"
    df_frames = read_jsonl_safe(p_frames)

    if not df_frames.empty:
        # 가능한 컬럼 추출
        for c in ("frames","frame_preds","frame_labels","labels"):
            if c in df_frames.columns:
                df_frames["frames_pretty"] = df_frames[c].apply(_frames_pretty)
                break
        # 대표 프레임/점수
        if "frames_pretty" not in df_frames:
            df_frames["frames_pretty"] = None
        # 보여줄 컬럼 구성(존재하는 것만)
        show = [c for c in ["url","title","frames_pretty","fused_score","date","id"] if c in df_frames.columns]
        st.dataframe(df_frames[show].head(200), use_container_width=True, height=420)
        st.caption(f"소스: {p_frames}")
    else:
        st.info("frames.jsonl이 없습니다. (키워드→룰부스트→프레임 분류 파이프라인 실행 필요)")

# ==========================
# 4) Gate (프레임→지표점수)
# ==========================
def page_gate(bundle_id: str):
    st.header("🚪 Gate – 프레임 기반 체크(Indicator)")

    df = load_jsonl(f"data/{bundle_id}/analyze.jsonl")
    if df.empty or "frames" not in df.columns:
        st.warning("먼저 Analyze 단계를 실행하세요.")
        return

    # 실제 indicator_scorer 호출
    df["indicator_score"] = df["frames"].apply(indicator_scorer.run)
    st.subheader("프레임 & 지표 점수")
    st.dataframe(df[["normalized", "frames", "indicator_score"]].head(), width='stretch')

    save_jsonl(df, f"data/{bundle_id}/gate.jsonl")
    st.success("✅ Gate 결과 저장 완료")

# ==========================
# 5) Scoring (DBN + Fused)
# ==========================
def page_scoring(bundle_id: str):
    st.header("📊 Scoring – DBN 추론 + 융합 점수 + 영향/시프트")

    df = load_jsonl(f"data/{bundle_id}/gate.jsonl")
    if df.empty or "frames" not in df.columns or "indicator_score" not in df.columns:
        st.warning("먼저 Gate 단계를 실행하세요.")
        return

    ind_scores, dbn_scores, fused_scores = [], [], []
    impacts, shifts = [], []
    history = []  # 직전 프레임 스냅샷 누적

    for _, row in df.iterrows():
        frames = row.get("frames", [])
        ind = float(row.get("indicator_score", 0.0))

        # DBN(간이): 직전 스냅샷과 비교
        prev_frames = history[-1] if history else None
        dbn_score = dbn_inference.run(frames, prev_frames=prev_frames)

        # Fused
        fused = fuse.combine([ind, dbn_score])

        # ✅ 영향 분석(impact_analyzer)
        imp = impact_analyzer.run(frames, fused)

        # ✅ 프레임 시프트(frame_shift_detector)
        if history:
            sh = frame_shift_detector.detect(history + [frames])
        else:
            sh = {"shift_detected": False, "shift_score": 0.0, "changed": []}

        # 누적
        history.append(frames)
        ind_scores.append(ind); dbn_scores.append(dbn_score); fused_scores.append(fused)
        impacts.append(imp); shifts.append(sh)

    df["dbn_score"]   = dbn_scores
    df["fused_score"] = fused_scores
    df["impact"]      = impacts
    df["shift"]       = shifts

    st.subheader("점수 & 분석 요약")
    st.dataframe(df[["indicator_score","dbn_score","fused_score","impact","shift"]].head(), width='stretch')

    # 시계열
    df["date"] = pd.to_datetime(df.get("date", datetime.today().strftime("%Y-%m-%d")))
    fig = px.line(df, x="date", y=["indicator_score", "dbn_score", "fused_score"], title="점수 시계열")
    st.plotly_chart(fig, width='stretch')

    save_jsonl(df, f"data/{bundle_id}/scoring.jsonl")
    st.success("✅ Scoring 결과 저장 완료")


# ==========================
# 6) Fusion (시각 비교)
# ==========================
def page_fusion(bundle_id: str):
    st.header("⚡ Fusion – 점수 비교/검증")

    df = load_jsonl(f"data/{bundle_id}/scoring.jsonl")
    if df.empty or "fused_score" not in df.columns:
        st.warning("먼저 Scoring 단계를 실행하세요.")
        return

    fig = px.scatter(df, x="indicator_score", y="fused_score", trendline="ols",
                     title="Indicator vs Fused")
    st.plotly_chart(fig, width='stretch')

    st.dataframe(df[["indicator_score", "dbn_score", "fused_score"]].head(), width='stretch')
    save_jsonl(df, f"data/{bundle_id}/fusion.jsonl")
    st.success("✅ Fusion 결과 저장 완료")

# ==========================
# 7) Blocks (주 프레임 기반 블록)
# ==========================
from nwwpkg.ui.network_viz import draw_network_pyvis


def _primary_frame(fs: list[dict]) -> str:
    if not fs:
        return "General"
    # score 가장 큰 프레임
    best = sorted(fs, key=lambda f: f.get("score", 0), reverse=True)[0]
    return best.get("frame", "General")

# def page_blocks(bundle_id: str):
#     st.header("🧩 Blocks (EDS) – 기사 ↔ 블록 매핑")

#     df = load_jsonl(f"data/{bundle_id}/fusion.jsonl")
#     if df.empty or "frames" not in df.columns:
#         st.warning("먼저 Fusion 단계를 실행하세요.")
#         return

#     df["block"] = df["frames"].apply(_primary_frame)

#     # ✅ 핵심 컬럼 보존 보강
#     keep_cols = [c for c in ["url","date","normalized","frames","indicator_score","dbn_score","fused_score"] if c in df.columns]
#     out = df[keep_cols + ["block"]].copy()

#     st.subheader("기사 ↔ 블록 매핑")
#     st.dataframe(out.head(), width='stretch')

#     save_jsonl(out, f"data/{bundle_id}/blocks.jsonl")
#     st.success("✅ Blocks 저장 완료")

# def page_block():
#     st.header("🧩 Block 페이지 - 블록 분석 & 위기 탐지")

#     blocks = st.session_state.get("blocks", [])
#     if not blocks:
#         st.warning("먼저 EventBlock 페이지에서 블록을 생성하세요.")
#         return

#     # 블록 선택
#     block_id = st.selectbox("분석할 블록 선택", [b["block_id"] for b in blocks])

#     # 블록 분석
#     if st.button("블록 분석"):
#         block = next(b for b in blocks if b["block_id"] == block_id)
#         report = block_analyzer.run(block)
#         st.session_state["block_report"] = report
#         st.success("블록 분석 완료")
#         st.json(report)

#         # # 네트워크 그래프 시각화
#         # st.subheader("네트워크 그래프")
#         # fig = draw_network(report["network"], block["events"])
#         # st.plotly_chart(fig, use_container_width=True)

#         # pyvis 네트워크 시각화
#         st.subheader("네트워크 그래프 (위험도 색상 표시)")
#         html = draw_network_pyvis(block, report)
#         components.html(html, height=500, scrolling=True)

#     # 위기 탐지
#     if st.button("위기 탐지"):
#         crisis = crisis_detector.run(blocks)
#         st.metric("Crisis Score", crisis["score"])
#         st.write(f"상태: {crisis['status']}")

#     # 시나리오 생성
#     if st.button("시나리오 생성"):
#         scenario = scenario_generator.run(block_id)
#         st.success("시나리오 생성 완료")
#         st.json(scenario)


# from nwwpkg.ui.network_viz import draw_network_pyvis

# def page_block():
#     st.header("🧩 Block 페이지 - 블록 분석 & 위기 탐지")

#     # 세션에서 블록 불러오기
#     blocks = st.session_state.get("blocks", [])
#     if not blocks:
#         st.warning("먼저 EventBlock 페이지에서 블록을 생성하세요.")
#         return

#     # 블록 선택
#     block_id = st.selectbox("분석할 블록 선택", [b["block_id"] for b in blocks])
#     block = next(b for b in blocks if b["block_id"] == block_id)

#     # 블록 분석
#     if st.button("블록 분석 실행"):
#         report = block_analyzer.run(block)
#         st.session_state["block_report"] = report
#         st.success("✅ 블록 분석 완료")

#         # 인물 분석 결과 표시
#         st.subheader("👤 인물 분석")
#         st.dataframe(pd.DataFrame(report["actors"]))

#         # 네트워크 그래프 표시 (pyvis 인터랙티브)
#         st.subheader("🌐 네트워크 그래프")
#         html = draw_network_pyvis(block, report)
#         components.html(html, height=500, scrolling=True)

#         # 프레임 변화 표시
#         st.subheader("🌀 프레임 변화")
#         frame_report = report["frame_report"]
#         st.json(frame_report)
#         if frame_report["frames_by_time"]:
#             df_frames = pd.DataFrame(frame_report["frames_by_time"])
#             st.line_chart(df_frames.set_index("time"))

#         # 위험 점수 표시
#         st.subheader("⚠️ 위험 점수")
#         st.metric("Crisis Risk Score", report["risk_score"])

#         # 요약 표시
#         st.info(report["summary"])

#     # 위기 탐지 실행
#     if st.button("위기 탐지 실행"):
#         crisis = crisis_detector.run(blocks)
#         st.success("🚨 위기 탐지 완료")
#         st.metric("Crisis Score", crisis["score"])
#         st.write(f"상태: {crisis['status']}")

#     # 시나리오 생성 실행
#     if st.button("시나리오 생성 실행"):
#         scenario = scenario_generator.run(block_id)
#         st.success("📑 시나리오 생성 완료")
#         st.json(scenario)

# import streamlit as st
# import streamlit.components.v1 as components
# import pandas as pd
# from pathlib import Path

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

from nwwpkg.utils.io import load_jsonl
from nwwpkg.eventblock.block_accumulator import accumulate_blocks


def page_block(bundle_id: str):
    st.header("🧩 EventBlock – 사건/블록 생성 및 확인")

    events_file = f"data/bundles/{bundle_id}/events.jsonl"
    blocks_file = f"data/bundles/{bundle_id}/blocks.jsonl"

    # -----------------------
    # 데이터 로드
    # -----------------------
    try:
        events_df = pd.DataFrame(load_jsonl(events_file))
    except Exception:
        st.warning("⚠️ 먼저 'Normalize & Entity' 단계를 완료해야 합니다. (events.jsonl 필요)")
        return

    try:
        events_df, blocks = accumulate_blocks(events_file, blocks_file, sim_thr=0.35)

        # blocks가 list라면 DataFrame으로 변환
        if isinstance(blocks, list):
            blocks_df = pd.DataFrame(blocks)
        else:
            blocks_df = blocks

    except Exception as e:
        st.error(f"❌ 블록 생성 오류: {e}")
        return

    # -----------------------
    # 사건 분류 결과
    # -----------------------
    st.subheader("📂 사건 분류 결과")
    if not events_df.empty:
        st.success(f"사건 {len(events_df)}건 분류 완료")

        base_cols = ["id", "time"]
        optional_cols = [c for c in ["normalized", "block"] if c in events_df.columns]
        show_cols = base_cols + optional_cols

        st.dataframe(events_df[show_cols].head(20), use_container_width=True)
    else:
        st.warning("사건 데이터가 없습니다.")
        return

    # -----------------------
    # 블록 적재 결과
    # -----------------------
    st.subheader("📦 블록 적재 결과")
    if blocks_df.empty:
        st.warning("⚠️ blocks.jsonl에 데이터가 없습니다.")
        return

    if "block_id" not in blocks_df.columns:
        st.warning("⚠️ blocks.jsonl에 block_id 컬럼이 없습니다. 블록 재생성이 필요합니다.")
        return

    st.success(f"블록 {len(blocks_df)}건 적재 완료")

    block_cols = [c for c in ["block_id", "block_label", "num_events", "frames"] if c in blocks_df.columns]
    st.dataframe(blocks_df[block_cols], use_container_width=True)

    # -----------------------
    # 블록 네트워크 그래프
    # -----------------------
    st.subheader("🌐 블록 간 관계 (네트워크 그래프)")

    try:
        G = nx.Graph()
        for _, row in blocks_df.iterrows():
            G.add_node(row["block_id"], label=row.get("block_label", "Unknown"))

        for i in range(len(blocks_df)):
            for j in range(i + 1, len(blocks_df)):
                frames_i = set(blocks_df.iloc[i].get("frames", []))
                frames_j = set(blocks_df.iloc[j].get("frames", []))
                if frames_i & frames_j:
                    G.add_edge(blocks_df.iloc[i]["block_id"], blocks_df.iloc[j]["block_id"])

        pos = nx.spring_layout(G, seed=42)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#888"),
                                hoverinfo="none", mode="lines")

        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}: {G.nodes[node]['label']}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=node_text,
            textposition="top center",
            marker=dict(size=15, color="skyblue", line_width=2)
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(showlegend=False, hovermode="closest",
                                         margin=dict(b=0, l=0, r=0, t=0)))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"네트워크 그래프 생성 실패: {e}")

    # -----------------------
    # 원본 events 미리보기
    # -----------------------
    with st.expander("🔎 원본 events.jsonl 데이터 미리보기"):
        st.dataframe(events_df.head(10), use_container_width=True)




# ==========================
# 8) Scenarios (매칭 + 예측)
# ==========================
from nwwpkg.scenario.page_scenarios import page_scenarios
# from nwwpkg.scenario.llm_utils import generate_scenario_with_llm
# def page_scenarios(bundle_id: str):
#     st.header("📑 Scenarios – 매칭/예측")

#     df = load_jsonl(f"data/{bundle_id}/blocks.jsonl")
#     if df.empty or "normalized" not in df.columns:
#         st.warning("먼저 Blocks 단계를 실행하세요.")
#         return

#     # 실제 임베딩 → 시나리오 매칭 & 예측
#     matched_list, predicted_list = [], []
#     for _, row in df.iterrows():
#         sents = row.get("sentences", [])
#         if not sents:
#             sents = tokenizer.split_sentences(row.get("normalized", ""))
#         vecs = embedder.embed(sents)
#         matched = scenario_matcher.match(vecs, top_k=3)
#         pred = scenario_predictor.generate(row.get("normalized", ""))
#         matched_list.append(matched)
#         predicted_list.append(pred)

#     df["scenario_matched"] = matched_list
#     df["scenario_predicted"] = predicted_list

#     # 시나리오별 평균 위험도
#     df["top_sim"] = df["scenario_matched"].apply(lambda xs: xs[0]["similarity"] if xs else 0.0)
#     top = df[["block", "fused_score", "top_sim"]].groupby("block", observed=True).mean().reset_index()
#     st.subheader("블록별 평균 점수/유사도")
#     st.dataframe(top, width='stretch')

#     save_jsonl(df, f"data/{bundle_id}/scenarios.jsonl")
#     st.success("✅ Scenarios 저장 완료")

# ==========================
# 9) Alerts (최종 경보)
# ==========================
def page_alerts(bundle_id: str):
    st.header("🚨 Alerts – 경보 발생")

    df = load_jsonl(f"data/{bundle_id}/scenarios.jsonl")
    if df.empty or "fused_score" not in df.columns:
        st.warning("먼저 Scenarios 단계를 실행하세요.")
        return

    # ✅ LLM Judge 설명 생성(점수+프레임)
    def _scores_of(r):
        return {
            "indicator": float(r.get("indicator_score", 0.0)),
            "dbn":       float(r.get("dbn_score", 0.0)),
            "fused":     float(r.get("fused_score", 0.0)),
        }

    df["explanation"] = df.apply(lambda r: llm_judge.explain(_scores_of(r), r.get("frames", [])), axis=1)

    # ✅ 최종 경보 결정(기존 유지)
    df["decision"] = df["fused_score"].apply(alert_decider.decide)

    def to_level(dec: str) -> str:
        if "High" in dec: return "High"
        if "Medium" in dec: return "Medium"
        return "Low"
    df["alert_level"] = df["decision"].apply(to_level)

    st.subheader("경보 결과")
    st.dataframe(df[["normalized","fused_score","decision","alert_level","explanation"]].head(), width='stretch')

    fig = px.histogram(df, x="alert_level", title="경보 분포")
    st.plotly_chart(fig, width='stretch')

    save_jsonl(df, f"data/{bundle_id}/alerts.jsonl")
    st.success("✅ Alerts 저장 완료")


# ==========================
# 10) Event Blocks (간이 클러스터링)
# ==========================
# 샘플 데이터 경로
DATA_PATH = Path("data/sample_articles.jsonl")


def page_eventblocks(bundle_id: str):
    st.header("📦 Event Blocks – 클러스터링(간이)")

    df = load_jsonl(f"data/{bundle_id}/alerts.jsonl")
    if df.empty:
        st.warning("먼저 Alerts 단계를 실행하세요.")
        return

    # 간이 클러스터
    df["cluster"] = df.index % 3
    pivot = df.groupby("cluster", observed=True)["fused_score"].mean().reset_index()

    st.subheader("클러스터별 평균 위험도")
    st.dataframe(pivot, width='stretch')

    fig = px.imshow([pivot["fused_score"].tolist()],
                    labels=dict(x="Cluster", y="Risk", color="Score"),
                    title="Event Block 위험도 히트맵")
    st.plotly_chart(fig, width='stretch')

    save_jsonl(df, f"data/{bundle_id}/eventblocks.jsonl")
    st.success("✅ Event Blocks 저장 완료")

from nwwpkg.eventblock.block_accumulator import accumulate_blocks


def page_eventblock(bundle_id: str):
    st.header("🧩 EventBlock – 사건/블록 생성 및 확인")

    root = Path(f"data/{bundle_id}")
    events_file = root / "events.jsonl"
    blocks_file = root / "blocks.jsonl"

    # ✅ 사건 데이터 확인
    events_df = load_jsonl(events_file)
    if events_df.empty:
        st.warning("⚠️ 먼저 'Normalize & Entity' 단계를 완료해야 합니다. (events.jsonl 필요)")
        return

    # ✅ 블록 생성 버튼
    if st.button("🔄 블록 재생성 (유사도 기반)"):
        clustered_df, blocks = accumulate_blocks(events_file, blocks_file, sim_thr=0.35)
        st.session_state["blocks_df"] = clustered_df
        st.session_state["blocks"] = blocks
        st.success(f"✅ {len(blocks)}개 블록 생성 완료")

    # ✅ 세션 상태에서 불러오기
    blocks = st.session_state.get("blocks", [])
    blocks_df = st.session_state.get("blocks_df", pd.DataFrame())

    if not blocks:
        st.info("아직 블록이 없습니다. '블록 재생성'을 눌러주세요.")
        return

    # ✅ 블록 개요 테이블
    st.subheader("📊 블록 개요")
    summary_rows = [
        {"block_id": b["block_id"], "num_events": b["num_events"]}
        for b in blocks
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    # ✅ 블록 선택
    block_ids = [b["block_id"] for b in blocks]
    block_id = st.selectbox("🔍 블록 선택", block_ids)

    block = next((b for b in blocks if b["block_id"] == block_id), None)
    if block:
        st.markdown(f"### 📦 블록 `{block_id}` 상세")

        # 블록 내 사건 표
        block_events = pd.DataFrame(block["events"])
        st.dataframe(block_events, use_container_width=True)

        # JSON 미리보기
        st.subheader("📑 JSON 미리보기")
        st.json(block, expanded=False)

    # ✅ 다운로드
    st.subheader("📥 블록 데이터 다운로드")
    json_buffer = "\n".join(json.dumps(b, ensure_ascii=False) for b in blocks)
    st.download_button(
        "⬇️ blocks.jsonl 다운로드",
        data=json_buffer,
        file_name="blocks.jsonl",
        mime="application/json",
    )
# ==========================
# 11) Ledger (파일 로그)
# ==========================
# def render_blocks(root: Path):
#     st.header("🧱 Blocks")
#     fn = _load_callable("nwwpkg.ui.pages.blocks", "page_blocks")
#     if fn is None:
#         _fallback_page("Blocks (placeholder)", root,
#                        "nwwpkg/ui/pages/blocks.py 가 없거나 import 에 실패했습니다.")
#         return
#     fn(root)

# ==========================
# Overview (대시보드)
# ==========================
def page_dashboard(bundle_id: str):
    """
    🌍 NWW Dashboard
    - 랜딩/요약 페이지
    - KPI → 확장 버튼(expander) → 세부 모듈 호출
    """

    st.title("🌍 NWW Dashboard (Critical Overview)")

    bundle_path = _bundle_path(bundle_id)

    # --- Pipeline & QA ---
    with st.expander("📦 Pipeline DAG (처리 단계 현황)", expanded=True):
        render_pipeline_dag(bundle_path)

    with st.expander("🧪 Stage QA (품질 점검)", expanded=False):
        render_stage_qa(bundle_path)

    # --- 데이터 로드 ---
    df_ingest  = load_jsonl(f"data/{bundle_id}/ingest.jsonl")
    df_ana     = load_jsonl(f"data/{bundle_id}/analyze.jsonl")
    df_blocks  = load_jsonl(f"data/{bundle_id}/blocks.jsonl")
    df_scen    = load_jsonl(f"data/{bundle_id}/scenarios.jsonl")
    df_alerts  = load_jsonl(f"data/{bundle_id}/alerts.jsonl")

    # ✅ 랜딩에서 바로 보이는 인라인 시계열
    render_alerts_timeseries_inline(df_alerts, bundle_id)
    # --- Alerts ---
    with st.expander("📡 Alerts (위험 경보) · 상세", expanded=False):
        render_alert_dashboard(df_alerts, bundle_id, bundle_path)  # ✅ 상세(컨트롤 없음)

    # --- Choropleth (위치 분석) ---
    with st.expander("🗺️ Location (경보 지역 분석)", expanded=False):
        render_choropleth(df_alerts)

    # --- Risk Blocks ---
    with st.expander("🔥 Risk Blocks (위험 블록)", expanded=False):
        render_risk_blocks(df_blocks, bundle_id)

    # --- Scenarios ---
    with st.expander("🧭 Scenarios (위험 시나리오)", expanded=False):
        render_scenarios(df_scen, bundle_id)

    # --- Entities ---
    with st.expander("🏷️ Entities (주요 인물/조직)", expanded=False):
        render_entities(df_ana)

    # --- Sources ---
    with st.expander("📰 Sources (언론사/출처 분포)", expanded=False):
        render_sources(df_ingest)

    # --- Backlog 실행 버튼 ---
    st.divider()
    if st.button("⚡ 백로그 일괄 처리 (ingest → alerts)"):
        from nwwpkg.utils.runner import _quick_pipeline, _safe_rerun
        ok = _quick_pipeline(bundle_id)
        if ok:
            st.success("✅ 파이프라인 실행 완료: ingest → alerts 생성")
        else:
            st.warning("⚠️ ingest 데이터가 없습니다.")
        _safe_rerun()


# ==========================
# Main
# ==========================
def main():
    st.set_page_config(page_title="NWW Early Warning Dashboard", layout="wide")

    st.sidebar.header("⚙️ 설정")
    bundle_id = st.sidebar.text_input("Bundle ID", DEFAULT_BUNDLE)
    page = st.sidebar.radio("📌 단계 이동", [
        "Landing", "Ingest", "Normalize", "Analyze", "Gate", "Scoring",
        "Risk", "Alerts","EventBlocks","Fusion", "Blocks", "Scenarios", "Ledger"
    ])
    
    # === [SIDEBAR] GeoJSON 입력 & 진단 ===
    import io, json, os

    st.sidebar.markdown("### 🗺️ 시군구 GeoJSON")
    geojson_file = st.sidebar.file_uploader("파일 업로드(.geojson/.json)", type=["geojson", "json"])
    geojson_path = st.sidebar.text_input("또는 파일 경로 입력", value=st.session_state.get("siggeo_path", ""))

    # 우선순위: 업로드 > 경로
    geo_obj = None
    diag = {"source": None, "exists": None, "size": None, "features": None, "code_key": None, "name_key": None, "error": None}

    try:
        if geojson_file is not None:
            raw = geojson_file.getvalue().decode("utf-8", "ignore")
            geo_obj = json.loads(raw)
            st.session_state["siggeo_obj"] = geo_obj
            st.session_state["siggeo_path"] = ""  # 경로 비활성
            diag["source"] = "uploaded"
        elif geojson_path.strip():
            path = os.path.abspath(os.path.expanduser(geojson_path.strip()))
            st.session_state["siggeo_path"] = path
            diag["source"] = f"path: {path}"
            diag["exists"] = os.path.exists(path)
            if diag["exists"]:
                diag["size"] = os.path.getsize(path)
                with open(path, "r", encoding="utf-8-sig") as f:
                    geo_obj = json.load(f)
                st.session_state["siggeo_obj"] = geo_obj
            else:
                diag["error"] = "파일이 존재하지 않습니다."
        else:
            # 이전 세션의 객체 재사용
            geo_obj = st.session_state.get("siggeo_obj")
            if geo_obj is not None:
                diag["source"] = "session"
    except Exception as e:
        diag["error"] = f"{type(e).__name__}: {e}"

    # 구조 키 자동 탐지
    def _detect_sig_props(geo: dict):
        if not geo or "features" not in geo or not geo["features"]:
            return None, None
        props = geo["features"][0].get("properties", {})
        candidates_code = ["SIG_CD", "adm_cd", "ADM_CD"]
        candidates_name = ["SIG_KOR_NM", "sig_kor_nm", "ADM_NM", "adm_nm", "EMD_KOR_NM"]
        code_key = next((k for k in candidates_code if k in props), None)
        name_key = next((k for k in candidates_name if k in props), None)
        return code_key, name_key

    if geo_obj:
        diag["features"] = len(geo_obj.get("features", []))
        diag["code_key"], diag["name_key"] = _detect_sig_props(geo_obj)

    with st.sidebar.expander("GeoJSON 진단", expanded=True):
        st.write(diag)
        if geo_obj and (not diag["code_key"] or not diag["name_key"]):
            st.warning("⚠️ properties에 시군구 코드/이름 키를 찾지 못했습니다. 예: SIG_CD, SIG_KOR_NM")
        elif not geo_obj:
            st.info("GeoJSON을 업로드하거나 경로를 입력하세요.")


    if page == "Landing":
        page_dashboard(bundle_id)
        render_risk_snapshot(ROOT)
    elif page == "Ingest":
        page_ingest(bundle_id)
    elif page == "Normalize":
        page_normalize(bundle_id)
    elif page == "Analyze":
        page_analyze(bundle_id)
        render_keywords_and_sentiment(ROOT, topn=20)
        render_frames_eval(ROOT)
    elif page == "Gate":
        render_gate_indicator(ROOT) 
    elif page == "Scoring":
        render_scoring(ROOT)
    elif page == "Risk":
        render_risk_page(ROOT)
    elif page == "Alerts":
        render_alerts_config(ROOT)
    elif page == "EventBlocks":
        render_eventblocks(ROOT)
        page_eventblock(bundle_id)
    elif page == "Fusion":
        _call_page("fusion", "page_fusion", ROOT)
    elif page == "Blocks":
        _call_page("blocks", "page_blocks", ROOT)
        page_block(bundle_id)
    elif page == "Scenarios":
        # _call_page("scenarios", "page_scenarios", ROOT)
        page_scenarios(ROOT)  
    elif page == "Ledger":
        _call_page("ledger", "page_ledger", ROOT)

if __name__ == "__main__":
    main()
