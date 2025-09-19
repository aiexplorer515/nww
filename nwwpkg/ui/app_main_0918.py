# app_main.py (실제 호출 버전: Ingest → Normalize → Analyze → Gate → Scoring → Fusion → Blocks → Scenarios → Alerts)
from __future__ import annotations
import os
import io
import re
import json
from datetime import datetime
from collections import Counter

import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import matplotlib.pyplot as plt
from importlib.util import spec_from_file_location, module_from_spec
import inspect
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
# from nwwpkg.ui.components.page_wrappers import (
#     render_fusion, render_blocks, render_scenarios, render_ledger
# )
from nwwpkg.ui.components.gate_indicator import render_gate_indicator
from nwwpkg.ui.components.scoring_page import render_scoring
from nwwpkg.ui.components.alerts_page import render_alerts
from nwwpkg.ui.components.eventblocks_page import render_eventblocks
from nwwpkg.ui.components.risk_snapshot import render_risk_snapshot
from nwwpkg.ui.components.risk_page import render_risk_page
from nwwpkg.ui.components.frames_eval import render_frames_eval
from nwwpkg.ui.components.analytics_basic import render_keywords_and_sentiment
from nwwpkg.ui.page_actor_ranking import page_analyze
# from nwwpkg.ui.pages.fusion import page_fusion
# from nwwpkg.ui.pages.blocks import page_blocks
# from nwwpkg.ui.pages.scenarios import page_scenarios
# from nwwpkg.ui.pages.ledger import page_ledger
# from nwwpkg.ui.components.frames_eval import render_frames_eval
# from nwwpkg.ui.components.analytics_basic import render_keywords_and_sentiment
# from nwwpkg.ui.components.risk_snapshot import render_risk_snapshot
# from nwwpkg.ui.components.risk_page import render_risk_page
# from nwwpkg.ui.components.gate_indicator import render_gate_indicator
# from nwwpkg.ui.components.scoring_page import render_scoring
# from nwwpkg.ui.components.alerts_page import render_alerts
# from nwwpkg.ui.components.eventblocks_page import render_eventblocks


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

def ensure_text_column(df):
    """데이터프레임에 df['text']를 보장한다. 있으면 그대로, 없으면 후보 컬럼을 복사/생성."""
    import pandas as pd
    # 1) 후보 중 있는 것 선택
    for c in TEXT_CANDIDATES:
        if c in df.columns:
            if c != "text":
                df["text"] = df[c].astype(str)
            return "text"
    # 2) title+body 결합 폴백
    if "title" in df.columns and "body" in df.columns:
        df["text"] = df["title"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
        return "text"
    # 3) 문자열(객체) dtype 첫 컬럼 폴백
    obj_cols = [c for c in df.columns if df[c].dtype == "O"]
    if obj_cols:
        df["text"] = df[obj_cols[0]].astype(str)
        return "text"
    # 4) 실패
    return None
# --- end add ---

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


def _count_lines(p: Path) -> int:
    try:
        with p.open(encoding="utf-8-sig") as f:
            for i, _ in enumerate(f, 1): pass
        return i
    except Exception:
        return 0

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
def page_normalize(bundle_id):
    st.header("🔤 Normalize – 텍스트 정규화")

    df = load_jsonl(f"data/{bundle_id}/ingest.jsonl")
    if df.empty:
        st.info("⚠️ 먼저 Ingest 단계에서 기사를 입력하세요.")
        return
    # ensure df['text']
    col = ensure_text_column(df)
    if col is None:
        st.warning("본문 텍스트 컬럼을 찾을 수 없습니다. (예: text/clean_text/body)")
        return


    # 실제 전처리 호출
    df["normalized"] = df["text"].astype(str).apply(cleaner.normalize)
    df["sentences"] = df["normalized"].apply(tokenizer.split_sentences)

    # 비교 테이블
    st.subheader("원문 vs 정규화 텍스트")
    st.dataframe(df[["text", "normalized"]].head(10), width='stretch')

    # 문장 수 분포
    df["sent_count"] = df["sentences"].apply(len)
    fig = px.histogram(df, x="sent_count", nbins=20, title="문장 수 분포")
    st.plotly_chart(fig, width='stretch')

    # 다운로드
    st.subheader("📥 정규화 데이터 다운로드")
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    st.download_button("⬇️ CSV 다운로드", data=csv_buffer.getvalue(),
                       file_name=f"{bundle_id}_normalized.csv", mime="text/csv")

    jsonl_buffer = io.StringIO()
    for _, row in df.iterrows():
        jsonl_buffer.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    st.download_button("⬇️ JSONL 다운로드", data=jsonl_buffer.getvalue(),
                       file_name=f"{bundle_id}_normalized.jsonl", mime="application/json")

    save_jsonl(df, f"data/{bundle_id}/normalize.jsonl")
    st.success("✅ 정규화 데이터 저장 완료")

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
    # kw_df = pd.DataFrame(Counter(toks).most_common(30), columns=["word", "freq"])
    # st.dataframe(kw_df.head(20), width='stretch')
    
    # 키워드 파일 읽기 (오류 처리 포함)
    # >>> START: keywords real
    # kwp = _bundle_path(bundle_id) / "keywords.jsonl"
    # df_kw = read_jsonl_safe(kwp)
    # st.subheader("🗂️ 키워드 Top-N (한글)")
    # if not df_kw.empty:
    #     df_kw["freq"] = pd.to_numeric(df_kw.get("freq", 0), errors="coerce").fillna(0).astype(int)
    #     df_kw = df_kw.sort_values("freq", ascending=False).head(50)
    #     st.dataframe(df_kw, use_container_width=True, height=420)
    #     st.caption(f"소스: {kwp}")
    # else:
    #     st.info("키워드 파일이 없습니다. 먼저 키워드 추출을 실행하세요.")
    # <<< END: keywords real



    # ------------------------
    # 감정 분포 (예시 데이터 유지)
    # ------------------------
    # >>> START: sentiment real
    # sumf = _bundle_path(bundle_id) / "sentiment.summary.json"
    # data = load_json_safe(sumf, default=None)

    # st.subheader("🧪 감정 분포 (실제)")
    # if data and "ratio" in data:
    #     import plotly.express as px
    #     ratio = data["ratio"]             # {'positive': 40.0, ...}
    #     labels = {"positive":"긍정","negative":"부정","mixed":"중립","neutral":"기타"}
    #     names  = [labels.get(k,k) for k in ratio.keys()]
    #     vals   = list(ratio.values())
    #     fig = px.pie(values=vals, names=names)
    #     fig.update_traces(textinfo="percent+label")
    #     st.plotly_chart(fig, use_container_width=True)
    #     st.caption(f"소스: {sumf}")
    # else:
    #     st.info("sentiment.summary.json이 없습니다. (생성 스크립트 실행 필요)")
    # <<< END: sentiment real


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
def _primary_frame(fs: list[dict]) -> str:
    if not fs:
        return "General"
    # score 가장 큰 프레임
    best = sorted(fs, key=lambda f: f.get("score", 0), reverse=True)[0]
    return best.get("frame", "General")

def page_blocks(bundle_id: str):
    st.header("🧩 Blocks (EDS) – 기사 ↔ 블록 매핑")

    df = load_jsonl(f"data/{bundle_id}/fusion.jsonl")
    if df.empty or "frames" not in df.columns:
        st.warning("먼저 Fusion 단계를 실행하세요.")
        return

    df["block"] = df["frames"].apply(_primary_frame)

    # ✅ 핵심 컬럼 보존 보강
    keep_cols = [c for c in ["url","date","normalized","frames","indicator_score","dbn_score","fused_score"] if c in df.columns]
    out = df[keep_cols + ["block"]].copy()

    st.subheader("기사 ↔ 블록 매핑")
    st.dataframe(out.head(), width='stretch')

    save_jsonl(out, f"data/{bundle_id}/blocks.jsonl")
    st.success("✅ Blocks 저장 완료")

# ==========================
# 8) Scenarios (매칭 + 예측)
# ==========================
def page_scenarios(bundle_id: str):
    st.header("📑 Scenarios – 매칭/예측")

    df = load_jsonl(f"data/{bundle_id}/blocks.jsonl")
    if df.empty or "normalized" not in df.columns:
        st.warning("먼저 Blocks 단계를 실행하세요.")
        return

    # 실제 임베딩 → 시나리오 매칭 & 예측
    matched_list, predicted_list = [], []
    for _, row in df.iterrows():
        sents = row.get("sentences", [])
        if not sents:
            sents = tokenizer.split_sentences(row.get("normalized", ""))
        vecs = embedder.embed(sents)
        matched = scenario_matcher.match(vecs, top_k=3)
        pred = scenario_predictor.generate(row.get("normalized", ""))
        matched_list.append(matched)
        predicted_list.append(pred)

    df["scenario_matched"] = matched_list
    df["scenario_predicted"] = predicted_list

    # 시나리오별 평균 위험도
    df["top_sim"] = df["scenario_matched"].apply(lambda xs: xs[0]["similarity"] if xs else 0.0)
    top = df[["block", "fused_score", "top_sim"]].groupby("block", observed=True).mean().reset_index()
    st.subheader("블록별 평균 점수/유사도")
    st.dataframe(top, width='stretch')

    save_jsonl(df, f"data/{bundle_id}/scenarios.jsonl")
    st.success("✅ Scenarios 저장 완료")

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

# ==========================
# 11) Ledger (파일 로그)
# ==========================
def page_ledger(bundle_id: str):
    st.header("📜 Ledger – 단계별 로그")

    files = [
        "ingest.jsonl","normalize.jsonl","analyze.jsonl","gate.jsonl",
        "scoring.jsonl","fusion.jsonl","blocks.jsonl","scenarios.jsonl",
        "alerts.jsonl","eventblocks.jsonl"
    ]
    logs = []
    for f in files:
        path = f"data/{bundle_id}/{f}"
        if os.path.exists(path):
            logs.append({"file": f, "size": os.path.getsize(path)})
    st.dataframe(pd.DataFrame(logs), width='stretch')

    col1, col2 = st.columns(2)
    if col1.button("📂 alerts.jsonl 열기(viewer)"):
        try:
            st.dataframe(viewer.load(bundle_id, "alerts.jsonl"), width='stretch')
        except Exception as e:
            st.warning(f"viewer.load 실패: {e}")

    if col2.button("📥 ledger.jsonl 저장"):
        ledger = {"bundle": bundle_id, "timestamp": datetime.now().isoformat(), "steps": files}
        save_jsonl(pd.DataFrame([ledger]), f"data/{bundle_id}/ledger.jsonl")
        st.success("ledger.jsonl 저장 완료")



# ==========================
# Overview (대시보드)
# ==========================
def page_overview(bundle_id: str):
    st.title("🌍 Crisis Overview")
    # === [ADD] DAG badges (counts) ===
    bundle_path = _bundle_path(bundle_id)

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

    st.markdown("### Pipeline DAG")
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
    # === [/ADD] ===

    # === [ADD] Stage QA tiles ===
    st.markdown("### Stage QA")
    qc_path = bundle_path / "prep.qc.json"
    if qc_path.exists():
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
    else:
        # Fallback: clean.jsonl에서 간이 계산
        import statistics as stats
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
    # === [/ADD] ===





    # --- 데이터 로드 ---
    df_ingest  = load_jsonl(f"data/{bundle_id}/ingest.jsonl")
    df_norm    = load_jsonl(f"data/{bundle_id}/normalize.jsonl")
    df_ana     = load_jsonl(f"data/{bundle_id}/analyze.jsonl")
    df_gate    = load_jsonl(f"data/{bundle_id}/gate.jsonl")
    df_score   = load_jsonl(f"data/{bundle_id}/scoring.jsonl")
    df_fuse    = load_jsonl(f"data/{bundle_id}/fusion.jsonl")
    df_blocks  = load_jsonl(f"data/{bundle_id}/blocks.jsonl")
    df_scen    = load_jsonl(f"data/{bundle_id}/scenarios.jsonl")  # ✅ 복구
    df_alerts  = load_jsonl(f"data/{bundle_id}/alerts.jsonl")

    # --- KPI: 수집 vs 경보 & 처리율 ---
    n_ingest = 0 if df_ingest.empty else len(df_ingest)
    n_alerts = 0 if df_alerts.empty else len(df_alerts)
    ratio    = (n_alerts / n_ingest) if n_ingest else 0.0

    k1, k2, k3 = st.columns(3)
    k1.metric("수집된 기사 수 (ingest)", n_ingest)
    k2.metric("경보 산출 기사 수 (alerts)", n_alerts)
    k3.metric("처리율", f"{ratio*100:.0f}%")
    st.progress(min(1.0, ratio))

        # --- 📡 시계열 경보(Alerts) ---
    st.subheader("📡 시계열 경보 (Alerts)")
    # Alerts 섹션에서 df_alerts 로드 직후
    basis = st.selectbox("시간 기준 (Time basis)", ["경보 생성시각", "원문 시각"], index=0)

        # === [REPLACE] Alerts rendering ===
    ap = bundle_path / "alerts.jsonl"
    if ap.exists() and _count_lines(ap) > 0:
        from io import StringIO
        with open(ap, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        if content.strip():
            df_alerts = pd.read_json(StringIO(content), lines=True)
        else:
            df_alerts = pd.DataFrame()
        # 컬럼 매핑(UI 호환)
        if "date" not in df_alerts.columns and "t_on" in df_alerts.columns:
            df_alerts["date"] = df_alerts["t_on"]
        if "alert_level" not in df_alerts.columns and "level" in df_alerts.columns:
            df_alerts["alert_level"] = df_alerts["level"]

        # 시간 정리(KST naive)
        # 원문 시각 후보 매핑(있을 때만)
        if "src_time" not in df_alerts:
            for c in ["published_at", "doc_time", "event_time"]:
                if c in df_alerts.columns:
                    df_alerts["src_time"] = df_alerts[c]
                    break

        # 선택한 기준으로 집계 컬럼 결정
        time_col = "date" if basis == "경보 생성시각" else "src_time"

        if time_col in df_alerts.columns:
            df_alerts[time_col] = _to_local_naive(df_alerts[time_col], output_tz="Asia/Seoul")
            # 내부 timeseries 함수가 'date'를 기대하면 리네임해서 넘기기
            ts_long = _alert_timeseries(df_alerts.rename(columns={time_col: "date"}), freq="D")
        else:
            st.info("선택한 시간 기준에 해당하는 컬럼이 없습니다. (date/t_on 또는 src_time/published_at)")

        # 카드 5개만
        for _, r in df_alerts.head(5).iterrows():
            lvl = str(r.get("alert_level", "watch")).upper()
            sc  = float(r.get("fused_score", 0.0))
            rsn = r.get("reason") or "(no reason)"
            st.markdown(f"**[{lvl}]** {rsn} · score={sc:.2f} · {r.get('date','')}")
    else:
        st.info("Alerts 데이터가 없습니다. (⚡ scores→alerts 생성 후 Rerun)")
    # === [/REPLACE] ===


    if not df_alerts.empty and {"date","alert_level"}.issubset(df_alerts.columns):
        freq_label = st.selectbox("집계 주기", ["일", "주", "월"], index=0, key="alert_ts_freq")
        freq_map = {"일":"D","주":"W","월":"M"}
        freq = freq_map[freq_label]

        ts_long = _alert_timeseries(df_alerts, freq=freq)
        ts_wide = _to_wide(ts_long)
        ts_wide = ts_wide.copy()
        ts_wide["Total"] = ts_wide["High"] + ts_wide["Medium"] + ts_wide["Low"]

        # 스무딩 옵션(rolling)
        smooth = st.checkbox("7-창 스무딩(rolling mean)", value=(freq=="D"), key="alert_ts_smooth")
        if smooth and not ts_wide.empty:
            ts_wide_sm = ts_wide.rolling(window=7, min_periods=1).mean()
        else:
            ts_wide_sm = ts_wide

        # 탭: 스택면적 / 레벨선 / 누적 / 표
        tabs_ts = st.tabs(["🟣 스택 면적", "📈 레벨별 선", "➕ 누적 합", "📄 표"])

        with tabs_ts[0]:
            # long-form이 area에 적합
            fig_area = px.area(
                ts_long, x="date", y="count", color="alert_level",
                title=f"경보 수 스택 면적 ({freq_label} 단위)",
                color_discrete_sequence=next_palette()
            )
            st.plotly_chart(fig_area, width='stretch')

        with tabs_ts[1]:
            df_plot = reset_index_as_date(ts_wide_sm)
            fig_line = px.line(
                df_plot, x="date", y=["High","Medium","Low"],
                title=f"경보 레벨별 추세 ({freq_label} 단위){' - 7창 스무딩' if smooth else ''}",
                color_discrete_sequence=next_palette()
            )
            st.plotly_chart(fig_line, width='stretch')

        with tabs_ts[2]:
            df_cum  = reset_index_as_date(ts_wide.cumsum())
            fig_cum = px.line(
                df_cum, x="date", y=["High","Medium","Low","Total"],
                title=f"경보 누적 합 ({freq_label} 단위)",
                color_discrete_sequence=next_palette()
            )
            st.plotly_chart(fig_cum, width='stretch')

        with tabs_ts[3]:
            # 표는 원시 wide + Total 제공
            st.dataframe(ts_wide.reset_index(), width='stretch')
    else:
        st.info("시계열 경보를 위해서는 df_alerts에 'date'와 'alert_level' 컬럼이 필요합니다.")

    
    # --- 평균 위험도 & 추세 ---
    if not df_alerts.empty:
        df_alerts = _ensure_fused(df_alerts, bundle_id)
        avg_score = df_alerts["fused_score"].mean()
        st.caption(f"평균 위험도(Alerts 기준): {avg_score:.2f}")

        if "date" in df_alerts.columns:
            try:
                df_alerts["date"] = pd.to_datetime(df_alerts["date"]).dt.date
                trend = df_alerts.groupby("date", observed=True).size().reset_index(name="count")
                tabs_trend = st.tabs(["📈 알림 추세(일별 건수)", "📉 일평균 위험도"])
                with tabs_trend[0]:
                    fig_trend = px.line(
                        trend, x="date", y="count", markers=True,
                        title="📈 최근 알림 추세(일별 건수)",
                        color_discrete_sequence=next_palette()
                    )
                    st.plotly_chart(fig_trend, width='stretch')
                with tabs_trend[1]:
                    daily = df_alerts.groupby("date", observed=True)["fused_score"].mean().reset_index()
                    fig_avg = px.line(
                        daily, x="date", y="fused_score", markers=True,
                        title="📉 일평균 위험도(fused_score)",
                        color_discrete_sequence=next_palette()
                    )
                    st.plotly_chart(fig_avg, width='stretch')
            except Exception:
                pass

        # 경보 등급 분포 (탭)
        if "alert_level" in df_alerts.columns:
            lvl = df_alerts["alert_level"].value_counts().reset_index()
            lvl.columns = ["level", "count"]
            lvl_tabs = st.tabs(["📊 Bar", "🥧 Pie", "📄 표"])
            with lvl_tabs[0]:
                fig_lvl_bar = px.bar(
                    lvl, x="level", y="count", text="count",
                    title="🚨 경보 등급 분포 (Bar)",
                    color_discrete_sequence=next_palette()
                )
                fig_lvl_bar.update_traces(textposition="outside")
                st.plotly_chart(fig_lvl_bar, width='stretch')
            with lvl_tabs[1]:
                fig_lvl_pie = px.pie(
                    lvl, names="level", values="count",
                    title="🚨 경보 등급 분포 (Pie)",
                    color_discrete_sequence=next_palette()
                )
                st.plotly_chart(fig_lvl_pie, width='stretch')
            with lvl_tabs[2]:
                st.dataframe(lvl, width='stretch')
    else:
        st.info("Alerts 데이터가 없습니다. (⚡ 백로그 일괄 처리로 생성 가능)")

    # --- Top 위험 블록 (백필 포함, 탭) ---
    st.subheader("🔥 Top 위험 블록")
    if not df_blocks.empty:
        df_blocks = backfill_fused_score(df_blocks, bundle_id)
        if "fused_score" in df_blocks.columns and "block" in df_blocks.columns:
            top_blk = (df_blocks.groupby("block", observed=True)["fused_score"]
                       .mean().reset_index()
                       .sort_values("fused_score", ascending=False)
                       .head(10))
            tabs_blk = st.tabs(["📊 차트", "📄 표"])
            with tabs_blk[0]:
                fig_blk = px.bar(
                    top_blk, x="block", y="fused_score", text="fused_score",
                    title="블록별 평균 위험도 (Top 10)",
                    labels={"block": "블록", "fused_score": "평균 위험도"},
                    color_discrete_sequence=next_palette()
                )
                fig_blk.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                st.plotly_chart(fig_blk, width='stretch')
            with tabs_blk[1]:
                st.dataframe(top_blk, width='stretch')
        else:
            st.info("블록 데이터가 부족합니다.")
    else:
        st.info("블록 데이터 없음")

    # --- 🧭 위험 시나리오 (표만) ---
    st.subheader("🧭 위험 시나리오 (표)")
    if not df_scen.empty:
        df_scen = _ensure_fused(df_scen, bundle_id)

        # 시나리오 이름 추출 유틸
        def _scen_name(v):
            if isinstance(v, dict):
                return v.get("scenario") or v.get("title") or str(v)
            if isinstance(v, list) and v:
                first = v[0]
                if isinstance(first, dict):
                    return first.get("title") or first.get("scenario") or str(first)
                return str(first)
            if isinstance(v, str):
                return v
            return "Unknown"

        scen_col = "scenario_predicted" if "scenario_predicted" in df_scen.columns else (
            "scenario_matched" if "scenario_matched" in df_scen.columns else None
        )

        if scen_col:
            tmp = df_scen.copy()
            tmp["scenario_name"] = tmp[scen_col].apply(_scen_name)

            # 평균 위험도 기준 Top-N 표
            topn = st.slider("표시 개수(Top-N)", 5, 50, 20, 5, key="topn_scen")
            top_scen = (tmp.groupby("scenario_name", observed=True)["fused_score"]
                        .mean().reset_index()
                        .sort_values("fused_score", ascending=False)
                        .head(topn))
            st.dataframe(
                top_scen.rename(columns={"scenario_name":"시나리오", "fused_score":"평균 위험도"}),
                width='stretch'
            )

            # 최근 N건 원자료 미리보기(옵션)
            with st.expander("최근 시나리오 원자료 미리보기", expanded=False):
                cols = [c for c in ["date","title","scenario_predicted","scenario_matched","fused_score"] if c in tmp.columns]
                st.dataframe(tmp.sort_values("date", ascending=False)[cols].head(topn), width='stretch')
        else:
            st.info("시나리오 컬럼(scenario_predicted / scenario_matched)이 없습니다.")
    else:
        st.info("Scenarios 데이터가 없습니다. (⚡ 백로그 일괄 처리로 생성 가능)")


    # --- 🗺️ 대한민국 시군구 Choropleth (탭) ---
    st.subheader("🗺️ 대한민국 시군구 분포 (Choropleth)")
    geo_tabs = st.tabs(["🗺️ 지도", "📄 표"])
    with geo_tabs[0]:
        geo_path = st.session_state.get("siggeo", "")
        geo = load_geojson(geo_path) if geo_path else None
        if not df_alerts.empty and geo is not None:
            # df_alerts에 sig_cd 확보 (없으면 region/sigungu에서 유추)
            df_geo = ensure_sig_cd(df_alerts.copy(), geo)
            if "sig_cd" in df_geo.columns and df_geo["sig_cd"].notna().any():
                agg = (df_geo.dropna(subset=["sig_cd"])
                            .groupby("sig_cd", observed=True)
                            .size().reset_index(name="count"))
                code_key, _name_key = _detect_sig_props(geo)
                featureidkey = f"properties.{code_key}" if code_key else None
                fig_ch = px.choropleth(
                    agg, geojson=geo, locations="sig_cd", featureidkey=featureidkey,
                    color="count", color_continuous_scale=next_cscale(),
                    title="시군구별 경보 기사 분포"
                )
                fig_ch.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(fig_ch, width='stretch')
            else:
                st.info("df_alerts에 sig_cd를 유추할 수 없습니다. 'sigungu' 또는 'region' 컬럼을 채우거나, df에 'sig_cd'(5자리 행정코드)를 직접 추가하세요.")
        else:
            st.info("GeoJSON 경로가 비었거나 Alerts 데이터가 없습니다. 사이드바에서 GeoJSON 경로를 지정하세요.")
    with geo_tabs[1]:
        # 표는 집계 결과를 보여줌
        geo_path = st.session_state.get("siggeo", "")
        geo = load_geojson(geo_path) if geo_path else None
        if not df_alerts.empty and geo is not None:
            df_geo = ensure_sig_cd(df_alerts.copy(), geo)
            if "sig_cd" in df_geo.columns and df_geo["sig_cd"].notna().any():
                st.dataframe(
                    df_geo.dropna(subset=["sig_cd"]).groupby("sig_cd", observed=True).size().reset_index(name="count").sort_values("count", ascending=False),
                    width='stretch'
                )
            else:
                st.info("집계 표를 만들 sig_cd가 없습니다.")
        else:
            st.info("GeoJSON/Alerts 데이터가 준비되지 않았습니다.")

    # --- 🏷️ 엔티티 상위 Top-N (analyze.jsonl의 network 기반) ---
    st.subheader("🏷️ 엔티티 상위 Top-N")
    ent_tabs = st.tabs(["📊 차트", "📄 표"])
    if not df_ana.empty and "network" in df_ana.columns:
        from collections import Counter
        counter = Counter()
        for g in df_ana["network"].dropna():
            try:
                nodes = (g or {}).get("nodes", {})
                counter.update({k: int(v) for k, v in nodes.items()})
            except Exception:
                continue
        topn = st.slider("표시 개수", 5, 50, 20, 5)
        ent_df = pd.DataFrame(counter.most_common(topn), columns=["entity","count"])
        with ent_tabs[0]:
            fig_ent = px.bar(
                ent_df, x="entity", y="count", text="count",
                title=f"엔티티 상위 Top-{topn}",
                color_discrete_sequence=next_palette()
            )
            fig_ent.update_traces(textposition="outside")
            st.plotly_chart(fig_ent, width='stretch')
        with ent_tabs[1]:
            st.dataframe(ent_df, width='stretch')
    else:
        with ent_tabs[0]:
            st.info("Analyze 데이터에 네트워크 컬럼이 없어 엔티티 상위 Top-N을 생성할 수 없습니다.")
        with ent_tabs[1]:
            st.empty()

    # --- 📰 수집 소스 분포 (탭) ---
    st.subheader("📰 수집 소스 분포")
    if not df_ingest.empty and "source" in df_ingest.columns:
        src = df_ingest["source"].fillna("Manual").value_counts().reset_index()
        src.columns = ["source", "count"]
        src_tabs = st.tabs(["📊 차트", "📄 표"])
        with src_tabs[0]:
            fig_src = px.bar(
                src.head(30), x="source", y="count", text="count",
                title="언론사/도메인 분포 (Top 30)",
                labels={"source": "소스", "count": "건수"},
                color_discrete_sequence=next_palette()
            )
            fig_src.update_traces(textposition="outside")
            st.plotly_chart(fig_src, width='stretch')
        with src_tabs[1]:
            st.dataframe(src, width='stretch')
    else:
        st.info("Ingest의 source 컬럼이 없습니다.")

    # --- 백로그 일괄 처리 버튼 ---
    st.divider()
    colA, colB = st.columns([1, 3])
    with colA:
        if st.button("⚡ 백로그 일괄 처리 (ingest → alerts)"):
            ok = _quick_pipeline(bundle_id)
            if ok:
                st.success("파이프라인 실행 완료: ingest → alerts 생성")
            else:
                st.warning("ingest 데이터가 없습니다.")
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
        page_overview(bundle_id)
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
        render_alerts(ROOT)
    elif page == "EventBlocks":
        render_eventblocks(ROOT)
    elif page == "Fusion":
        _call_page("fusion", "page_fusion", ROOT)
    elif page == "Blocks":
        _call_page("blocks", "page_blocks", ROOT)
    elif page == "Scenarios":
        _call_page("scenarios", "page_scenarios", ROOT)
    elif page == "Ledger":
        _call_page("ledger", "page_ledger", ROOT)

if __name__ == "__main__":
    main()
