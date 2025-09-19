# nwwpkg/utils/runner.py
from __future__ import annotations

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Iterable, Optional

from nwwpkg.utils.io import load_jsonl, save_jsonl, count_lines
from nwwpkg.utils.scoring import backfill_fused_score

# 우선 사용해볼 상위 산출물 우선순위
PREFERRED_SOURCES = [
    "fusion.jsonl",     # 가장 가공된 결과
    "scoring.jsonl",
    "scores.jsonl",
    "gate.jsonl",
    "analyze.jsonl",
    "ingest.jsonl",     # 최소 원천
]


def _pick_first_existing(bundle_path: Path, names: Iterable[str]) -> Optional[Path]:
    """존재하고 라인수>0인 첫 파일 선택"""
    for n in names:
        p = bundle_path / n
        if p.exists() and count_lines(p) > 0:
            return p
    return None


def _ensure_basic_alert_fields(df: pd.DataFrame) -> pd.DataFrame:
    """alerts.jsonl에 필요한 최소 필드를 보강"""
    df = df.copy()

    # 1) fused_score 보강
    df = backfill_fused_score(df, bundle_id=None)
    if "fused_score" not in df.columns:
        df["fused_score"] = 0.0

    # 2) alert_level 보강
    if "alert_level" not in df.columns and "level" in df.columns:
        df["alert_level"] = df["level"]

    if "alert_level" not in df.columns:
        def _to_level(x):
            try:
                v = float(x)
            except Exception:
                v = 0.0
            if v >= 0.75:
                return "High"
            elif v >= 0.5:
                return "Medium"
            return "Low"
        df["alert_level"] = df["fused_score"].apply(_to_level)

    # 3) date 보강
    if "date" not in df.columns:
        for c in ["t_on", "published_at", "event_time", "doc_time", "created_at", "time", "timestamp"]:
            if c in df.columns:
                df["date"] = df[c]
                break
    if "date" not in df.columns:
        df["date"] = pd.Timestamp.utcnow().isoformat()

    # 4) reason 보강(요약/텍스트 중 하나 잘라 쓰기)
    if "reason" not in df.columns:
        for c in ["reason", "why", "summary", "title", "clean_text", "text", "body"]:
            if c in df.columns:
                df["reason"] = df[c].astype(str).str.slice(0, 140)
                break
        if "reason" not in df.columns:
            df["reason"] = "auto-generated"

    return df


def _project_alert_columns(df: pd.DataFrame) -> pd.DataFrame:
    """alerts.jsonl에 실을 핵심 컬럼만 정리"""
    keep = [c for c in [
        "date", "alert_level", "fused_score", "reason",
        "title", "url", "source",
        "country", "region", "sigungu", "sig_cd",
        "id", "doc_id"
    ] if c in df.columns]
    return df[keep].copy()


def _write_alerts(df: pd.DataFrame, alerts_path: Path) -> int:
    rows = df.to_dict(orient="records")
    save_jsonl(alerts_path, rows)
    return len(rows)


def _quick_pipeline(bundle_id: str) -> bool:
    """
    ingest → alerts 최소 파이프라인 (MVP)
    - 가능한 상위 산출물부터 활용(fusion → scoring → … → ingest)
    - alerts.jsonl 생성
    반환: 성공 여부(bool)
    """
    bundle_path = Path(f"data/{bundle_id}")
    alerts_path = bundle_path / "alerts.jsonl"
    bundle_path.mkdir(parents=True, exist_ok=True)

    # 이미 alerts 있으면 OK
    if alerts_path.exists() and count_lines(alerts_path) > 0:
        return True

    # 소스 파일 선택
    src_path = _pick_first_existing(bundle_path, PREFERRED_SOURCES)
    if src_path is None:
        return False

    df = load_jsonl(src_path)
    if df is None or df.empty:
        return False

    # 필수 필드 보강 및 정리
    df = _ensure_basic_alert_fields(df)
    df = _project_alert_columns(df)

    # 위험도 높은 순으로 최대 N개만 저장 (과한 용량 방지)
    if "fused_score" in df.columns:
        df = df.sort_values("fused_score", ascending=False)
    max_rows = 5000
    df = df.head(max_rows)

    n = _write_alerts(df, alerts_path)
    return n > 0


def _safe_rerun():
    """Streamlit rerun (버전 호환)"""
    try:
        st.rerun()  # Streamlit >= 1.30
    except Exception:
        try:
            st.experimental_rerun()  # 구버전
        except Exception:
            pass
