# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import os
import inspect
import streamlit as st

# --- 안전 import: 누락되면 UI에 명확히 표시 ---
try:
    from nwwpkg.ui.pages.fusion import page_fusion
    from nwwpkg.ui.pages.blocks import page_blocks
    from nwwpkg.ui.pages.scenarios import page_scenarios
    from nwwpkg.ui.pages.ledger import page_ledger
except Exception as e:
    st.error(f"[import error] 페이지 모듈을 불러오지 못했습니다: {e}")
    st.stop()

__all__ = ["render_fusion", "render_blocks", "render_scenarios", "render_ledger"]

# --- 유틸 ---
def _bundle_id(root: Path) -> str:
    """data/<bundle> 구조 기준. 예외적으로 env가 우선이면 NWW_BUNDLE 사용."""
    return os.getenv("NWW_BUNDLE", root.name or "b01")

def _guard_root(root: Path):
    if not root.exists():
        st.error(f"[경로 오류] 데이터 루트가 없습니다: {root}")
        st.stop()

def _safe_call(func, bid: str):
    """기존 시그니처 page_xxx(bundle_id) 보존. callable 검증."""
    if not callable(func):
        st.error(f"[함수 오류] 호출 불가: {func}")
        st.stop()
    # 필요시 시그니처 점검(인자 수가 다르면 도움말 출력)
    sig = inspect.signature(func)
    if len(sig.parameters) != 1:
        st.warning(f"[시그니처 주의] {func.__name__}{sig} → (bundle_id:str) 1개 인자 권장")
    return func(bid)

# --- Render Wrappers ---
def render_fusion(root: Path):
    _guard_root(root)
    st.header("🧬 Fusion")
    bid = _bundle_id(root)
    with st.spinner("Fusion 페이지 로딩 중..."):
        _safe_call(page_fusion, bid)

def render_blocks(root: Path):
    _guard_root(root)
    st.header("🧱 Blocks")
    bid = _bundle_id(root)
    with st.spinner("Blocks 페이지 로딩 중..."):
        _safe_call(page_blocks, bid)

def render_scenarios(root: Path):
    _guard_root(root)
    st.header("🗺️ Scenarios")
    bid = _bundle_id(root)
    with st.spinner("Scenarios 페이지 로딩 중..."):
        _safe_call(page_scenarios, bid)

def render_ledger(root: Path):
    _guard_root(root)
    st.header("📒 Ledger")
    bid = _bundle_id(root)
    with st.spinner("Ledger 페이지 로딩 중..."):
        _safe_call(page_ledger, bid)
