# nwwpkg/ui/dashboard/alerts.py
import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO
from pathlib import Path

from nwwpkg.utils.io import _count_lines
from nwwpkg.utils.time import _ensure_datetime_col
from nwwpkg.utils.transform import _alert_timeseries, _to_wide, reset_index_as_date
from nwwpkg.utils.style import next_palette
# --- INLINE TIMESERIES FOR LANDING ------------------------------------------
import pandas as pd
import plotly.express as px

from nwwpkg.utils.time import _to_local_naive
from nwwpkg.utils.transform import _alert_timeseries, _to_wide, reset_index_as_date
from nwwpkg.utils.style import next_palette

# ❶ 날짜/시간 후보 자동탐지
def _autodetect_datetime_col(df: pd.DataFrame, min_ratio: float = 0.1) -> str | None:
    """
    모든 컬럼을 돌며 datelike로 파싱 가능한 비율이 가장 높은 컬럼을 고른다.
    min_ratio 미만이면 None.
    """
    best = (0.0, None)
    for c in df.columns:
        s = df[c]
        # 문자열/정수형 위주로만 시도
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s)):
            continue
        parsed = pd.to_datetime(s, errors="coerce", utc=True)
        ratio = float(parsed.notna().mean())
        if ratio > best[0]:
            best = (ratio, c)
    return best[1] if best[0] >= min_ratio else None

# --- 공통: 시간 컬럼 보강 ---
# ❷ 시간 컬럼 보강 (기존보다 더 공격적으로)
def _ensure_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 먼저 우리가 아는 표준 키들
    if "date" not in df.columns:
        for c in ["t_on", "created_at", "timestamp", "time", "dt"]:
            if c in df.columns:
                df["date"] = df[c]
                break

    if "src_time" not in df.columns:
        for c in ["published_at", "event_time", "doc_time", "pub_date", "pubDate", "datetime"]:
            if c in df.columns:
                df["src_time"] = df[c]
                break

    # 그래도 없으면 자동탐지 → date에 채움
    if "date" not in df.columns:
        cand = _autodetect_datetime_col(df, min_ratio=0.1)
        if cand:
            df["date"] = df[cand]

    return df


# ❸ 인라인·상세 공용 컨트롤 (인라인에서만 UI 노출)
def _alerts_controls(df: pd.DataFrame, key_prefix: str = "alerts", show_controls: bool = True):
    """
    - show_controls=True  → 라디오 노출 + state 저장
    - show_controls=False → 인라인에서 저장된 state 재사용 (없으면 자동 선택)
    반환: (basis, freq_label, time_col) 또는 (None, None, None)
    """
    df = _ensure_time_cols(df)

    # 후보 구성
    basis_opts = []
    if "date" in df.columns:     basis_opts.append("경보 생성시각")
    if "src_time" in df.columns: basis_opts.append("원문 시각")

    # 자동탐지 실패 시: 수동 선택+임시시간 옵션 제공
    if not basis_opts:
        # datelike 후보로 파싱이 되는 컬럼 목록
        candidates = []
        for c in df.columns:
            parsed = pd.to_datetime(df[c], errors="ignore")
            if pd.api.types.is_datetime64_any_dtype(parsed) or parsed.dtype == "datetime64[ns, UTC]":
                candidates.append(c)
        with st.container():
            st.warning("선택한 시간 기준에 해당하는 컬럼이 없습니다. (date / src_time)")
            if show_controls:
                with st.expander("🛠 시간 컬럼 직접 지정 / 임시 생성", expanded=True):
                    sel = st.selectbox("시간 컬럼 선택", options=["(없음)"] + candidates, index=0, key=f"{key_prefix}_manual_col")
                    fake = st.checkbox("임시 시간 생성(인덱스→연속 일자)", value=False, key=f"{key_prefix}_fake_time")
                    if sel != "(없음)":
                        df["date"] = df[sel]
                        basis_opts = ["경보 생성시각"]
                    elif fake:
                        # 오늘 기준 역순 분포(가장 단순한 placeholder)
                        start = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=len(df)-1)
                        df["date"] = pd.date_range(start=start, periods=len(df), freq="D")
                        basis_opts = ["경보 생성시각"]
                    else:
                        return None, None, None, df  # 여전히 불가
            else:
                return None, None, None, df  # 상세에서는 조작 불가

    # 컨트롤 표시 여부
    saved_basis = st.session_state.get(f"{key_prefix}_basis", basis_opts[0] if basis_opts else None)
    saved_freq  = st.session_state.get(f"{key_prefix}_freq",  "일")

    if show_controls:
        c1, c2 = st.columns([1,1])
        with c1:
            basis = st.radio("시간 기준 (Time basis)", basis_opts,
                             index=basis_opts.index(saved_basis) if saved_basis in basis_opts else 0,
                             horizontal=True, key=f"{key_prefix}_basis")
        with c2:
            freq_label = st.radio("집계(Granularity)", ["일","주","월"],
                                  index={"일":0,"주":1,"월":2}[saved_freq] if saved_freq in ["일","주","월"] else 0,
                                  horizontal=True, key=f"{key_prefix}_freq")
    else:
        # 인라인에서 저장한 값 재사용, 없으면 자동
        basis     = saved_basis if saved_basis in basis_opts else basis_opts[0]
        freq_label= saved_freq  if saved_freq  in ["일","주","월"] else "일"

    time_col = "date" if basis == "경보 생성시각" else "src_time"
    return basis, freq_label, time_col, df

def render_alerts_timeseries_inline(df_alerts: pd.DataFrame, bundle_id: str | None = None):
    st.subheader("📈 Alerts Trend (Inline)")
    if df_alerts is None or df_alerts.empty:
        st.info("⚠️ Alerts 데이터가 없습니다.")
        return

    basis, freq_label, time_col, df = _alerts_controls(df_alerts, key_prefix="alerts_inline", show_controls=True)
    if time_col is None:
        return

    df = _ensure_datetime_col(df, time_col, "Asia/Seoul")

    freq = {"일":"D","주":"W","월":"M"}[freq_label]
    ts_long = _alert_timeseries(df.rename(columns={time_col:"date"}), freq=freq)
    if ts_long is None or ts_long.empty:
        st.info("시계열 집계를 만들 수 없습니다.")
        return

    ts_wide = _to_wide(ts_long)
    num_cols = ts_wide.select_dtypes("number").columns
    if "Total" not in ts_wide.columns:
        ts_wide["Total"] = ts_wide[num_cols].sum(axis=1) if len(num_cols) > 0 else 0

    smooth = st.checkbox("7-창 스무딩(rolling mean)", value=(freq=="D"), key="alerts_inline_smooth")
    plot_df = (ts_wide[["Total"]].rolling(window=7, min_periods=1).mean() if smooth else ts_wide[["Total"]])
    plot_df = reset_index_as_date(plot_df)

    fig = px.line(plot_df, x="date", y="Total", markers=True,
                  title=f"전체 경보 추세 (Total) · {freq_label} 집계{' · 7창' if smooth else ''}",
                  color_discrete_sequence=next_palette())
    st.plotly_chart(fig, use_container_width=True)

def render_alert(df_alerts: pd.DataFrame, bundle_id: str, bundle_path=None):
    st.caption("Alerts 상세 보기")
    if df_alerts is None or df_alerts.empty:
        st.info("⚠️ Alerts 데이터가 없습니다.")
        return

    # 인라인과 동일 로직이지만 show_controls=False
    basis, freq_label, time_col, df = _alerts_controls(df_alerts, key_prefix="alerts_inline", show_controls=False)
    if time_col is None:
        st.warning("시간 기준 컬럼이 없어 상세 차트를 생성할 수 없습니다.")
        return

    df = _ensure_datetime_col(df, time_col, "Asia/Seoul")
    freq = {"일":"D","주":"W","월":"M"}[freq_label]

    ts_long = _alert_timeseries(df.rename(columns={time_col:"date"}), freq=freq)
    if ts_long is None or ts_long.empty:
        st.info("시계열 집계를 만들 수 없습니다.")
        return

    ts_wide = _to_wide(ts_long)
    num_cols = ts_wide.select_dtypes("number").columns
    if "Total" not in ts_wide.columns:
        ts_wide["Total"] = ts_wide[num_cols].sum(axis=1) if len(num_cols) > 0 else 0

    smooth = st.checkbox("7-창 스무딩(rolling mean)", value=(freq=="D"), key="alerts_full_smooth")
    ts_wide_sm = ts_wide.rolling(window=7, min_periods=1).mean() if smooth else ts_wide

    tabs = st.tabs(["🟣 스택 면적", "📈 레벨별 선", "➕ 누적 합", "📄 표"])
    with tabs[0]:
        fig_area = px.area(ts_long, x="date", y="count", color="alert_level",
                           title=f"경보 수 스택 면적 ({freq_label})", color_discrete_sequence=next_palette())
        st.plotly_chart(fig_area, use_container_width=True)
    with tabs[1]:
        df_plot = reset_index_as_date(ts_wide_sm)
        ycols = [c for c in ["High","Medium","Low"] if c in df_plot.columns]
        fig_line = px.line(df_plot, x="date", y=ycols,
                           title=f"경보 레벨별 추세 ({freq_label}){' · 7창' if smooth else ''}",
                           color_discrete_sequence=next_palette())
        st.plotly_chart(fig_line, use_container_width=True)
    with tabs[2]:
        df_cum = reset_index_as_date(ts_wide.cumsum())
        ycols = [c for c in ["High","Medium","Low","Total"] if c in df_cum.columns]
        fig_cum = px.line(df_cum, x="date", y=ycols,
                          title=f"경보 누적 합 ({freq_label})",
                          color_discrete_sequence=next_palette())
        st.plotly_chart(fig_cum, use_container_width=True)
    with tabs[3]:
        st.dataframe(ts_wide.reset_index(), use_container_width=True)
