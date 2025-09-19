# nwwpkg/utils/transform.py
import pandas as pd


def _alert_timeseries(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    경보 데이터(alerts DataFrame)를 시계열 집계 형태로 변환
    - freq: 집계 단위 (D=일, W=주, M=월)
    """
    if "date" not in df.columns:
        raise ValueError("DataFrame에 'date' 컬럼이 필요합니다.")

    # datetime 변환
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if "alert_level" not in df.columns:
        df["alert_level"] = "Unknown"

    # groupby 후 count
    ts = (
        df.groupby([pd.Grouper(key="date", freq=freq), "alert_level"], observed=True)
        .size()
        .reset_index(name="count")
        .sort_values("date")
    )

    return ts


def _to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    long-form (date, alert_level, count) → wide-form (date, 각 level별 count)
    """
    if not {"date", "alert_level", "count"}.issubset(df.columns):
        raise ValueError("DataFrame에 'date', 'alert_level', 'count' 컬럼이 필요합니다.")

    wide = df.pivot_table(
        index="date",
        columns="alert_level",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )

    wide = wide.sort_index()
    return wide


def reset_index_as_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    reset_index() 후 index를 date 컬럼으로 변환
    """
    if isinstance(df.index, pd.DatetimeIndex):
        df_reset = df.reset_index().rename(columns={"index": "date"})
    else:
        df_reset = df.reset_index()
        if "index" in df_reset.columns:
            df_reset = df_reset.rename(columns={"index": "date"})
    return df_reset
