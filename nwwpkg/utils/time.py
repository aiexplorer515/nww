# nwwpkg/utils/time.py
from datetime import datetime
import pandas as pd
import pytz


def _to_local_naive(
    ts: str | datetime | pd.Timestamp,
    input_tz: str = "UTC",
    output_tz: str = "Asia/Seoul"
) -> datetime | None:
    """
    UTC 기반 timestamp를 한국시간(Asia/Seoul)으로 변환한 naive datetime 반환
    - ts: 문자열, datetime, 또는 pandas.Timestamp
    - input_tz: 입력 타임존 (기본값: UTC)
    - output_tz: 출력 타임존 (기본값: Asia/Seoul)
    """
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None

    try:
        if isinstance(ts, str):
            dt = pd.to_datetime(ts, errors="coerce", utc=True)
        elif isinstance(ts, (datetime, pd.Timestamp)):
            dt = pd.to_datetime(ts, errors="coerce", utc=True)
        else:
            return None

        if dt is None or pd.isna(dt):
            return None

        # 입력 타임존 지정
        if dt.tzinfo is None:
            dt = dt.tz_localize(input_tz)

        # 출력 타임존 변환 후 naive로 반환
        return dt.tz_convert(output_tz).to_pydatetime().replace(tzinfo=None)

    except Exception:
        return None


def _ensure_datetime_col(df: pd.DataFrame, col: str, output_tz: str = "Asia/Seoul"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        df[col] = df[col].dt.tz_convert(output_tz).dt.tz_localize(None)
    return df
