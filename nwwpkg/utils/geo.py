# nwwpkg/utils/geo.py
import json
import pandas as pd
from pathlib import Path


def load_geojson(path: str | Path):
    """
    GeoJSON 파일을 로드하여 dict 반환
    """
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as e:
        print(f"[geo] GeoJSON 로드 실패: {e}")
        return None


def ensure_sig_cd(df: pd.DataFrame, geojson: dict) -> pd.DataFrame:
    """
    df에 행정구역 코드(sig_cd)를 보강.
    - df에 sig_cd가 있으면 그대로 사용
    - 없으면 region, sigungu 컬럼에서 유추 시도
    """
    if "sig_cd" in df.columns:
        return df

    if "sigungu" in df.columns:
        # 예: "서울특별시 강남구" 같은 문자열에서 코드 매핑
        df = df.copy()
        df["sig_cd"] = df["sigungu"].astype(str).str.slice(0, 5)
        return df

    if "region" in df.columns:
        df = df.copy()
        df["sig_cd"] = df["region"].astype(str).str.slice(0, 5)
        return df

    return df


def _detect_sig_props(geojson: dict) -> tuple[str, str]:
    """
    GeoJSON의 feature properties에서 code / name 필드 키 추론
    반환: (code_key, name_key)
    """
    try:
        props = geojson["features"][0]["properties"]
        code_key = None
        name_key = None
        for k in props.keys():
            if "cd" in k.lower():
                code_key = k
            if "name" in k.lower():
                name_key = k
        return code_key, name_key
    except Exception:
        return None, None
