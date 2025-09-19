# nwwpkg/utils/style.py
import itertools

# 기본 색상 팔레트 (Plotly / Streamlit 호환)
_DEFAULT_COLORS = [
    "#1f77b4",  # 파랑
    "#ff7f0e",  # 주황
    "#2ca02c",  # 초록
    "#d62728",  # 빨강
    "#9467bd",  # 보라
    "#8c564b",  # 갈색
    "#e377c2",  # 분홍
    "#7f7f7f",  # 회색
    "#bcbd22",  # 올리브
    "#17becf",  # 청록
]

# 연속 색상 팔레트
_DEFAULT_CSCALE = "Viridis"

# 순환 가능한 팔레트 제너레이터
_palette_cycle = itertools.cycle(_DEFAULT_COLORS)


def next_palette():
    """
    순차적으로 색상 팔레트 반환 (plotly / px용)
    """
    return list(itertools.islice(_palette_cycle, 0, len(_DEFAULT_COLORS)))


def next_cscale():
    """
    Choropleth 등에서 사용할 연속 색상 스케일 반환
    """
    return _DEFAULT_CSCALE
