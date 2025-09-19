# nwwpkg/ui/dashboard/choropleth.py
import streamlit as st
import pandas as pd
import plotly.express as px

from nwwpkg.utils.geo import load_geojson, ensure_sig_cd, _detect_sig_props
from nwwpkg.utils.style import next_cscale

def render_choropleth(df_alerts: pd.DataFrame):
    """
    Render Choropleth maps for alerts distribution:
    - 국내 시군구 단위 분포
    - 세계 국가 단위 분포
    """
    st.markdown("### 🗺️ Choropleth (경보 위치 분포)")

    if df_alerts is None or df_alerts.empty:
        st.info("⚠️ Alerts 데이터가 없습니다. 위치 분포를 표시할 수 없습니다.")
        return

    # 이중 탭 구조
    tabs = st.tabs(["🇰🇷 국내 시군구", "🌍 세계 국가별"])
    geo_path = st.session_state.get("siggeo", "")

    # --- 국내 시군구 Choropleth ---
    with tabs[0]:
        st.subheader("🇰🇷 시군구별 경보 기사 분포")
        geo = load_geojson(geo_path) if geo_path else None

        if geo is not None:
            df_geo = ensure_sig_cd(df_alerts.copy(), geo)
            if "sig_cd" in df_geo.columns and df_geo["sig_cd"].notna().any():
                agg = (
                    df_geo.dropna(subset=["sig_cd"])
                          .groupby("sig_cd", observed=True)
                          .size().reset_index(name="count")
                )
                code_key, _ = _detect_sig_props(geo)
                featureidkey = f"properties.{code_key}" if code_key else None

                fig_ch = px.choropleth(
                    agg, geojson=geo, locations="sig_cd", featureidkey=featureidkey,
                    color="count", color_continuous_scale=next_cscale(),
                    title="시군구별 경보 기사 분포"
                )
                fig_ch.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(fig_ch, use_container_width=True)

                with st.expander("📄 집계 표", expanded=False):
                    st.dataframe(
                        agg.sort_values("count", ascending=False),
                        use_container_width=True
                    )
            else:
                st.warning("⚠️ df_alerts에 'sig_cd'를 유추할 수 없습니다. 'region'/'sigungu' 컬럼 확인 필요.")
        else:
            st.info("GeoJSON 경로가 비어있습니다. 사이드바에서 GeoJSON 파일을 지정하세요.")

    # --- 세계 국가별 Choropleth ---
    with tabs[1]:
        st.subheader("🌍 국가별 경보 기사 분포")
        if "country" not in df_alerts.columns:
            st.warning("⚠️ Alerts 데이터에 'country' 컬럼이 없습니다. 국가별 분포를 생성할 수 없습니다.")
        else:
            agg = df_alerts["country"].value_counts().reset_index()
            agg.columns = ["country", "count"]

            fig_world = px.choropleth(
                agg, locations="country", locationmode="country names",
                color="count", color_continuous_scale=next_cscale(),
                title="국가별 경보 기사 분포"
            )
            fig_world.update_geos(showcoastlines=True, showland=True, fitbounds="locations")
            st.plotly_chart(fig_world, use_container_width=True)

            with st.expander("📄 국가별 집계 표", expanded=False):
                st.dataframe(agg, use_container_width=True)
