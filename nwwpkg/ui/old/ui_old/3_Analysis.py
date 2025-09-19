import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nwwpkg.ui.components.sidebar_nav import render_sidebar_nav

st.set_page_config(page_title="Analyze", layout="wide")
st.title("🔍 Analyze")

render_sidebar_nav()

# 키워드 테이블
df_kw = pd.DataFrame({"keyword":["war","conflict","peace"], "count":[20,15,5]})
st.dataframe(df_kw)

# 감정 분석 / 프레임 분포
df_sent = pd.DataFrame({"frame":["긍정","부정","중립"], "count":[5,15,10]})
st.plotly_chart(px.pie(df_sent, names="frame", values="count", title="프레임 분포"))

# 워드클라우드
wc = WordCloud(width=400, height=200).generate(" ".join(df_kw["keyword"]))
fig, ax = plt.subplots()
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)
