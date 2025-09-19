# app_main.py (확장: Ingest → Normalize → Analyze)
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
from datetime import datetime
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json


# --------------------------
# 유틸
# --------------------------
import streamlit as st
import pandas as pd
import os
import json   # ✅ 빠진 부분 추가
from datetime import datetime
# Default bundle_id
bundle_id = "sample"  # 실제 데이터 폴더명으로 바꿔야 함

# 실제 기능 모듈 import
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

# JSONL 로드/저장 함수
def load_jsonl(path: str) -> pd.DataFrame:
    """JSONL → DataFrame 로드"""
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    return pd.DataFrame(lines)


def save_jsonl(df: pd.DataFrame, path: str):
    """DataFrame → JSONL 저장 (Timestamp, NaN 변환 포함)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = row.to_dict()

            # 🔹 안전 변환
            for k, v in record.items():
                if isinstance(v, pd.Timestamp):         # 날짜/시간 → 문자열
                    record[k] = v.isoformat()
                elif pd.isna(v):                         # NaN → None
                    record[k] = None
                elif isinstance(v, (pd.Series, pd.DataFrame)):  
                    record[k] = str(v)                   # 중첩 구조 방지

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
# ---------------- Ingest 탭 ----------------
def page_ingest(bundle_id="sample"):
    st.header("📰 Ingest – 기사 수집")

    url = st.text_input("기사 URL 입력")
    text = st.text_area("기사 원문 입력", height=200)

    if st.button("기사 저장"):
        if url:
            # 실제 기사 본문 수집 호출
            text = news_collector.collect(url) if not text else text
            source = news_collector.get_domain(url)
        else:
            source = "Manual"

        if url or text:
            # JSONL 저장
            df = pd.DataFrame([{
                "url": url if url else None,
                "text": text if text else None,
                "date": datetime.today().strftime("%Y-%m-%d"),
                "source": source
            }])
            recorder.save(bundle_id, "ingest.jsonl", df)
            st.success("✅ 기사 저장 완료")
        else:
            st.warning("⚠️ URL 또는 기사 원문 입력 필요")

    # 저장된 기사 미리보기
            # 저장된 기사 미리보기
    st.subheader("📑 저장된 기사 목록")
    df = load_jsonl(f"data/{bundle_id}/ingest.jsonl")
    if not df.empty:
        st.dataframe(df.tail(10))

        # ✅ 언론사(source) 분포 차트
        counts = df["source"].value_counts().reset_index()
        counts.columns = ["source", "count"]  # 컬럼명 명시적으로 변경
        import plotly.express as px
        fig = px.bar(
            counts,
            x="source",   # ✅ 가로축 = 언론사
            y="count",    # ✅ 세로축 = 건수
            text="count", # 막대 위에 숫자 표시
            labels={"source": "언론사", "count": "기사 수"},
            title="언론사별 기사 분포"
        )
        fig.update_traces(textposition="outside")  # 숫자를 막대 위로
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("아직 저장된 기사가 없습니다.")

      

# --------------------------
# Normalize
# --------------------------
import re
import io

def normalize_text(text: str) -> str:
    if not isinstance(text, str):  # None, NaN 등 예외 처리
        return ""
    return re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text.lower()).strip()


def page_normalize(bundle_id):
    st.header("🔤 Normalize – 텍스트 정규화")

    df = load_jsonl(f"data/{bundle_id}/ingest.jsonl")
    if df.empty:
        st.info("⚠️ 먼저 Ingest 단계에서 기사를 입력하세요.")
        return

    # 정규화 실행
    df["normalized"] = df["text"].apply(normalize_text)

    # 비교 테이블
    st.subheader("원문 vs 정규화 텍스트")
    st.dataframe(df[["text", "normalized"]].head(10), use_container_width=True)

    # 문장 길이 분포
    df["length"] = df["normalized"].apply(lambda x: len(x.split()))
    fig = px.histogram(df, x="length", nbins=20, title="문장 길이(토큰 수) 분포")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # 📥 다운로드 기능 추가
    # -----------------------------
    st.subheader("📥 정규화 데이터 다운로드")

    # CSV 변환
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    st.download_button(
        label="⬇️ CSV 다운로드",
        data=csv_buffer.getvalue(),
        file_name=f"{bundle_id}_normalized.csv",
        mime="text/csv"
    )

    # JSONL 변환
    jsonl_buffer = io.StringIO()
    for _, row in df.iterrows():
        jsonl_buffer.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    st.download_button(
        label="⬇️ JSONL 다운로드",
        data=jsonl_buffer.getvalue(),
        file_name=f"{bundle_id}_normalized.jsonl",
        mime="application/json"
    )

    # 저장
    save_jsonl(df, f"data/{bundle_id}/normalize.jsonl")
    st.success("정규화 데이터가 저장되었습니다.")


# --------------------------
# Analyze
# --------------------------
def extract_keywords(texts, topn=20):
    words = " ".join(texts).split()
    counter = Counter(words)
    return pd.DataFrame(counter.most_common(topn), columns=["word", "freq"])

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def page_analyze(bundle_id):
    st.header("🔍 Analyze – 키워드/프레임/감정 분석")

    df = load_jsonl(f"data/{bundle_id}/normalize.jsonl")
    if df.empty or "normalized" not in df.columns:
        st.info("⚠️ 먼저 Normalize 단계에서 데이터를 준비하세요.")
        return

    # ---------------------
    # 키워드 테이블 예시
    # ---------------------
    st.subheader("📑 키워드 샘플")
    sample_keywords = [{"keyword": "군사훈련", "count": 5}, {"keyword": "협상", "count": 3}]
    st.dataframe(sample_keywords)

    # ---------------------
    # 감정/프레임 시각화 예시
    # ---------------------
    st.subheader("📊 감정 분포 (예시 데이터)")
    fig = px.pie(values=[40, 30, 20, 10], names=["긍정", "부정", "중립", "기타"])
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------
    # 워드클라우드 생성
    # ---------------------
    st.subheader("☁️ 워드클라우드")

    text_corpus = " ".join([t for t in df["normalized"].dropna().tolist() if t.strip() != ""])
    if len(text_corpus.split()) > 0:
        font_path = get_korean_font_path(st.session_state.get("kofont") if "kofont" in st.session_state else font_path_input if "font_path_input" in locals() else None)
        if not font_path:
            st.warning("⚠️ 한글 폰트를 찾지 못했습니다. 사이드바에서 TTF/TTC 경로를 지정해주세요.")
        else:
            ensure_matplotlib_korean(font_path)
            wc = WordCloud(
                width=1000, height=500,
                background_color="white",
                font_path=font_path,           # ✅ 핵심
                prefer_horizontal=1.0
            ).generate(text_corpus)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig, clear_figure=True)
    else:
        st.warning("⚠️ 유효한 단어가 없어 워드클라우드를 생성할 수 없습니다.")


# --------------------------
# Placeholder (이후 단계)
# --------------------------
# --------------------------
# Gate (체크리스트 매칭)
# --------------------------
CHECKLIST = {
    "military": {"병력": 0.4, "무기": 0.3, "훈련": 0.2},
    "diplomatic": {"협상": 0.3, "결렬": 0.4, "제재": 0.3},
    "economic": {"위기": 0.3, "금리": 0.2, "물가": 0.5}
}

def match_checklist(text: str):
    results = {}
    for domain, rules in CHECKLIST.items():
        score = 0
        for kw, w in rules.items():
            if kw in text:
                score += w
        results[domain] = round(score, 2)
    return results

def page_gate(bundle_id: str):
    st.header("🚪 Gate – 체크리스트 매칭")

    df = load_jsonl(f"data/{bundle_id}/normalize.jsonl")
    if df.empty:
        st.warning("먼저 Normalize 단계를 실행하세요.")
        return

    df["signals"] = df["normalized"].apply(match_checklist)

    st.subheader("체크리스트 매칭 결과")
    st.write(df[["normalized", "signals"]].head())

    # 가중치 vs 신호값 차트
    scores = pd.DataFrame(df["signals"].tolist())
    fig = px.bar(scores.sum().reset_index(), x="index", y=0,
                 labels={"index": "Domain", "0": "총 신호값"},
                 title="도메인별 신호 합계")
    st.plotly_chart(fig, use_container_width=True)

    threshold = st.slider("임계값", 0.0, 1.0, 0.5, 0.05)
    df["alert"] = scores.max(axis=1) > threshold
    st.write(f"🚨 경보 발생 기사 수: {df['alert'].sum()}")

    save_jsonl(df, f"data/{bundle_id}/gate.jsonl")

# --------------------------
# Scoring
# --------------------------
def page_scoring(bundle_id: str):
    st.header("📊 Scoring – 위험 점수 산출")

    df = load_jsonl(f"data/{bundle_id}/gate.jsonl")
    if df.empty:
        st.warning("먼저 Gate 단계를 실행하세요.")
        return

    # 단순 합산 점수
    scores = pd.DataFrame(df["signals"].tolist())
    df["score"] = scores.sum(axis=1)

    st.subheader("위험 점수 breakdown")
    st.write(df[["normalized", "signals", "score"]].head())

    # 시계열 그래프
    df["date"] = pd.to_datetime(df.get("date", datetime.today().strftime("%Y-%m-%d")))
    fig = px.line(df, x="date", y="score", title="위험 점수 시계열")
    st.plotly_chart(fig, use_container_width=True)

    save_jsonl(df, f"data/{bundle_id}/scoring.jsonl")



# --------------------------
# Fusion
# --------------------------
def page_fusion(bundle_id: str):
    st.header("⚡ Fusion – 신뢰 보정 및 통합")

    df = load_jsonl(f"data/{bundle_id}/scoring.jsonl")
    if df.empty:
        st.warning("먼저 Scoring 단계를 실행하세요.")
        return

    df["fusion_score"] = df["score"] * (0.9 + 0.2 * pd.Series(range(len(df))) / len(df))
    st.subheader("Fusion 전/후 비교")
    st.dataframe(df[["score", "fusion_score"]].head())

    fig = px.scatter(df, x="score", y="fusion_score", title="Fusion 전/후 비교")
    st.plotly_chart(fig, use_container_width=True)

    save_jsonl(df, f"data/{bundle_id}/fusion.jsonl")

# --------------------------
# Blocks (EDS)
# --------------------------
def page_blocks(bundle_id: str):
    st.header("🧩 Blocks (EDS) – 기사 ↔ 블록 매핑")

    df = load_jsonl(f"data/{bundle_id}/fusion.jsonl")
    if df.empty:
        st.warning("먼저 Fusion 단계를 실행하세요.")
        return

    df["block"] = df["normalized"].apply(lambda x: "Military" if "병력" in x else "Economic")
    st.subheader("기사 ↔ 블록 매핑")
    st.dataframe(df[["normalized", "block", "fusion_score"]].head())

    save_jsonl(df, f"data/{bundle_id}/blocks.jsonl")

# --------------------------
# Scenarios
# --------------------------
def page_scenarios(bundle_id: str):
    st.header("📑 Scenarios – 시나리오 생성")

    df = load_jsonl(f"data/{bundle_id}/blocks.jsonl")
    if df.empty:
        st.warning("먼저 Blocks 단계를 실행하세요.")
        return

    df["scenario"] = df["block"] + "_Scenario"
    top = df.groupby("scenario")["fusion_score"].mean().reset_index()
    st.subheader("시나리오별 평균 점수")
    st.dataframe(top)

    save_jsonl(df, f"data/{bundle_id}/scenarios.jsonl")

# --------------------------
# Alerts
# --------------------------
def page_alerts(bundle_id: str):
    st.header("🚨 Alerts – 경보 발생")

    df = load_jsonl(f"data/{bundle_id}/scenarios.jsonl")
    if df.empty:
        st.warning("먼저 Scenarios 단계를 실행하세요.")
        return

    df["alert_level"] = pd.cut(df["fusion_score"],
                               bins=[-1, 0.3, 0.6, 1.0],
                               labels=["Low", "Medium", "High"])
    st.subheader("경보 조건 근거")
    st.dataframe(df[["normalized", "fusion_score", "alert_level"]].head())

    fig = px.histogram(df, x="alert_level", title="경보 분포")
    st.plotly_chart(fig, use_container_width=True)

    save_jsonl(df, f"data/{bundle_id}/alerts.jsonl")

# --------------------------
# Event Blocks
# --------------------------
def page_eventblocks(bundle_id: str):
    st.header("📦 Event Blocks – 클러스터링")

    df = load_jsonl(f"data/{bundle_id}/alerts.jsonl")
    if df.empty:
        st.warning("먼저 Alerts 단계를 실행하세요.")
        return

    df["cluster"] = df.index % 3
    pivot = df.groupby("cluster")["fusion_score"].mean().reset_index()

    st.subheader("클러스터별 평균 위험도")
    st.dataframe(pivot)

    fig = px.imshow([pivot["fusion_score"].tolist()],
                    labels=dict(x="Cluster", y="Risk", color="Score"),
                    title="Event Block 위험도 히트맵")
    st.plotly_chart(fig, use_container_width=True)

    save_jsonl(df, f"data/{bundle_id}/eventblocks.jsonl")

# --------------------------
# Ledger
# --------------------------
def page_ledger(bundle_id: str):
    st.header("📜 Ledger – 단계별 로그")

    files = [
        "ingest.jsonl", "normalize.jsonl", "analyze.jsonl",
        "gate.jsonl", "scoring.jsonl", "fusion.jsonl",
        "blocks.jsonl", "scenarios.jsonl", "alerts.jsonl",
        "eventblocks.jsonl"
    ]
    logs = []
    for f in files:
        path = f"data/{bundle_id}/{f}"
        if os.path.exists(path):
            logs.append({"file": f, "size": os.path.getsize(path)})
    st.dataframe(pd.DataFrame(logs))

    if st.button("📥 ledger.jsonl 다운로드"):
        ledger = {"bundle": bundle_id, "timestamp": datetime.now().isoformat(), "steps": files}
        save_jsonl(pd.DataFrame([ledger]), f"data/{bundle_id}/ledger.jsonl")
        st.success("ledger.jsonl 저장 완료")


# ----------------------------
# 1. Landing 함수 (main 전에 정의)
# ----------------------------
def page_overview(df_alerts, df_scen):
    st.title("🌍 Crisis Overview")

    df_alerts = load_jsonl(f"data/{bundle_id}/alerts.jsonl")
    df_scen = load_jsonl(f"data/{bundle_id}/scenarios.jsonl")

    # ---- KPI ----
    total_articles = len(df_alerts) if not df_alerts.empty else 0
    active_alerts = df_alerts["alert"].sum() if "alert" in df_alerts else 0
    avg_score = df_alerts["score"].mean() if "score" in df_alerts else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("기사 수", total_articles)
    col2.metric("활성 알림 수", int(active_alerts))
    col3.metric("평균 위험도", f"{avg_score:.2f}")

    # ---- 최근 7일 알림 추세 ----
    if "date" in df_alerts:
        trend = df_alerts.groupby("date").size().reset_index(name="count")
        fig = px.line(trend, x="date", y="count", title="📈 최근 7일 알림 추세")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Top 5 위험 시나리오 ----
    st.subheader("🔥 Top 5 위험 시나리오")
    if "score" in df_scen:
        top5 = df_scen.sort_values("score", ascending=False).head(5)
        st.dataframe(top5)
    else:
        st.info("시나리오 데이터 없음")

    # ---- 지역별 분포 ----
    if "region" in df_alerts:
        fig = px.bar(df_alerts["region"].value_counts().reset_index(),
                     x="index", y="region",
                     labels={"index": "지역", "region": "건수"},
                     title="📊 지역별 분포")
        st.plotly_chart(fig, use_container_width=True)

    # ---- 분야별 분포 ----
    if "domain" in df_alerts:
        fig = px.pie(df_alerts, names="domain", title="🥧 분야별 분포")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Main
# --------------------------
def main():
    st.set_page_config(page_title="NWW Early Warning Dashboard", layout="wide")

    st.sidebar.header("⚙️ 설정")
    bundle_id = st.sidebar.text_input("Bundle ID", "sample")
    page = st.sidebar.radio("📌 단계 이동", [
        "Landing", "Ingest", "Normalize", "Analyze", "Gate", "Scoring",
        "Fusion", "Blocks", "Scenarios", "Alerts", "EventBlocks", "Ledger"
    ])

    df_alerts = load_jsonl(f"data/{bundle_id}/alerts.jsonl")
    df_scen = load_jsonl(f"data/{bundle_id}/scenarios.jsonl")

    if page == "Landing":
        page_overview(df_alerts, df_scen)
    elif page == "Ingest":
        page_ingest(bundle_id)
    elif page == "Normalize":
        page_normalize(bundle_id)
    elif page == "Analyze":
        page_analyze(bundle_id)
    elif page == "Gate":
        page_gate(bundle_id)
    elif page == "Scoring":
        page_scoring(bundle_id)
    elif page == "Fusion":
        page_fusion(bundle_id)
    elif page == "Blocks":
        page_blocks(bundle_id)
    elif page == "Scenarios":
        page_scenarios(bundle_id)
    elif page == "Alerts":
        page_alerts(bundle_id)
    elif page == "EventBlocks":
        page_eventblocks(bundle_id)
    elif page == "Ledger":
        page_ledger(bundle_id)

if __name__ == "__main__":
    main()
