import os, json, re, io, math, random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# =========================
# 안전 IO (통합)
# =========================
def load_jsonl(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            try:
                lines.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(lines)

def save_jsonl(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = row.to_dict()
            for k, v in rec.items():
                if isinstance(v, pd.Timestamp):
                    rec[k] = v.isoformat()
                elif isinstance(v, float) and pd.isna(v):
                    rec[k] = None
                elif isinstance(v, (pd.Series, pd.DataFrame)):
                    rec[k] = str(v)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# =========================
# 텍스트 유틸
# =========================
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text.strip())
    text = text.lower()
    # 영/숫/한/공백만 남김
    return re.sub(r"[^a-z0-9가-힣\s]", " ", text).strip()

def sent_split(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    # 간단 분리(.!?/개행)
    parts = re.split(r"[\.!\?\n]+", text)
    return [p.strip() for p in parts if p.strip()]

def tok(text: str) -> list[str]:
    return [t for t in text.split() if t]

# =========================
# 간이 감정/프레임/도메인 룰
# =========================
POS_WORDS = set("진전 완화 회복 합의 개선 반등 상승 안정 평화 협력 지원 구조 성공 확장 도약".split())
NEG_WORDS = set("위기 전쟁 충돌 교전 사망 부상 파업 붕괴 침체 폭락 압박 제재 긴장 비상 경고 파탄".split())

FRAME_LEXICON = {
    "Conflict":  ["충돌","전쟁","교전","군사","미사일","공습","사망","부상","발포","병력","훈련","핵","포격","교전"],
    "Diplomacy": ["협상","회담","합의","중재","외교","제재완화","정상회담","통화","서한","접촉"],
    "Economic":  ["물가","금리","환율","주가","성장","실업","위기","제조업","수출","공급망","재정","부채","적자","투자","파업","임금"],
    "Domestic":  ["총선","대선","의회","법안","시위","검찰","대법원","내각","언론","부패","탄핵"],
    "Disaster":  ["지진","태풍","폭우","홍수","가뭄","화재","산불","참사","전염병","확진","격리","피해"]
}

DOMAIN_FROM_FRAME = {
    "Conflict": "Security",
    "Diplomacy": "Political",
    "Economic": "Economic",
    "Domestic": "Political",
    "Disaster": "Disaster"
}

REGION_REGEX = {
    "Korea": r"(한국|대한민국|서울|부산|인천|대구|대전|광주)",
    "US": r"(미국|워싱턴|뉴욕|LA|캘리포니아|바이든|펜타곤)",
    "China": r"(중국|베이징|상하이|시진핑)",
    "Japan": r"(일본|도쿄|오사카|기시다)",
    "Russia": r"(러시아|모스크바|푸틴)",
    "EU": r"(유럽연합|EU|브뤼셀|독일|프랑스|이탈리아|스페인)",
}

def infer_region(text: str) -> str:
    for r, pat in REGION_REGEX.items():
        if re.search(pat, text or "", flags=re.IGNORECASE):
            return r
    return "Unknown"

def infer_sentiment(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in POS_WORDS)
    neg = sum(1 for t in tokens if t in NEG_WORDS)
    # [-1, 1] 범위
    if pos == 0 and neg == 0:
        return 0.0
    score = (pos - neg) / max(1, (pos + neg))
    return float(max(-1, min(1, score)))

def infer_frame(tokens: list[str]) -> tuple[str, dict]:
    hits = {}
    for frame, kws in FRAME_LEXICON.items():
        hits[frame] = sum(1 for t in tokens for k in kws if k in t)
    best = max(hits, key=hits.get) if hits else "Unknown"
    return best if hits[best] > 0 else "Unknown", hits

# =========================
# TF-IDF (간이 구현)
# =========================
def tfidf_topk(docs: list[list[str]], topk: int = 10) -> list[list[tuple[str,float]]]:
    if not docs:
        return [[]]
    # DF
    df_counter = {}
    for tokens in docs:
        for w in set(tokens):
            df_counter[w] = df_counter.get(w, 0) + 1
    n_docs = len(docs)
    out = []
    for tokens in docs:
        tf = {}
        for w in tokens:
            tf[w] = tf.get(w, 0) + 1
        scores = []
        for w, c in tf.items():
            idf = math.log((n_docs + 1) / (1 + df_counter.get(w, 1))) + 1
            scores.append((w, (c / len(tokens)) * idf))
        scores.sort(key=lambda x: x[1], reverse=True)
        out.append(scores[:topk])
    return out

# =========================
# 공통 경로
# =========================
def P(bundle: str, name: str) -> str:
    return os.path.join("data", bundle, name)

# =========================
# Landing
# =========================
def page_overview(bundle: str):
    import plotly.express as px
    import pandas as pd
    import streamlit as st
    from datetime import date, timedelta

    st.title("🌍 NWW Crisis Overview")

    # 데이터 로드
    df_alerts = load_jsonl(P(bundle, "alerts.jsonl"))
    df_scen   = load_jsonl(P(bundle, "scenarios.jsonl"))
    df_an     = load_jsonl(P(bundle, "analyze.jsonl"))  # 프레임 통계용

    # =========================
    # KPI (기사 수 / 활성 알림 수 / 평균 위험도)
    # =========================
    total_articles = len(df_alerts) if not df_alerts.empty else 0

    # 활성 알림 수: is_alert 또는 alert 컬럼 우선 사용
    if not df_alerts.empty and "is_alert" in df_alerts.columns:
        active_alerts = int(pd.to_numeric(df_alerts["is_alert"], errors="coerce").fillna(0).sum())
    elif not df_alerts.empty and "alert" in df_alerts.columns:
        active_alerts = int(pd.to_numeric(df_alerts["alert"], errors="coerce").fillna(0).sum())
    else:
        active_alerts = 0

    # 평균 위험도: risk > risk_fused > score 순으로 탐색
    if not df_alerts.empty:
        if "risk" in df_alerts.columns:
            avg_series = pd.to_numeric(df_alerts["risk"], errors="coerce").fillna(0)
        elif "risk_fused" in df_alerts.columns:
            avg_series = pd.to_numeric(df_alerts["risk_fused"], errors="coerce").fillna(0)
        elif "score" in df_alerts.columns:
            avg_series = pd.to_numeric(df_alerts["score"], errors="coerce").fillna(0)
        else:
            avg_series = pd.Series([], dtype="float64")
        avg_score = float(avg_series.mean()) if not avg_series.empty else 0.0
    else:
        avg_score = 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("기사 수", total_articles)
    c2.metric("활성 알림 수", active_alerts)
    c3.metric("평균 위험도", f"{avg_score:.2f}")

    st.markdown("---")

    # =========================
    # 최근 7일 알림 추세 (line chart)
    # =========================
    st.subheader("📈 최근 7일 알림 추세")
    if not df_alerts.empty and "date" in df_alerts.columns:
        dcol = pd.to_datetime(df_alerts["date"], errors="coerce").dt.date
        tmp = df_alerts.copy()
        tmp["__date"] = dcol

        # is_alert가 있으면 알림만 집계, 없으면 전체 기사 수 집계
        if "is_alert" in tmp.columns:
            tmp["__alerts"] = pd.to_numeric(tmp["is_alert"], errors="coerce").fillna(0)
        elif "alert" in tmp.columns:
            tmp["__alerts"] = pd.to_numeric(tmp["alert"], errors="coerce").fillna(0)
        else:
            tmp["__alerts"] = 1  # 대체: 알림 컬럼 없으면 전체 건수

        daily = tmp.dropna(subset=["__date"]).groupby("__date")["__alerts"].sum().reset_index()
        daily.columns = ["date", "alerts"]

        # 최근 7일만 보이도록 보정(없으면 0으로 채움)
        if not daily.empty:
            end = date.today()
            idx = pd.date_range(end=end, periods=7, freq="D").date
            grid = pd.DataFrame({"date": idx})
            daily = grid.merge(daily, on="date", how="left").fillna({"alerts": 0})
            fig = px.line(daily, x="date", y="alerts", markers=True, title="최근 7일 알림 발생 추세")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("표시할 일자 데이터가 없습니다.")
    else:
        st.info("알림 시계열을 표시할 데이터가 없습니다.")

    st.markdown("---")

    # =========================
    # Top 5 위험 시나리오 (테이블/카드)
    # =========================
    st.subheader("🔥 Top 5 위험 시나리오")
    if not df_scen.empty:
        # 위험도 기준 컬럼 선택
        if "risk" in df_scen.columns:
            key = "risk"
        elif "score" in df_scen.columns:
            key = "score"
        elif "risk_fused" in df_scen.columns:
            key = "risk_fused"
        else:
            key = None

        if key:
            tmp = df_scen.copy()
            tmp[key] = pd.to_numeric(tmp[key], errors="coerce").fillna(0)
            top5 = tmp.sort_values(key, ascending=False).head(5)

            # 보기 좋은 컬럼 우선 선택
            cols = [c for c in ["title","scenario","frame","domain",key] if c in top5.columns]
            if not cols:
                cols = [key]
            st.dataframe(top5[cols], use_container_width=True)
        else:
            st.info("시나리오 위험도 컬럼이 없습니다.")
    else:
        st.info("시나리오 데이터가 없습니다.")

    st.markdown("---")

    # =========================
    # 지역별 분포 (bar) / 분야별 분포 (pie)
    # =========================
    st.subheader("🌐 지역/분야 분포")
    # 지역 데이터 소스 우선순위: alerts → analyze → scenarios
    region_src = None
    if not df_alerts.empty and "region" in df_alerts.columns:
        region_src = df_alerts["region"]
    elif not df_an.empty and "region" in df_an.columns:
        region_src = df_an["region"]
    elif not df_scen.empty and "region" in df_scen.columns:
        region_src = df_scen["region"]

    if region_src is not None:
        rc = region_src.fillna("Unknown").value_counts().reset_index()
        rc.columns = ["region", "count"]
        fig_region = px.bar(rc, x="region", y="count", text="count", title="지역별 분포")
        fig_region.update_traces(textposition="outside")
        st.plotly_chart(fig_region, use_container_width=True)
    else:
        st.info("지역 정보가 없습니다.")

    # 도메인 데이터 소스 우선순위: alerts → analyze → scenarios
    domain_src = None
    if not df_alerts.empty and "domain" in df_alerts.columns:
        domain_src = df_alerts["domain"]
    elif not df_an.empty and "domain" in df_an.columns:
        domain_src = df_an["domain"]
    elif not df_scen.empty and "domain" in df_scen.columns:
        domain_src = df_scen["domain"]

    if domain_src is not None:
        dom = pd.DataFrame({"domain": domain_src.fillna("Unknown")})
        fig_dom = px.pie(dom, names="domain", title="분야별 분포")
        st.plotly_chart(fig_dom, use_container_width=True)
    else:
        st.info("분야 정보가 없습니다.")

    st.markdown("---")

    # =========================
    # 프레임 분포 (bar) – 옵션(있을 때만)
    # =========================
    st.subheader("🧩 프레임 분포")
    frame_src = None
    if not df_alerts.empty and "frame" in df_alerts.columns:
        frame_src = df_alerts["frame"]
    elif not df_an.empty and "frame" in df_an.columns:
        frame_src = df_an["frame"]

    if frame_src is not None:
        fr = frame_src.fillna("Unknown").value_counts().reset_index()
        fr.columns = ["frame", "count"]
        fig_fr = px.bar(fr, x="frame", y="count", text="count", title="프레임 분포")
        fig_fr.update_traces(textposition="outside")
        st.plotly_chart(fig_fr, use_container_width=True)
    else:
        st.info("프레임 정보가 없습니다.")

# =========================
# Ingest (URL/원문)
# =========================
def try_fetch(url: str) -> str | None:
    """trafilatura/bs4가 있으면 활용, 없으면 폴백"""
    if not url:
        return None
    # trafilatura
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url, timeout=8)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text and len(text) > 80:
                return text
    except Exception:
        pass
    # requests + bs4
    try:
        import requests
        from bs4 import BeautifulSoup
        r = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
        if r.ok:
            soup = BeautifulSoup(r.text, "html.parser")
            ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            text = " ".join(ps)
            return text if len(text) > 80 else None
    except Exception:
        pass
    return None

def page_ingest(bundle: str):
    st.header("📰 Ingest – 기사 수집")
    url  = st.text_input("기사 URL (선택)")
    text = st.text_area("기사 원문 (선택)", height=200, placeholder="URL이 안 될 때 원문을 붙여넣으세요.")

    colA, colB = st.columns([1,1])
    with colA:
        if st.button("💾 기사 저장"):
            if not (url or text):
                st.warning("URL 또는 원문 중 하나는 입력해야 합니다.")
            else:
                fetched = text
                if not fetched and url:
                    with st.spinner("URL에서 본문 추출 중..."):
                        fetched = try_fetch(url)
                df = load_jsonl(P(bundle, "ingest.jsonl"))
                new = {
                    "id": int(datetime.utcnow().timestamp()*1000) + random.randint(0, 999),
                    "url": url or None,
                    "text": fetched or "",
                    "date": datetime.utcnow().date().isoformat(),
                    "source": "URL" if url and fetched else ("Manual" if text else "URL(no-body)")
                }
                df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
                save_jsonl(df, P(bundle, "ingest.jsonl"))
                st.success("✅ 저장 완료")
    with colB:
        if st.button("🧹 전체 삭제(ingest.jsonl)"):
            p = P(bundle, "ingest.jsonl")
            if os.path.exists(p):
                os.remove(p)
                st.warning("삭제 완료")

    st.markdown("### 📄 저장된 기사")
    df = load_jsonl(P(bundle, "ingest.jsonl"))
    if df.empty:
        st.info("저장된 기사가 없습니다.")
        return
    st.dataframe(df[["date","url","source"]].tail(50), use_container_width=True)

    st.markdown("### 🏷 언론사(소스) 분포")
    counts = df["source"].fillna("Unknown").value_counts().reset_index()
    counts.columns = ["source","count"]
    fig = px.bar(counts, x="source", y="count", text="count", title="언론사별 기사 분포")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Normalize
# =========================
def page_normalize(bundle: str):
    st.header("🔤 Normalize – 텍스트 정규화")
    df = load_jsonl(P(bundle, "ingest.jsonl"))
    if df.empty:
        st.info("먼저 Ingest에서 기사를 입력하세요.")
        return

    df["norm"]   = df["text"].apply(normalize_text)
    df["sents"]  = df["norm"].apply(sent_split)
    df["tokens"] = df["norm"].apply(tok)
    df["tok_len"]= df["tokens"].apply(len)

    st.subheader("원문 ↔ 정규화 비교 (샘플)")
    st.dataframe(df[["text","norm"]].head(10), use_container_width=True)

    st.subheader("문장 길이(토큰 수) 분포")
    st.plotly_chart(px.histogram(df, x="tok_len", nbins=20), use_container_width=True)

    save_jsonl(df, P(bundle, "normalize.jsonl"))
    st.success("normalize.jsonl 저장 완료")

    # 다운로드
    buf = io.StringIO(); df.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button("⬇️ CSV 다운로드", buf.getvalue(), file_name=f"{bundle}_normalized.csv", mime="text/csv")

# =========================
# Analyze (TF-IDF/감정/프레임/지역/도메인)
# =========================
def page_analyze(bundle: str):
    st.header("🔎 Analyze – 키워드·감정·프레임")
    df = load_jsonl(P(bundle, "normalize.jsonl"))
    if df.empty:
        st.info("Normalize 결과가 필요합니다.")
        return

    # TF-IDF 상위 키워드
    docs = df["tokens"].apply(lambda x: x if isinstance(x, list) else []).tolist()
    topk = tfidf_topk(docs, topk=10)
    df["keywords"] = [kws for kws in topk]

    # 감정/프레임/지역/도메인
    df["sentiment"] = df["tokens"].apply(infer_sentiment)
    fr_dom = df["tokens"].apply(infer_frame)
    df["frame"]  = [fd[0] for fd in fr_dom]
    df["frame_hits"] = [fd[1] for fd in fr_dom]
    df["domain"] = df["frame"].apply(lambda f: DOMAIN_FROM_FRAME.get(f, "Other"))
    df["region"] = df["norm"].apply(infer_region)

    st.subheader("키워드/프레임/감정 (샘플)")
    show = df[["id","region","domain","frame","sentiment"]].copy()
    show["keywords"] = df["keywords"].apply(lambda ks: ", ".join([w for w, _ in ks][:8]))
    st.dataframe(show.head(20), use_container_width=True)

    st.subheader("프레임 분포")
    fr = df["frame"].fillna("Unknown").value_counts().reset_index()
    fr.columns = ["frame","count"]
    st.plotly_chart(px.bar(fr, x="frame", y="count", text="count"), use_container_width=True)

    # 워드클라우드(전체)
    st.subheader("☁️ 워드클라우드")
    corpus = " ".join(df["norm"].dropna().tolist())
    if corpus.strip():
        wc = WordCloud(width=900, height=400, background_color="white").generate(corpus)
        fig, ax = plt.subplots(figsize=(10,4)); ax.imshow(wc); ax.axis("off"); st.pyplot(fig)
    else:
        st.warning("유효한 단어가 없어 워드클라우드 생성을 건너뜁니다.")

    save_jsonl(df, P(bundle, "analyze.jsonl"))
    st.success("analyze.jsonl 저장 완료")

# =========================
# Gate (체크리스트 룰 → 신호)
# =========================
CHECKLIST = {
    "Conflict":  {"교전":0.4, "전쟁":0.5, "사망":0.6, "군사":0.3, "미사일":0.5, "훈련":0.2},
    "Diplomacy": {"협상":0.4, "합의":0.5, "회담":0.4, "중재":0.3, "제재":0.3},
    "Economic":  {"위기":0.5, "물가":0.3, "금리":0.3, "환율":0.3, "파업":0.4, "공급망":0.4},
    "Domestic":  {"시위":0.4, "법안":0.2, "검찰":0.2, "부패":0.3, "탄핵":0.5},
    "Disaster":  {"지진":0.6, "태풍":0.5, "폭우":0.4, "화재":0.4, "전염병":0.5},
}
def match_signals(text: str, frame: str) -> float:
    w = CHECKLIST.get(frame, {})
    score = 0.0
    for k, val in w.items():
        if k in (text or ""):
            score += val
    return float(min(1.0, score))

def page_gate(bundle: str, threshold: float):
    st.header("🚪 Gate – 체크리스트 매칭")
    df = load_jsonl(P(bundle, "analyze.jsonl"))
    if df.empty:
        st.info("Analyze 결과가 필요합니다.")
        return

    df["signal"] = df.apply(lambda r: match_signals(r.get("norm",""), r.get("frame","Unknown")), axis=1)
    st.subheader("신호값 분포")
    st.plotly_chart(px.histogram(df, x="signal", nbins=20), use_container_width=True)

    st.subheader("임계값 조정")
    th = st.slider("Gate 임계값", 0.0, 1.0, float(threshold), 0.05, key="gate_th")
    df["gate_pass"] = df["signal"] >= th
    st.write(f"✅ Gate 통과 기사 수: {int(df['gate_pass'].sum())} / {len(df)}")

    save_jsonl(df, P(bundle, "gate.jsonl"))
    st.success("gate.jsonl 저장 완료")

# =========================
# Scoring (가중합 + EWMA)
# =========================
def ewma(series: pd.Series, alpha=0.3):
    out = []
    s = None
    for v in series:
        s = v if s is None else (alpha*v + (1-alpha)*s)
        out.append(s)
    return out

def page_scoring(bundle: str):
    st.header("📊 Scoring – 위험도 점수화")
    df = load_jsonl(P(bundle, "gate.jsonl"))
    if df.empty:
        st.info("Gate 결과가 필요합니다.")
        return

    # 위험도 = 0.5*signal + 0.3*neg(-sentiment) + 0.2*frame_weight
    frame_w = df["frame"].map({
        "Conflict":0.9,"Disaster":0.8,"Economic":0.6,"Domestic":0.5,"Diplomacy":0.4
    }).fillna(0.3)
    neg = (-pd.to_numeric(df["sentiment"], errors="coerce").fillna(0)).clip(lower=0)
    sig = pd.to_numeric(df["signal"], errors="coerce").fillna(0)
    risk = 0.5*sig + 0.3*neg + 0.2*frame_w
    df["risk_raw"] = risk.clip(0,1)

    # 날짜 정렬 후 EWMA
    dates = pd.to_datetime(df.get("date", datetime.utcnow().date()))
    df = df.assign(date=dates).sort_values("date")
    df["risk"] = ewma(df["risk_raw"], alpha=0.35)

    st.subheader("위험 점수 추이")
    st.plotly_chart(px.line(df, x="date", y="risk", markers=True), use_container_width=True)

    save_jsonl(df, P(bundle, "scoring.jsonl"))
    st.success("scoring.jsonl 저장 완료")

# =========================
# Fusion (보정/신뢰도)
# =========================
def page_fusion(bundle: str):
    st.header("🔗 Fusion – 점수 보정/통합")
    df = load_jsonl(P(bundle, "scoring.jsonl"))
    if df.empty:
        st.info("Scoring 결과가 필요합니다.")
        return

    # min-max 보정 + 신뢰도(키워드 수 + frame_hits 합)
    mm = (pd.Series(df["risk"]) - pd.Series(df["risk"]).min()) / (pd.Series(df["risk"]).max() - pd.Series(df["risk"]).min() + 1e-9)
    df["risk_fused"] = mm.fillna(0)

    conf = df["keywords"].apply(lambda ks: len(ks) if isinstance(ks, list) else 0) + \
            df["frame_hits"].apply(lambda d: sum(d.values()) if isinstance(d, dict) else 0)
    df["confidence"] = (conf - conf.min()) / (conf.max() - conf.min() + 1e-9)

    st.subheader("Fusion 전/후 비교")
    st.plotly_chart(px.scatter(df, x="risk", y="risk_fused", color="confidence"), use_container_width=True)

    save_jsonl(df, P(bundle, "fusion.jsonl"))
    st.success("fusion.jsonl 저장 완료")

# =========================
# Blocks (EDS): 프레임/도메인 기반 블록
# =========================
def page_blocks(bundle: str):
    st.header("🧱 Blocks (EDS)")
    df = load_jsonl(P(bundle, "fusion.jsonl"))
    if df.empty:
        st.info("Fusion 결과가 필요합니다.")
        return

    df["block"] = df.apply(lambda r: f"{r.get('frame','Unknown')}_{r.get('domain','Other')}", axis=1)

    st.subheader("기사 ↔ 블록 매핑 (샘플)")
    st.dataframe(df[["id","region","domain","frame","block","risk_fused"]].head(30), use_container_width=True)

    save_jsonl(df, P(bundle, "blocks.jsonl"))
    st.success("blocks.jsonl 저장 완료")

# =========================
# Scenarios: 블록 묶음 → 시나리오
# =========================
def page_scenarios(bundle: str):
    st.header("📜 Scenarios – 시나리오 생성")
    df = load_jsonl(P(bundle, "blocks.jsonl"))
    if df.empty:
        st.info("Blocks 결과가 필요합니다.")
        return

    grp = df.groupby(["block","frame","domain","region"], dropna=False)["risk_fused"].mean().reset_index()
    grp["title"] = grp.apply(lambda r: f"{r['region']} - {r['frame']} 리스크", axis=1)
    grp["risk"]  = grp["risk_fused"].round(3)

    st.subheader("시나리오 카드 (Top 8)")
    top = grp.sort_values("risk", ascending=False).head(8)
    for _, r in top.iterrows():
        st.markdown(
            f"<div style='padding:10px;border-radius:10px;background:#f6f8fb;margin:6px 0'>"
            f"<b>{r['title']}</b><br>도메인: {r['domain']} | 블록: {r['block']}<br>"
            f"<b>Risk:</b> {r['risk']}</div>", unsafe_allow_html=True
        )

    save_jsonl(grp, P(bundle, "scenarios.jsonl"))
    st.success("scenarios.jsonl 저장 완료")

# =========================
# Alerts: 시나리오 위험도 → 경보
# =========================
def page_alerts(bundle: str, alert_threshold: float):
    st.header("🚨 Alerts – 경보 생성")
    scen = load_jsonl(P(bundle, "scenarios.jsonl"))
    if scen.empty:
        st.info("Scenarios 결과가 필요합니다.")
        return

    th = st.slider("경보 기준 점수", 0.0, 1.0, float(alert_threshold), 0.05, key="alert_th")
    alerts = scen.copy()
    alerts["is_alert"] = alerts["risk"] >= th
    st.write(f"발생 경보 수: {int(alerts['is_alert'].sum())}")

    st.subheader("지역/분야 분포")
    if "region" in alerts:
        st.plotly_chart(px.bar(alerts["region"].value_counts().reset_index().rename(columns={"index":"region","region":"count"}), x="region", y="count"), use_container_width=True)
    if "domain" in alerts:
        st.plotly_chart(px.pie(alerts, names="domain", title="도메인 분포"), use_container_width=True)

    save_jsonl(alerts, P(bundle, "alerts.jsonl"))
    st.success("alerts.jsonl 저장 완료")

# =========================
# Event Blocks: 프레임×일자 그룹
# =========================
def page_event_blocks(bundle: str):
    st.header("📦 Event Blocks – 간이 클러스터")
    df = load_jsonl(P(bundle, "alerts.jsonl"))
    if df.empty:
        st.info("Alerts 결과가 필요합니다.")
        return

    # 일자 생성
    if "date" not in df.columns:
        df["date"] = datetime.utcnow().date().isoformat()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    df["cluster"] = df["frame"].astype(str) + "_" + pd.Series(df["date"]).astype(str)
    agg = df.groupby("cluster")["risk"].mean().reset_index().rename(columns={"risk":"risk_mean"})
    st.dataframe(agg.head(30), use_container_width=True)

    # 히트맵 느낌(막대)
    st.plotly_chart(px.bar(agg, x="cluster", y="risk_mean"), use_container_width=True)

    save_jsonl(agg, P(bundle, "event_blocks.jsonl"))
    st.success("event_blocks.jsonl 저장 완료")

# =========================
# Ledger: 로그 테이블
# =========================
def page_ledger(bundle: str):
    st.header("📒 Ledger")
    files = [
        "ingest.jsonl","normalize.jsonl","analyze.jsonl","gate.jsonl","scoring.jsonl",
        "fusion.jsonl","blocks.jsonl","scenarios.jsonl","alerts.jsonl","event_blocks.jsonl"
    ]
    rows = []
    for f in files:
        p = P(bundle, f)
        rows.append({"file": f, "exists": os.path.exists(p), "size": (os.path.getsize(p) if os.path.exists(p) else 0)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# =========================
# 전체 파이프라인(Run All)
# =========================
def run_all(bundle: str, gate_th: float, alert_th: float):
    # 전 단계 연속 실행 (데이터가 없으면 해당 단계 건너뛰지 않고 빈 처리)
    if load_jsonl(P(bundle, "ingest.jsonl")).empty:
        st.info("Run All을 위해 최소 1개 기사를 Ingest에 저장하세요.")
        return
    page_normalize(bundle)
    page_analyze(bundle)
    page_gate(bundle, gate_th)
    page_scoring(bundle)
    page_fusion(bundle)
    page_blocks(bundle)
    page_scenarios(bundle)
    page_alerts(bundle, alert_th)
    page_event_blocks(bundle)
    page_ledger(bundle)
    st.success("🎉 파이프라인 완료")

# =========================
# Main
# =========================
def main():
    st.set_page_config(page_title="NWW Early Warning Dashboard", layout="wide", initial_sidebar_state="expanded")

    st.sidebar.header("⚙️ 설정")
    bundle = st.sidebar.text_input("Bundle ID (data/<bundle>)", "sample")
    gate_th = st.sidebar.slider("Gate 임계값", 0.0, 1.0, 0.5, 0.05)
    alert_th = st.sidebar.slider("Alert 임계값", 0.0, 1.0, 0.7, 0.05)

    if st.sidebar.button("▶ Run All"):
        run_all(bundle, gate_th, alert_th)
        return  # 한 번에 그려줌

    st.sidebar.markdown("---")
    page = st.sidebar.radio("📌 단계", [
        "Landing","Ingest","Normalize","Analyze","Gate","Scoring","Fusion",
        "Blocks (EDS)","Scenarios","Alerts","Event Blocks","Ledger"
    ])

    if page == "Landing":         page_overview(bundle)
    elif page == "Ingest":        page_ingest(bundle)
    elif page == "Normalize":     page_normalize(bundle)
    elif page == "Analyze":       page_analyze(bundle)
    elif page == "Gate":          page_gate(bundle, gate_th)
    elif page == "Scoring":       page_scoring(bundle)
    elif page == "Fusion":        page_fusion(bundle)
    elif page == "Blocks (EDS)":  page_blocks(bundle)
    elif page == "Scenarios":     page_scenarios(bundle)
    elif page == "Alerts":        page_alerts(bundle, alert_th)
    elif page == "Event Blocks":  page_event_blocks(bundle)
    elif page == "Ledger":        page_ledger(bundle)

if __name__ == "__main__":
    main()
