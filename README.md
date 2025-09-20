# 🌍 NWW (News War earlyWarning) MVP

## 📖 프로젝트 목표 (Project Goal)

**NWW (News War earlyWarning)**는 뉴스·SNS·유튜브 등 공개 데이터를 수집·분석하여  
**전쟁 및 사회 위기 조기경보(Early Warning)**를 제공하는 **AI 주도형 분석·경보 시스템**입니다.  

- **문제의식**: 단일 기사나 제한된 분석에 의존한 위기 해석은 오류와 편향이 많음  
- **해결책**: AI가 주도적으로 데이터 수집·정규화·분석·시나리오 생성을 수행하고,  
  **전문가는 검증자(Verifier)**로 참여하여 신뢰성과 투명성을 보강  
- **비전**: AI-First + Expert-Verified 조기경보 체계 → 국가·국제기구·NGO가 공동 활용 가능한 **개방형 Early Warning 인프라**  

---

## 📂 파일/폴더 구조 (Directory Structure)

```
nww/
├── nwwpkg/               # 핵심 Python 모듈
│   ├── ingest/           # 데이터 수집
│   ├── preprocess/       # 정규화 + 엔티티 추출
│   ├── analyze/          # 키워드/프레임 분석 + 워드클라우드
│   ├── scoring/          # 위험 점수화 (IS/DBN/LLM Fusion)
│   ├── scenario/         # 시나리오 생성·매칭
│   ├── fusion/           # 사건·인물 종합 추론
│   └── ui/               # Streamlit 기반 UI
├── data/
│   └── bundles/b01/      # 샘플 데이터 번들
├── tools/
│   └── mvp_check.py      # 종합 성능 테스트 스크립트
├── docs/
│   ├── UI_Guide.md       # UI 설명서
│   ├── TESTING.md        # 테스트 가이드
│   └── ErrorManual.md    # 에러 매뉴얼
└── logs/
    └── error.log         # 실행/오류 로그
```

---

## 🔄 NWW 파이프라인 (Pipeline Overview)

**AI 주도 분석 → 전문가 검증 → 경보 발령**  

| 단계 | 기능 | 출력물 |
|------|------|--------|
| Ingest | 뉴스·SNS·유튜브 데이터 수집 | `raw.jsonl` |
| Normalize | 텍스트 정제 + 인물/사건 중심 엔티티 추출 | `clean.enriched.jsonl`, `entities.jsonl` |
| Analyze | 키워드 추출, 프레임 태깅, 워드클라우드 | `keyword.jsonl`, `frame.jsonl`, `keyword_wc.png` |
| Gate | 전문가 체크리스트(EDS) 매칭 | `gated.jsonl` |
| Scoring | IS + DBN + LLM Fusion 위험 점수화 | `scored.jsonl` |
| Risk | Crisis Block 도출 | `blocks.jsonl` |
| Alerts | 경보 발령 (Low/Med/High) | `alerts.jsonl` |
| Scenarios | GPT 기반 시나리오 생성/매칭 | `scenarios.jsonl` |
| Ledger | 실행 로그, 버전 관리 | `logs/error.log` |

📄 **UI 화면 구성과 기능 설명은 [`docs/UI_Guide.md`](./docs/UI_Guide.md) 참고.**

---

## ✨ 주요 기능 (Key Features)

- **AI 주도 자동화**: 데이터 수집 → 분석 → 시나리오 생성 자동 수행  
- **전문가 참여·검증**: 체크리스트(EDS)와 시나리오 리뷰를 통해 **검증형 구조** 보강  
- **위험 점수화**: Indicator Scoring + DBN 추론 + LLM Fusion  
- **시나리오 카드화**: GPT 기반 위기 시나리오 자동 생성 및 전문가 피드백 반영  
- **경보 발령**: 위험 레벨 (Low/Medium/High) 자동 산출 + 전문가 검토 로그화  
- **투명성 보장**: 모든 결과는 Ledger에 기록 → 추적 가능  

---

## ⚡️ MVP 성능 검증 (Performance Validation)

MVP는 **AI 자동화 + 전문가 검증** 구조를 바탕으로 **최소 성능 요건(Exit Criteria)**을 만족합니다.  

- **E2E 안정성**: URL 100건 실행 성공률 ≥ 95% → ✅ 97%  
- **본문 추출 품질**: Coverage ≥ 90% → ✅ 92%  
- **프레임 태깅**: 목표 F1 ≥ 0.60 → ⚠️ 현재 0.58 (개선 필요)  
- **경보 일관성**: Score 변동 ≤ ±5% → ✅ 4%  
- **UI 품질**: 랜딩·대시보드 오류 없음 → ✅ 달성  

📄 **세부 테스트 가이드는 [`docs/TESTING.md`](./docs/TESTING.md) 참고.**

---

## 🧩 확장 계획 (Future Extensions)

- **멀티모달 분석**: 이미지·음성·동영상 기반 시나리오 분석  
- **가짜뉴스 탐지 (Fake News Detection)**  
  - Source 평판 기반 신뢰도 평가  
  - Cross-Source Fact-Check 자동화  
  - GPT 기반 Claim ↔ Evidence 매칭  
- **블록체인 연동**: Crisis Block 공유·무결성 보장  
- **전문가 네트워크 강화**: 국제 전문가 그룹 참여형 검증 시스템  
- **국제 협력형 Early Warning 모델** 구축  

---

## ⚙️ 실행 방법 (Quick Start)

```bash
# 1. 가상환경 생성
conda create -n nww python=3.11
conda activate nww

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 변수 설정
export NWW_DATA_HOME="data"
export NWW_BUNDLE="b01"
export PYTHONUTF8=1

# 4. UI 실행
streamlit run nwwpkg/ui/app_main.py

# 5. 성능 테스트 실행
python tools/mvp_check.py --bundle b01
```

---

## 📜 라이선스 (License)

본 프로젝트는 연구/실험용 MVP이며,  
향후 오픈소스화 및 국제 협력 모델로의 확장을 목표로 합니다.  

---

## 🙌 기여 (Contributing)

- Issue 등록 및 Pull Request 환영  
- 데이터셋/체크리스트 제안 가능  
- 국제 협력 및 전문가 네트워크 참여 가능  

---

## ✉️ 문의 (Contact)

- Project Lead: 리더 (던전)  
- 연구 분야: AI · LLM 보안 · OSINT 기반 위기 조기경보  
- 📧 Contact: [your_email@example.com]  
