# 📖 NWW UI 설명서 (UI_Guide.md)

## 🔑 핵심 개념 (Key Concepts)

- **UI 모듈화 (UI Modularization)**  
  모든 화면은 **단계별(수집 → 전처리 → 체크리스트 → 스코어링 → 시나리오 → 경보)** 흐름으로 구성됨.  
  → Streamlit 멀티페이지 구조, 또는 `app_main.py`에서 모듈 import 방식으로 구현.

- **데이터 번들(Bundle)**  
  `NWW_BUNDLE` 환경변수 기반으로 UI는 항상 특정 데이터셋(`data/bundles/bXX/`)과 연결됨.  

- **사이드바(Sidebar)**  
  공통 메뉴 + 환경 설정(번들 선택, 모델 선택, 경보 임계치 조정)을 제공.  

---

## 🖥️ 화면 구조 (UI Pages)

### 1. **Landing**
- **개요 페이지**  
  - 전체 파이프라인(Pipeline DAG)의 처리 단계별 카운트 표시 (ingest, clean, dedup, keyword, kboost, frame, score, alert).  
  - **Stage QA(품질 점검)**: 각 단계별 데이터 품질 확인.  
  - **Alerts Trend**: 시계열 기반 경보 생성 추세를 보여줌.  
    - 집계 단위: 일/주/월  
    - 스무딩(rolling mean) 적용 가능.  

---

### 2. **Ingest (수집)**
- 외부 뉴스·SNS·문서 데이터를 가져오는 단계.  
- 수집된 기사/데이터 건수가 **Pipeline DAG**의 `ingest`에 반영됨.  
- 업로드/크롤링 로그 확인 가능.  

---

### 3. **Normalize (정규화 + 엔티티 추출)**
- **텍스트 클린징**: HTML 태그, 특수문자 제거.  
- **언어 감지**: 원문 언어 탐지 및 필요 시 번역.  
- **중복 제거(dedup)**: 동일 기사/문서 제거.  
- **엔티티 추출(Entity Extraction)**:  
  - 인물(Person), 사건(Event), 조직(Organization), 장소(Location) 중심 엔티티 식별.  
  - 이후 **Analyze/Scenarios** 단계에서 시나리오 기반 분석의 핵심 입력으로 활용.  
  - 추출 결과는 `entities.jsonl` 또는 DB에 저장되어 후속 모듈과 연동됨.  
- **출력**: `clean.enriched.jsonl` (정규화 + 엔티티 포함)  

---

### 4. **Analyze (분석)**
- **키워드 추출(`keyword`)**, **프레임 태깅(`frame`)** 수행.  
- GPT/Lexicon 기반 분석 적용.  
- **워드클라우드 시각화(Word Cloud Visualization)**:  
  - 기사 및 뉴스에서 빈출 단어를 직관적으로 표시.  
  - 프레임 신호·핵심 키워드의 밀집도를 빠르게 파악 가능.  
- 결과물은 **EDA(탐색적 데이터 분석)** 차트, 표, 워드클라우드로 미리보기 제공.  

---

### 5. **Gate (체크리스트 매칭)**
- **전문가데이터시스템(EDS) 체크리스트**와 수집된 데이터 비교.  
- 위기 신호 여부(`kboost`)를 판단해 필터링.  
- 이후 Risk/Scoring 단계의 입력으로 전달.  

---

### 6. **Scoring (위험 점수화)**
- Indicator Scoring(IS), DBN 추론, LLM Fusion 수행.  
- 결과는 `score` 카운트로 반영.  
- 위험도 시계열 그래프 제공.  

---

### 7. **Risk (위험 분석)**
- 위험 점수 기반으로 **분야별 리스크 블록(Crisis Block)** 도출.  
- 단일 사건이 아닌 **복합적 패턴**으로 위험도 정량화.  

---

### 8. **Alerts (경보)**
- 위험도가 임계치 초과 시 `alert` 발생.  
- **경보 등급**: Low / Medium / High.  
- Alerts Trend에서 집계 단위별로 시각화 확인 가능.  

---

### 9. **EventBlocks (사건 블록)**
- Trigger된 사건들을 구조화된 블록 단위로 묶음.  
- 시나리오/위험 분석의 기초 단위.  

---

### 10. **Fusion (종합 추론)**
- 사건·인물·지표 데이터를 결합해 **맥락 기반 추론** 수행.  
- 여러 위험 블록을 통합해 **위험 시그널 결론**을 생성.  

---

### 11. **Blocks**
- Crisis Block 저장소.  
- 각 블록은 사건/인물/프레임/점수/시간축을 포함.  
- 후속 시나리오 생성에 사용됨.  

---

### 12. **Scenarios (시나리오)**
- GPT 기반 자동 생성 모드 지원.  
- FAISS 기반 과거 시나리오와의 유사도 비교.  
- 전문가 피드백 입력 가능.  

---

### 13. **Ledger (원장/기록)**
- 실행 로그 및 모든 데이터 버전 관리.  
- `logs/error.log`와 연결, 오류 추적 가능.  

---

## 📂 저장 구조 (Data & Logs)

- **데이터 번들**
  ```
  data/bundles/b01/
    ├── raw.jsonl              # 수집 원본
    ├── clean.jsonl            # 전처리 결과
    ├── clean.enriched.jsonl   # 언어·메타데이터·엔티티 확장
    ├── entities.jsonl         # 인물/사건 엔티티
    ├── keyword_wc.png         # 워드클라우드 이미지
    ├── gated.jsonl            # 체크리스트 매칭
    ├── scored.jsonl           # 위험 점수화
    ├── scenarios.jsonl        # 시나리오
    └── alerts.jsonl           # 경보 로그
  ```

- **로그 & 문서**
  ```
  logs/error.log        # 에러 자동 기록
  docs/UI_Guide.md      # 본 설명서
  docs/ErrorManual.md   # 에러 매뉴얼
  ```

---

## 📊 요약표

| 메뉴 | 중심 기능 | 출력/결과 |
|------|-----------|-----------|
| Landing | 개요, 경보 추세 | 파이프라인 현황, 경보 시계열 |
| Ingest | 뉴스·SNS 수집 | ingest 카운트 |
| Normalize | 전처리 + 엔티티 추출 | clean, dedup, entities.jsonl |
| Analyze | 키워드/프레임 분석 + 워드클라우드 | keyword, frame, 워드클라우드 |
| Gate | 체크리스트 매칭 | kboost 카운트 |
| Scoring | 위험 점수화 | score 카운트, 위험도 그래프 |
| Risk | 리스크 블록 분석 | Crisis Block |
| Alerts | 경보 발령 | alert 카운트, 알림 로그 |
| EventBlocks | 사건 단위 블록화 | Event Block DB |
| Fusion | 사건·인물 종합 추론 | 위험 결론 |
| Blocks | Crisis Block 저장 | 블록 DB |
| Scenarios | 시나리오 생성/매칭 | Scenario Card |
| Ledger | 실행·에러 로그 | 기록/버전 관리 |

######

- ** ⚙️ 실행 절차 (Step-by-Step Commands)

** 가상환경 활성화

conda activate nww   # 또는 venv


- ** UTF-8 설정

$env:PYTHONUTF8=1


- ** 앱 실행

streamlit run nwwpkg/ui/app_main.py


- ** 사이드바 설정

NWW_BUNDLE 선택 (기본: b01)

모델 선택: gpt-4o-mini, gpt-4.1, gpt-3.5 등

경보 임계치 조정 (기본: 0.7)

- ** 📂 저장 구조 (Data & Logs)

데이터 번들

data/bundles/b01/
  ├── raw.jsonl              # 수집 원본
  ├── clean.jsonl            # 전처리 결과
  ├── clean.enriched.jsonl   # 언어·메타데이터 확장
  ├── gated.jsonl            # 체크리스트 매칭
  ├── scored.jsonl           # 위험 점수화
  ├── scenarios.jsonl        # 시나리오
  └── alerts.jsonl           # 경보 로그


- ** 로그 & 문서

logs/error.log        # 에러 자동 기록
docs/UI_Guide.md      # 본 설명서
docs/ErrorManual.md   # 에러 매뉴얼

- ** 🧩 확장 기능 (Extensions)

Batch Mode: 여러 문서를 동시에 처리

병렬 실행: 전처리/스코어링 속도 향상 (멀티스레딩)

Expert Feedback UI: 전문가 검토 후 DB 업데이트

시각화 강화: 시계열·네트워크·GeoJSON 지도 지원
