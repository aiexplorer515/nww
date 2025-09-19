# Module Design – ui/app.py

## 1. 목적
Streamlit 기반 대시보드 제공

## 2. 입력
- 파일 업로드 (.json, .csv)
- 내부 DB (blocks/, scores/)

## 3. 출력
- 대시보드 UI (웹 브라우저)

## 4. 주요 기능
- Analytics: 위험도/감정/카테고리 차트
- Original Articles: 기사별 분석 카드
- Fused Articles: 융합 기사
- Crisis Blocks: 블록별 점수 뷰어
- Scenarios: 시나리오 카드
- Settings: API 키, 가중치 조정

## 5. 예외 처리
- 잘못된 입력 파일 → 경고 메시지
- API Key 미입력 시 → 제한 모드 실행
