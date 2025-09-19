# Module Design – rules/indicator_scorer.py

## 1. 목적
전문가 체크리스트(EDS)를 기반으로 위험 점수를 계산한다.

## 2. 입력
- `AnalysisResult`

## 3. 출력
- float (위험 점수: 0~1)

## 4. 내부 로직
- weights.yaml 로드
- 체크리스트 이벤트 매칭
- 가중치 합산 → risk_score 반환

## 5. 예외 처리
- config 파일 미존재 시 default weight 적용
