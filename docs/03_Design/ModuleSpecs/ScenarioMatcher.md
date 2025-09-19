# Module Design – scenario/scenario_matcher.py

## 1. 목적
위기블록을 시나리오 패턴과 매칭하여 시나리오 카드를 생성한다.

## 2. 입력
- `CrisisBlock` 리스트
- `scenarios.yaml`

## 3. 출력
- `ScenarioCard`
```json
{
  "scenario_id": "SCN-001",
  "title": "군사 충돌 위험 고조",
  "description": "국경 병력 이동과 외교 결렬로 충돌 가능성이 커지고 있음",
  "risk_level": "high",
  "related_entities": ["국방부", "인접국 정부"],
  "timestamp": "2025-09-14T03:10:00Z"
}
```

## 4. 내부 로직
- scenarios.yaml 로드
- 패턴 매칭 수행
- 시나리오 카드 생성

## 5. 예외 처리
- 시나리오 매칭 실패 시 "Unclassified Scenario" 반환
