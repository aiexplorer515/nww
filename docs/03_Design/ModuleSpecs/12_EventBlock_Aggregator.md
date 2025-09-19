# 12. 사건 누적 위기블록 분석 (Event Block Aggregator)

## 목적 (Purpose)
- 개별 기사 단위 위기블록을 사건(Event) 단위로 누적/집계
- 시계열·슬롯·지표별 누적 위험도 곡선 산출
- 사건별 연관 블록(articles → blocks → event) 관계 구축

## 입력 (Inputs)
- `block_hits.jsonl` (단일 기사 기반 블록 매칭 결과)
- `scores.jsonl` (FUSE 단계 점수 포함)
- `articles.norm.jsonl` (참조 메타, region/actors)

## 출력 (Outputs)
- `event_blocks.jsonl`
```json
{
  "event_id": "EVT-20250914-001",
  "block_ids": ["MIL-ESC-01","DIP-NEG-02"],
  "articles": ["a1","a2","a3"],
  "actors": ["국가A군","국가B정부"],
  "location": "국경지역",
  "time_window": ["2025-09-10","2025-09-14"],
  "risk_curve": [0.42,0.58,0.71],
  "current_risk": 0.71,
  "delta": 0.06,
  "state": "elevated"
}
```

## 내부 로직 (Internal Logic)
1. Event ID 생성 (`EVT-<date>-<seq>`)
2. 동일 actors/location/time_window 기준으로 기사·블록 클러스터링
3. 각 블록 위험도 점수를 시간순으로 누적하여 `risk_curve` 생성
4. delta/EMA/Hysteresis 반영하여 사건 단위 `state` 업데이트
5. 관련 기사·블록·스코어 해시 참조 저장

## 예외 처리 (Error Handling)
- 슬롯 불완전 시 coverage<0.5는 tentative_event로 표시
- 시간/지역 불명확한 경우 fallback = 기사 ts/region

## 코드 스켈레톤 (Code Skeleton)
```python
import json
from datetime import datetime, timedelta

class EventBlockAggregator:
    def __init__(self, window_days: int = 5):
        self.window_days = window_days

    def run(self, bundle_dir: str, out_path: str) -> None:
        blocks = self._load_blocks(bundle_dir)
        events = self._aggregate(blocks)
        with open(out_path, "w", encoding="utf-8") as out:
            for evt in events:
                out.write(json.dumps(evt, ensure_ascii=False) + "\n")

    def _load_blocks(self, bundle_dir: str):
        # load block_hits.jsonl and scores.jsonl
        return []

    def _aggregate(self, blocks: list) -> list:
        # cluster by actor/location/time_window
        return []

    def _risk_curve(self, scores: list) -> list:
        # build risk curve with EMA/hysteresis
        return []
```

## UI 반영 (Streamlit)
- Blocks 탭: 단일 기사 블록 보기
- Events 탭 (신규):
  - 사건별 누적 위험도 그래프
  - 관련 기사/블록/시나리오 하이라이트
