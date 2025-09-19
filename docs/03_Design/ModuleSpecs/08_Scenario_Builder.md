# Module Design – scenario/build.py

## 1. 목적
엔티티/슬롯 채움, claim별 evidence_id 귀속, CREDO/FACT/coverage 산출.

## 2. 입력
- block_hits.jsonl, kyw_sum.jsonl, articles.norm.jsonl

## 3. 출력
- scenarios.jsonl + reports/scenario_cards.md
```json
{"scenario_id":"SCN-001","title":"군사 충돌 위험 고조","claims":[{"text":"...","evidence_id":37}],"score":{"credo":0.78,"fact":0.74,"coverage":0.81},"lang":"ko"}
```

## 4. 코드 스켈레톤
```python
class ScenarioBuilder:
    def run(self, bundle_dir: str, out_jsonl: str, out_cards_md: str) -> None: ...
```
