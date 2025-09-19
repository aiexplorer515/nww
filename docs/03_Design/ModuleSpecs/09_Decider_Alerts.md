# Module Design – ops/decider.py

## 1. 목적
규칙: fused≥0.70 & ci_low≥0.55 & evidence≥2 → ALERT, intl_flag 판단.

## 2. 입력
- scores.jsonl(FUSE), scenarios.jsonl

## 3. 출력 (alerts.jsonl)
```json
{"id":"a1","alert":"ALERT","intl_flag":true,"why":["fused≥0.70","ci_low≥0.55","evidence≥2"],"ts":"2025-09-14T02:10:00Z"}
```

## 4. 코드 스켈레톤
```python
class Decider:
    def run(self, bundle_dir: str, out_path: str) -> None: ...
```
