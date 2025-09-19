# Module Design – score/is.py

## 1. 목적
IS(Indicator) 점수 계산: Σ(weight×hit×conf×source_rep).

## 2. 입력
- gated.jsonl, weights.yaml

## 3. 출력 (append → scores.jsonl)
```json
{"id":"a1","stage":"IS","score":0.62,"detail":{"병력 이동":0.35,"무기 배치":0.27},"rep_adj":0.95}
```

## 4. 코드 스켈레톤
```python
class ScoreIS:
    def run(self, in_path: str, scores_path: str, weights_path="config/weights.yaml") -> None: ...
```
