# Module Design – rules/gating.py

## 1. 목적
체크리스트 매칭(정규식/패턴/스코어) + z-score/로지스틱 사전 필터.

## 2. 입력
- kyw_sum.jsonl
- config/weights.yaml (indicator patterns)

## 3. 출력 (gated.jsonl)
```json
{"id":"a1","hits":[{"indicator":"병력 이동","val":1,"conf":0.92,"span":[12,34]}],"z":0.84,"logit":0.76,"rep_adj":0.95}
```

## 4. 내부 로직
- 인디케이터 룰 매칭, 소스 평판/언어품질 보정
- 오경보 억제 규칙

## 5. 코드 스켈레톤
```python
class Gating:
    def run(self, in_path: str, out_path: str, weights_path="config/weights.yaml") -> None: ...
```
