# Module Design – score/dbn.py

## 1. 목적
동일 id/엔티티/지역의 과거 시점 창(t-2,t-1,t) 기반 DBN 보정.

## 2. 입력
- scores.jsonl(이전 단계들), bundle context

## 3. 출력 (append → scores.jsonl)
- stage="DBN", score_DBN

## 4. 코드 스켈레톤
```python
class ScoreDBN:
    def run(self, bundle_dir: str, scores_path: str) -> None: ...
```
