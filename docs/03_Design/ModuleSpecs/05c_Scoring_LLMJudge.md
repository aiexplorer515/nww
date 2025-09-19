# Module Design – judge/llm.py

## 1. 목적
증거귀속 강제(no_new_facts), 상위 k문장 evidence만 투입해 LLM Judge 점수 산출.

## 2. 입력
- articles.norm.jsonl, kyw_sum.jsonl (span 기반 evidence)
- 상위 evidence 선택 로직

## 3. 출력 (append → scores.jsonl)
```json
{"id":"a1","stage":"LLM","score":0.58,"rationale":"...","evidence_ids":[12,37],"contra":false}
```

## 4. 코드 스켈레톤
```python
class LLMJudge:
    def __init__(self, api_key_env="OPENAI_API_KEY"): ...
    def run(self, bundle_dir: str, scores_path: str, top_k: int = 5) -> None: ...
```
