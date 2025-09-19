# Module Design – fusion/fuse.py

## 1. 목적
IS/DBN/LLM 가중합 → 보정 → 신뢰구간 → EMA/Hysteresis 로 최종 fused 점수 생성.

## 2. 입력
- scores.jsonl (IS/DBN/LLM)

## 3. 출력 (scores.jsonl, stage="FUSE")
```json
{"id":"a1","stage":"FUSE","fused":0.71,"p_calib":0.68,"ci":[0.56,0.82],"ema":0.69,"state":"elevated"}
```

## 4. 코드 스켈레톤
```python
class FusionPipeline:
    def run(self, scores_path: str, domain_weights: dict, alpha: float=0.9, up=0.70, down=0.60) -> None: ...
```
