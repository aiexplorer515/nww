# Module Design – eds/match.py

## 1. 목적
EDS 블록(JSON v1) 룰/슬롯과 매칭하여 coverage/에스컬레이션(delta) 계산.

## 2. 입력
- scores.jsonl(FUSE), config/eds_blocks.json

## 3. 출력 (block_hits.jsonl)
```json
{"id":"a1","block_id":"MIL-ESC-01","slot_coverage":{"actors.primary":1,"location":0.8},"delta":0.06,"block_ver":"1.0","rule_hash":"..."} 
```

## 4. 코드 스켈레톤
```python
class EDSMatcher:
    def run(self, bundle_dir: str, out_path: str, blocks_path="config/eds_blocks.json") -> None: ...
```
