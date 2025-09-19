# Module Design – audit/ledger.py

## 1. 목적
각 단계 입력/출력 해시, 파라미터, seed, ver, prev_hash 기록.

## 2. 입력
- 단계별 입출력/파라미터 스냅샷

## 3. 출력
- ledger/YYYY-MM-DDTHHMMSS.jsonl
```json
{"step":"FUSE","hash_in":"...","hash_out":"...","params":{"w_IS":0.5},"seed":123,"ver":"1.2.0","prev_hash":"..."}
```

## 4. 코드 스켈레톤
```python
class Ledger:
    def append(self, bundle_dir: str, step: str, params: dict, in_hash: str, out_hash: str, seed: int, ver: str, prev_hash: str) -> None: ...
```
