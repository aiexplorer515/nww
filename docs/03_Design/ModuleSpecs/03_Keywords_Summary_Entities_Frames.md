# Module Design – analyze/tag.py

## 1. 목적
키프레이즈, 압축요약, NER/행위자, 프레임 태깅(LSL/ESD + LLM 보조) 수행.

## 2. 입력
- articles.norm.jsonl

## 3. 출력 (kyw_sum.jsonl)
```json
{"id":"a1","kw":["병력 이동","포병"],"summary":"...","actors":["국가A군"],"frames":["강경 대응"],"span_evidence":[[12,34],[56,78]]}
```

## 4. 내부 로직
- 키워드 추출 + 요약(압축)
- NER/행위자 식별
- 프레임 라벨링 + 신뢰도(frame_conf)
- 문장단위 span 증거 저장

## 5. 예외 처리
- 긴 문서 샘플링, 다국어 혼합

## 6. 코드 스켈레톤
```python
class Tagger:
    def run(self, in_path: str, out_path: str) -> None: ...
```
