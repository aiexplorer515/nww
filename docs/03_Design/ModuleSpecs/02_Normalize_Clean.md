# Module Design – preprocess/normalize.py

## 1. 목적
토큰/문장분리, 소문자화, 표기 정규화, 언어감지 확정, 중복제거(hash)로 정제 텍스트 생성.

## 2. 입력
- articles.jsonl

## 3. 출력
- articles.norm.jsonl (+ logs/normalize.log)
- 필드: id, ts, lang_final, hash, norm_text, segments[]

## 4. 내부 로직
- 언어 혼용 문단 분할, 오타 일부 정정, 숫자/단위 정규화
- near-duplicate 제거(Minhash/SimHash)

## 5. 예외 처리
- 비어있는 본문, 극단 길이 본문 처리

## 6. 코드 스켈레톤
```python
class Normalizer:
    def run(self, in_path: str, out_path: str, log_path: str) -> None:
        ...
```
