# Module Design – ingest/extract.py

## 1. 목적
URL/File에서 HTML을 수집하고 Readability로 본문/메타를 추출하여 `articles.jsonl`을 생성한다.

## 2. 입력
- data/bundles/<BUNDLE>/sources.csv | urls.txt | raw/*.html
- config/sources.yaml (headers, rate limit, retries)

## 3. 출력 (articles.jsonl, UTF-8)
```json
{"id":"a1","ts":"2025-09-10T12:34:00Z","title":"...","text":"...","domain":"military|diplomacy|economy","region":"X","source":"url|file","byline":"...","url":"...","lang":"ko|en"}
```

## 4. 내부 로직
- HTTP 요청(캐시, 재시도, 백오프), Robots/RateLimit 준수
- HTML → Readability → 본문/메타(OG, byline, ts, region)
- 메타 누락 보완, 언어 초기값 추정

## 5. 예외 처리
- 4xx/5xx 재시도, 타임아웃, 깨진 HTML 복원

## 6. 코드 스켈레톤
```python
class Extractor:
    def __init__(self, bundle_dir: str, cfg_path="config/sources.yaml"):
        self.bundle = bundle_dir
        self.cfg = self._load_cfg(cfg_path)

    def run(self, out_path: str) -> None:
        sources = self._load_sources()
        with open(out_path, "w", encoding="utf-8") as out:
            for src in sources:
                html = self._fetch(src)
                meta, text = self._readability(html, src)
                rec = self._to_record(meta, text, src)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _fetch(self, src): ...
    def _readability(self, html, src): ...
    def _to_record(self, meta, text, src): ...
    def _load_sources(self): ...
    def _load_cfg(self, path): ...
```
