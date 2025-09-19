# Module Design – ingest/news_collector.py

## 1. 목적 (Purpose)
뉴스 소스를 수집하고 표준화된 `NewsArticle` 객체를 생성한다.

## 2. 입력 (Inputs)
- RSS feed (XML)
- NewsAPI (REST API)
- JSON 파일

## 3. 출력 (Outputs)
- `NewsArticle` 객체
```json
{
  "article_id": "A-001",
  "title": "뉴스 제목",
  "content": "뉴스 본문",
  "source": "Reuters",
  "published_date": "2025-09-14T02:30:00Z",
  "url": "https://..."
}
```

## 4. 내부 로직 (Internal Logic)
- 소스별 파서 (RSSParser, NewsAPIConnector, JSONLoader)
- 중복 기사 제거
- 타임존 정규화

## 5. 예외 처리 (Error Handling)
- 네트워크 오류 시 재시도
- JSON 파싱 실패 시 fallback
