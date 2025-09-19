# Module Design – judge/llm_judge.py

## 1. 목적
LLM 분석을 통해 기사에서 감정, 중요도, 핵심 엔티티, 위험도를 추출한다.

## 2. 입력
- 정제된 기사 본문 (string)

## 3. 출력
- `AnalysisResult` 객체
```json
{
  "sentiment_score": -0.3,
  "importance_score": 0.7,
  "category": "military",
  "key_entities": ["국방부", "인접국"],
  "summary": "국경 병력 이동",
  "risk_level": "high",
  "confidence": 0.8
}
```

## 4. 내부 로직
- LLM(OpenAI API) 호출
- Rule-based fallback
- Confidence 산출

## 5. 예외 처리
- API 호출 실패 시 재시도
- LLM 응답 실패 시 규칙 기반 분석 적용
