# Module Design – fusion/fuse.py

## 1. 목적
유사 기사를 그룹화하여 FusedArticle을 생성한다.

## 2. 입력
- `AnalysisResult` 리스트

## 3. 출력
- `FusedArticle` 객체
```json
{
  "fused_id": "F-001",
  "fused_title": "군사 긴장 고조",
  "fused_content": "여러 기사를 종합한 요약",
  "key_entities": ["군", "정부"],
  "risk_level": "high",
  "confidence_score": 0.9
}
```

## 4. 내부 로직
- Cosine similarity 기반 기사 클러스터링
- 핵심 엔티티 집계
- 종합 위험도 계산

## 5. 예외 처리
- 기사 데이터 부족 시 skip
- 클러스터링 실패 시 개별 기사 유지
