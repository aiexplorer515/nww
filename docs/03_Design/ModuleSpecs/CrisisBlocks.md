# Module Design – block/crisis_block_manager.py

## 1. 목적
기사와 점수화 결과를 JSONL 포맷 위기블록으로 저장한다.

## 2. 입력
- 기사 ID + 점수화 결과

## 3. 출력
- `CrisisBlock` (JSONL 저장)
```json
{
  "block_id": "B-20250914-001",
  "article_id": "A-001",
  "scores": {"military_movement": 0.4, "diplomatic_breakdown": 0.3},
  "risk_score": 0.7,
  "timestamp": "2025-09-14T03:00:00Z"
}
```

## 4. 내부 로직
- 블록 ID 생성
- JSONL append 저장
- 중복 방지 (hash 체크)

## 5. 예외 처리
- 파일 쓰기 실패 시 로그 기록 후 재시도
