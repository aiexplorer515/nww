# NWW Test Bundle (b01)

이 번들은 **MVP 파이프라인 점검**을 위한 최소 동작/평가용 데이터입니다.

## 구성
- `data/b01/clean.jsonl` – 기사 원본 (120건, 90일 범위, ko/en)
- `data/b01/alerts_timeseries.csv` – 경보 시계열(90일)
- `data/gold/articles_labeled.jsonl` – 라벨 60건(프레임/경보)
- `data/dicts/source_reputation.csv` – 소스 평판 시드
- `data/dicts/frame_lexicon.json` – 프레임 사전 시드
- `tests/regression/inputs.txt` – 회귀 테스트 URL 50건
- `schema/*.json` – 스키마 정의

## 빠른 사용(예시)
```bash
# 스키마 검증 예: (프로젝트 내 validate 스크립트 가정)
python tools/validate_schema.py --file data/b01/clean.jsonl --schema schema/articles_v1.json
python tools/validate_schema.py --file data/gold/articles_labeled.jsonl --schema schema/articles_labeled_v1.json

# 대시보드 실행(프로젝트 기준)
streamlit run nwwpkg/ui/app_main.py
```
