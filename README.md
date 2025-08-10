# NWW API + pytest Pack (v1)

## 설치
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip -r requirements.txt
```

## API 실행
```powershell
# PowerShell
.un_api.ps1
# 또는
uvicorn api.app.main:app --reload --port 8080
```

## 엔드포인트
- `GET /health` → { status: "ok" }
- `GET /health/ready` → readiness 정보 (stub)
- `POST /v1/events` → 이벤트 수신 stub
- `POST /graphql` (strawberry 설치 시) → GraphQL

## 테스트 실행
```powershell
python -m pip install pytest httpx
pytest
```
