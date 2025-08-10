
# NWW Codegen Pack v1

## 구성
- `scripts/codegen_models.py` — JSON Schema → **Pydantic(BaseModel)** 생성
- `scripts/codegen_ts.mjs` — JSON Schema → **TypeScript interfaces**
- `scripts/codegen_skeletons.py` — M01–M13 **모듈 스켈레톤(stub)** 생성

## 사용 (Windows PowerShell 예시)
```powershell
# 가상환경 (선택)
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip pydantic

# 1) 파이썬 모델 생성
python scripts\codegen_models.py --schemas schemas --out-py src\models_py --package nww_models

# 2) 모듈 스켈레톤 생성
python scripts\codegen_skeletons.py --out src\modules --models-package nww_models

# 3) (선택) 타입스크립트 인터페이스 생성
node scripts\codegen_ts.mjs --schemas schemas --out-ts src\models_ts
```
