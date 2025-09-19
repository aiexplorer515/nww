# Development Guide – Windows IDE + Cursor (NWW Project)

## 1. 환경 준비 (Windows IDE + Cursor)
- **IDE 추천**: Cursor (VSCode 기반), PyCharm, VSCode
- **필수 설치**:
  - Python 3.11+
  - Git
  - pip + 가상환경 (`venv`)
  - Cursor IDE (AI 코드 자동생성 지원)

## 폴더 구조 예시
```
nww/
├── docs/
│   └── 03_Design/ModuleSpecs/
│       ├── ingest_news_collector.md
│       ├── judge_llm_judge.md
│       ├── ...
├── nwwpkg/
│   ├── ingest/
│   ├── judge/
│   ├── fusion/
│   ├── rules/
│   ├── scenario/
│   ├── ops/
│   └── ui/
├── data/
└── README.md
```

---

## 2. Markdown → Python 코드 변환 규칙
각 `.md` 파일에는 공통 섹션이 있음:
- 목적 (Purpose)
- 입력 (Inputs)
- 출력 (Outputs)
- 내부 로직 (Internal Logic)
- 예외 처리 (Error Handling)
- 코드 스켈레톤 (Code Skeleton)

👉 Cursor 또는 변환 스크립트(`tools/md_to_py.py`)를 이용해 자동 변환.

---

## 3. 자동 코드 생성기 설계 (md → py)

### 실행 명령 (PowerShell 기준)
```powershell
python tools/md_to_py.py --input docs/03_Design/ModuleSpecs --output nwwpkg
```

### 변환 스크립트 (tools/md_to_py.py)
```python
import os
import re
import argparse

def extract_code(md_text: str) -> str:
    code_blocks = re.findall(r"```python(.*?)```", md_text, re.S)
    return "\n\n".join([block.strip() for block in code_blocks])

def md_to_py(md_path: str, py_path: str):
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    code = extract_code(md_text)
    if not code:
        print(f"[WARN] No code found in {md_path}")
        return
    os.makedirs(os.path.dirname(py_path), exist_ok=True)
    with open(py_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated from Markdown spec\n\n")
        f.write(code)
    print(f"[OK] Generated {py_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for root, _, files in os.walk(args.input):
        for file in files:
            if file.endswith(".md"):
                md_path = os.path.join(root, file)
                module_name = os.path.splitext(file)[0].replace("_", "/")
                py_path = os.path.join(args.output, module_name + ".py")
                md_to_py(md_path, py_path)
```

---

## 4. 워크플로우 (Cursor 활용)
1. **문서 작성**: `docs/03_Design/ModuleSpecs/*.md`
2. **코드 생성**: Cursor에서 `.md` 열고 "Generate Python File" 실행
3. **자동 변환**: `python tools/md_to_py.py`
4. **코드 검증**: `pytest tests/` 실행 → Cursor로 실패 수정

---

## 5. 운영 가이드 (Windows)
```powershell
cd nww
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# 코드 자동 생성
python tools/md_to_py.py --input docs/03_Design/ModuleSpecs --output nwwpkg

# 실행
streamlit run nwwpkg/ui/app.py
```

### requirements.txt 예시
```
streamlit
openai
requests
pyyaml
scikit-learn
plotly
```

---

## ✅ 요약
- `docs/03_Design/ModuleSpecs/*.md` → 설계/코드 스켈레톤 포함
- `tools/md_to_py.py` → Markdown → Python 자동 변환
- Cursor IDE → 코드 보완 및 테스트 자동 수정
- Windows venv, Streamlit 실행 절차 포함
