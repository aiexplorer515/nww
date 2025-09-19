# Module Design – ui/app.py

## 1. 목적
Streamlit UI: 원클릭 Run MVP(1~10), 탭 기반 가시화/내보내기(STIX 포함).

## 2. 탭
Overview / Ingest / Scoring / Timeline / Blocks / Scenarios / Artifacts / Ledger

## 3. 기능
- 상태 표시(스피너/토스트), 설정 편집(threshold/alpha/ema/hysteresis/weights)
- 국제공조 패키지 내보내기(JSON/BRIEF/ZIP/STIX)

## 4. 코드 스켈레톤
```python
def main():
    import streamlit as st
    st.set_page_config(page_title="NWW Dashboard", layout="wide")
    # tabs, actions, run-all button
    ...

if __name__ == "__main__":
    main()
```
