@echo off
REM ====================================
REM NWW 완전 자동 실행 스크립트 (Windows)
REM ====================================

echo [1] 가상환경 확인/생성...
if not exist venv (
    python -m venv venv
)

call venv\Scripts\activate

echo [2] 의존성 설치...
pip install -r requirements.txt

echo [3] 패키지 테스트...
python test_nww.py

echo [4] Streamlit UI 실행...
python launch_ui.py
