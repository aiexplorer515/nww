# nwwpkg/ui/components/sidebar_nav.py
import streamlit as st

PAGES = [
    ("Overview", "ui/app_main.py", "🌍"),
    ("1_Ingest", "ui/pages/1_Ingest.py", "📥"),
    ("2_Normalize", "ui/pages/2_Normalize.py", "🧹"),
    ("3_Analyze", "ui/pages/3_Analyze.py", "🔎"),
    ("4_Gate", "ui/pages/4_Gate.py", "🚪"),
    ("5_Scoring", "ui/pages/5_Scoring.py", "📊"),
    ("6_Fusion", "ui/pages/6_Fusion.py", "🧬"),
    ("7_Blocks", "ui/pages/7_Blocks.py", "🧱"),
    ("8_Scenarios", "ui/pages/8_Scenarios.py", "📄"),
    ("9_Alerts", "ui/pages/9_Alerts.py", "🚨"),
    ("10_EventBlocks", "ui/pages/10_EventBlocks.py", "🟩"),
    ("11_Ledger", "ui/pages/11_Ledger.py", "📜"),
    ("12_Overview", "ui/pages/12_Overview.py", "🌐"),
]

def render_sidebar_nav():
    st.sidebar.header("📌 단계 이동")
    for label, relpath, icon in PAGES:
        if st.sidebar.button(f"{icon} {label}", key=f"nav_{label}"):
            st.switch_page(relpath)
