import yaml, shlex
from pathlib import Path
import streamlit as st
from .runner import run_cli

def _fmt(cmd: list[str], bundle: str) -> list[str]:
    base = f"data/bundles/{bundle}"
    repl = {
        "{raw}":      f"{base}/raw.jsonl",
        "{clean}":    f"{base}/clean.jsonl",
        "{dedup}":    f"{base}/clean.dedup.jsonl",
        "{kyw}":      f"{base}/kyw.jsonl",
        "{kboost}":   f"{base}/kyw_boosted.jsonl",
        "{frames}":   f"{base}/frames.jsonl",
        "{scores}":   f"{base}/scores.jsonl",
        "{alerts}":   f"{base}/alerts.jsonl",
    }
    return [repl.get(x, x) for x in cmd]

def ext_panel(bundle: str, registry_path="ui/plugin_registry.yaml"):
    st.subheader("🧩 Extensibility (Plugin Registry)")
    p = Path(registry_path)
    if not p.exists():
        st.warning("plugin_registry.yaml이 없습니다.")
        return
    reg = yaml.safe_load(p.read_text(encoding="utf-8"))
    for plg in reg.get("plugins", []):
        enabled = plg.get("enabled", False)
        with st.container(border=True):
            st.markdown(f"**{plg['name']}** · _{plg['id']}_ · stage=`{plg['stage']}` · {'✅enabled' if enabled else '⛔disabled'}")
            st.caption("tags: " + ", ".join(plg.get("tags", [])))
            c1, c2, c3 = st.columns([2,1,1])
            c1.code(" ".join(plg["entry"]), language="bash")
            if enabled and c2.button("Run", key=f"run_{plg['id']}"):
                cmd = _fmt(plg["entry"], bundle)
                code, out = run_cli(cmd)
                st.code(out or "(no output)")
            toggle = c3.toggle("Enable", value=enabled, key=f"en_{plg['id']}")
            # UI에서 토글만, 실제 파일 반영은 별도 save 버튼(필요 시 구현)
