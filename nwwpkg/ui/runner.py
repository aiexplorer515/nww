# nwwpkg/ui/runner.py
import subprocess, time, streamlit as st
from pathlib import Path

def run_cli(cmd: list[str], cwd: str = ".") -> tuple[int, str]:
    try:
        st.toast("▶ " + " ".join(cmd))
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, shell=False)
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return p.returncode, out[-4000:]  # UI에 4KB만
    except Exception as e:
        return 1, f"EXC: {e}"

def run_stage(bundle: str, stage: str):
    base = f"data/bundles/{bundle}"
    if stage == "preprocess":
        cmds = [
            ["python","-m","nwwpkg.prep.clean","--in",f"{base}/raw.jsonl","--out",f"{base}/clean.jsonl"],
            ["python","-m","nwwpkg.prep.dedupe_shingle","--in",f"{base}/clean.jsonl","--out",f"{base}/clean.dedup.jsonl","--thr","0.85"],
            ["python","-m","nwwpkg.prep.qc_report","--in",f"{base}/clean.dedup.jsonl","--out",f"{base}/prep.qc.json"],
        ]
    elif stage == "keyword":
        cmds = [
            ["python","-m","nwwpkg.prep.keyword","--in",f"{base}/clean.dedup.jsonl","--out",f"{base}/kyw.jsonl","--topk","12"],
            ["python","-m","nwwpkg.rules.lexicon_boost","--in",f"{base}/kyw.jsonl","--rules","rules/checklist.csv","--out",f"{base}/kyw_boosted.jsonl"],
        ]
    elif stage == "frame":
        cmds = [["python","-m","nwwpkg.rules.frame_classifier","--in",f"{base}/kyw_boosted.jsonl","--out",f"{base}/frames.jsonl","--on","0.45"]]
    elif stage == "score":
        cmds = [
            ["python","-m","nwwpkg.fusion.indicator_scorer","--in",f"{base}/frames.jsonl","--out",f"{base}/scores.jsonl"],
            ["python","-m","nwwpkg.fusion.fuse","--scores",f"{base}/scores.jsonl","--weights","rules/weights.yaml","--out",f"{base}/scores.jsonl"],
        ]
    elif stage == "alert":
        cmds = [["python","-m","nwwpkg.ops.alert_decider","--scores",f"{base}/scores.jsonl","--on","0.70","--off","0.55","--out",f"{base}/alerts.jsonl"]]
    else:
        return "unknown stage", ""

    logs = []
    with st.status(f"Running {stage}…", expanded=True) as s:
        for c in cmds:
            code, out = run_cli(c)
            logs.append((c, code, out))
            st.write(f"$ {' '.join(c)}"); st.code(out or "(no output)")
            if code != 0:
                s.update(state="error", label=f"{stage} failed")
                return "error", logs
        s.update(state="complete", label=f"{stage} done")
    return "ok", logs
