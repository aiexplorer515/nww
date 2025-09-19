import os, json, time, subprocess, sys
from pathlib import Path

def newer(srcs, out):
    out = Path(out)
    if not out.exists(): return True
    t_out = out.stat().st_mtime
    return any(Path(s).exists() and Path(s).stat().st_mtime > t_out for s in srcs)

def run(cmd):
    print(">", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True).returncode

def main(bundle, lite=True):
    base = Path(f"data/bundles/{bundle}")
    raw = base/("raw.sample.jsonl" if lite and (base/"raw.sample.jsonl").exists() else "raw.jsonl")
    clean = base/"clean.jsonl"
    dedup = base/("clean.cheap.jsonl" if lite else "clean.dedup.jsonl")
    qc    = base/"prep.qc.json"
    kyw   = base/"kyw.jsonl"; kboost=base/"kyw_boosted.jsonl"; frames=base/"frames.jsonl"
    scores= base/"scores.jsonl"; alerts=base/"alerts.jsonl"

    # 1) clean
    if newer([raw], clean):
        run(["python","-m","nwwpkg.prep.clean","--in",str(raw),"--out",str(clean)])

    # 2) dedupe
    if lite:
        if newer([clean], dedup):
            run(["python","-m","nwwpkg.prep.dedupe_cheap","--in",str(clean),"--out",str(dedup)])
    else:
        if newer([clean], dedup):
            run(["python","-m","nwwpkg.prep.dedupe_shingle","--in",str(clean),"--out",str(dedup),"--thr","0.85"])

    # 3) qc_report
    if newer([dedup], qc):
        run(["python","-m","nwwpkg.prep.qc_report","--in",str(dedup),"--out",str(qc)])

    # 4) keyword → lex-boost → frame
    if newer([dedup], kyw):
        run(["python","-m","nwwpkg.prep.keyword","--in",str(dedup),"--out",str(kyw),"--topk","8"])
    if newer([kyw], kboost):
        run(["python","-m","nwwpkg.rules.lexicon_boost","--in",str(kyw),"--rules","rules/checklist.csv","--out",str(kboost)])
    if newer([kboost], frames):
        run(["python","-m","nwwpkg.rules.frame_classifier","--in",str(kboost),"--out",str(frames),"--on","0.45"])

    # 5) score → alert (LITE: indicator+fuse만)
    if newer([frames], scores):
        run(["python","-m","nwwpkg.fusion.indicator_scorer","--in",str(frames),"--out",str(scores)])
        run(["python","-m","nwwpkg.fusion.fuse","--scores",str(scores),"--weights","rules/weights.yaml","--out",str(scores)])
    if newer([scores], alerts):
        run(["python","-m","nwwpkg.ops.alert_decider","--scores",str(scores),"--on","0.70","--off","0.55","--out",str(alerts)])

if __name__=="__main__":
    b = sys.argv[sys.argv.index("--bundle")+1]
    lite = os.environ.get("NWW_LITE","0") == "1"
    main(b, lite=lite)
