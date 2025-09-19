import json, sys
from pathlib import Path

def migrate_frames(p: Path):
    tmp = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            # 과거형: frame_label/ frame_conf → 새형: frame:{label,conf}
            if "frame_label" in r or "frame_conf" in r:
                r["frame"] = r.get("frame", {})
                if "frame_label" in r: r["frame"]["label"] = r.pop("frame_label")
                if "frame_conf"  in r: r["frame"]["conf"]  = float(r.pop("frame_conf"))
            tmp.append(r)
    p.write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in tmp), encoding="utf-8")

def migrate_alerts(p: Path):
    tmp=[]
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if isinstance(r.get("ci"), dict):  # {low:.., high:..} → [low, high]
                ci = r["ci"]; r["ci"] = [float(ci.get("low",0.0)), float(ci.get("high",1.0))]
            if isinstance(r.get("fused_score"), str):
                try: r["fused_score"] = float(r["fused_score"])
                except: pass
            tmp.append(r)
    p.write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in tmp), encoding="utf-8")

if __name__=="__main__":
    base = Path(sys.argv[sys.argv.index("--bundle")+1]); 
    f1=base/"frames.jsonl"; f2=base/"alerts.jsonl"
    if f1.exists(): migrate_frames(f1)
    if f2.exists(): migrate_alerts(f2)
    print("migrated.")
