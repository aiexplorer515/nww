import sys, json, pathlib
fin, fout = map(pathlib.Path, sys.argv[1:3])
with fin.open(encoding="utf-8-sig") as f, fout.open("w", encoding="utf-8-sig") as w:
    for line in f:
        r = json.loads(line)
        conf = float(r.get("frame",{}).get("conf", 0.0))
        out = {"id": r.get("id"),
               "subscores": {"indicator": conf, "dbn": None, "llm": None},
               "fused_score": conf}
        w.write(json.dumps(out, ensure_ascii=False)+"\n")
print("scores ->", fout)
