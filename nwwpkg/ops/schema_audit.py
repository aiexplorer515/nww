import json, sys, yaml
from pathlib import Path
from nwwpkg.contracts.validator import validate_file

def main(bundle: str):
    base = Path(f"data/bundles/{bundle}")
    reg  = yaml.safe_load(Path("schema/registry.yaml").read_text(encoding="utf-8"))
    report = []
    for c in reg["contracts"]:
        jf = base/c["file"]; sf = Path(c["spec"])
        if jf.exists():
            report.append(validate_file(jf, sf))
        else:
            report.append({"file": str(jf), "spec": str(sf), "total": 0, "ok": 0, "rate": 0.0, "errors": [{"err":"missing file"}]})
    outp = base/"contract.report.json"
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    bad = [r for r in report if r["rate"] < 0.99]
    print(f"Schema audit written -> {outp}")
    print("FAILS:", len(bad))
    sys.exit(0 if not bad else 1)

if __name__=="__main__":
    b = sys.argv[sys.argv.index("--bundle")+1]
    main(b)
