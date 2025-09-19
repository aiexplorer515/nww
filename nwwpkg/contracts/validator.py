import json, re, sys, math
from pathlib import Path
import yaml

def _get(d, path):
    cur=d
    for p in path.split("."):
        if p.endswith("]"):  # ci[0] 같은 형식
            name, idx = re.match(r"(.+)\[(\d+)\]", p).groups()
            cur = cur.get(name, [])
            cur = cur[int(idx)] if isinstance(cur, list) and len(cur)>int(idx) else None
        else:
            cur = cur.get(p) if isinstance(cur, dict) else None
        if cur is None: return None
    return cur

def _type_ok(val, typ):
    if typ=="none": return val is None
    return (typ=="str" and isinstance(val,str)) or \
           (typ=="int" and isinstance(val,int) and not isinstance(val,bool)) or \
           (typ=="float" and isinstance(val,(int,float)) and not isinstance(val,bool)) or \
           (typ=="list" and isinstance(val,list)) or \
           (typ=="dict" and isinstance(val,dict))

def _parse_types(ts):
    if isinstance(ts,list): return ts
    if isinstance(ts,str): return [t.strip() for t in ts.split("|")]
    return []

def _check_constraint(val, rule:str):
    if val is None: return True
    if rule.startswith("range="):
        lo,hi = rule[len("range="):].strip("[]").split(",")
        lo = float(lo) if lo.strip()!="-inf" else -math.inf
        hi = float(hi) if hi.strip()!="+inf" else math.inf
        return (val>=lo and val<=hi)
    if rule.startswith("minlen="): return isinstance(val,str) and len(val)>=int(rule.split("=")[1])
    if rule.startswith("len="):    return isinstance(val,(list,tuple)) and len(val)==int(rule.split("=")[1])
    if rule.startswith("enum="):   return str(val) in rule.split("=",1)[1].strip("[]").split(",")
    if rule=="iso8601":
        return isinstance(val,str) and re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", val)
    return True

def validate_file(jsonl_path: Path, spec_path: Path, max_errors=50):
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    req = spec.get("required", [])
    types = spec.get("types", {})
    cons  = spec.get("constraints", {})
    allow_extra = spec.get("allow_extra", True)

    total=0; ok=0; errors=[]
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                rec = json.loads(line)
            except Exception:
                errors.append({"line": total, "err":"invalid json"})
                continue
            fail=False
            # required
            for r in req:
                key, *ty = r.split(":")
                val = _get(rec, key)
                if val is None:
                    errors.append({"line": total, "field": key, "err":"missing"}); fail=True; continue
                if ty:
                    ok_type = any(_type_ok(val, t) for t in _parse_types(ty[0]))
                    if not ok_type:
                        errors.append({"line": total, "field": key, "err":"type", "got": type(val).__name__}); fail=True
            # typed fields
            for k, t in types.items():
                val = _get(rec, k)
                if val is None: continue
                if not any(_type_ok(val, x) for x in _parse_types(t)):
                    errors.append({"line": total, "field": k, "err":"type", "got": type(val).__name__}); fail=True
            # constraints
            for k, rule in cons.items():
                val = _get(rec, k)
                if not _check_constraint(val, rule):
                    errors.append({"line": total, "field": k, "err": f"constraint:{rule}", "value": val}); fail=True
            if not fail: ok += 1
            if len(errors)>=max_errors: break
    rate = ok/max(1,total)
    return {"file": str(jsonl_path), "spec": str(spec_path), "total": total, "ok": ok, "rate": rate, "errors": errors}
