
import json, sys, os
from jsonschema import Draft7Validator
def main():
    schemas_dir = sys.argv[1] if len(sys.argv)>1 else "schemas"
    golden_dir = sys.argv[2] if len(sys.argv)>2 else "tests/golden"
    schemas = {fn: json.load(open(os.path.join(schemas_dir, fn), encoding="utf-8"))
               for fn in os.listdir(schemas_dir) if fn.endswith(".schema.json")}
    ok = True
    for g in sorted(os.listdir(golden_dir)):
        if not g.endswith(".json"): continue
        base = g.replace(".golden.json","").replace(".demo.json","")
        sc = f"{base}.schema.json"
        if sc in schemas:
            data = json.load(open(os.path.join(golden_dir, g), encoding="utf-8"))
            Draft7Validator(schemas[sc]).validate(data)
            print(f"OK: {g} ✓ against {sc}")
        else:
            print(f"SKIP: {g} (no {sc})")
    return 0 if ok else 1
if __name__ == "__main__":
    sys.exit(main())
