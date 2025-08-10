import json, os, glob
from jsonschema import Draft7Validator

def load_schemas(sdir):
    return {fn: json.load(open(os.path.join(sdir, fn), encoding="utf-8"))
            for fn in os.listdir(sdir) if fn.endswith(".schema.json")}

def test_golden_conform_to_schema(repo_paths):
    sdir, gdir = repo_paths["schemas"], repo_paths["golden"]
    schemas = load_schemas(sdir)
    jsons = glob.glob(os.path.join(gdir, "*.json"))
    assert jsons, "no golden files found"
    for g in jsons:
        base = os.path.basename(g).replace(".golden.json","").replace(".demo.json","")
        sc = os.path.join(sdir, f"{base}.schema.json")
        assert os.path.exists(sc), f"missing schema for {g}"
        Draft7Validator(json.load(open(sc, encoding="utf-8"))).validate(
            json.load(open(g, encoding="utf-8"))
        )
