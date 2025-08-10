
import json, os
from jsonschema import Draft7Validator
def main():
    root = os.getcwd()
    schemas = os.path.join(root, "schemas")
    seeds = os.path.join(root, "seeds", "tariff")
    assert os.path.isdir(schemas), "schemas/ not found"
    assert os.path.isdir(seeds), "seeds/tariff not found"
    fpd_schema = json.load(open(os.path.join(schemas, "fpd.schema.json"), encoding="utf-8"))
    esd_schema = json.load(open(os.path.join(schemas, "esd.schema.json"), encoding="utf-8"))
    ipd_schema = json.load(open(os.path.join(schemas, "ipd.schema.json"), encoding="utf-8"))
    for fn in os.listdir(os.path.join(seeds, "fpd")):
        Draft7Validator(fpd_schema).validate(json.load(open(os.path.join(seeds, "fpd", fn), encoding="utf-8")))
        print("OK FPD:", fn)
    for fn in os.listdir(os.path.join(seeds, "esd")):
        Draft7Validator(esd_schema).validate(json.load(open(os.path.join(seeds, "esd", fn), encoding="utf-8")))
        print("OK ESD:", fn)
    for fn in os.listdir(os.path.join(seeds, "ipd")):
        Draft7Validator(ipd_schema).validate(json.load(open(os.path.join(seeds, "ipd", fn), encoding="utf-8")))
        print("OK IPD:", fn)
    print("All seeds validated ✓")
if __name__ == "__main__":
    main()
