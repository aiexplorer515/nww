
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JSON Schema -> Python (Pydantic v2) 모델 코드 생성기.
- 사용:
    python scripts/codegen_models.py --schemas schemas --out-py src/models_py --package nww_models
"""
import os, json, argparse, re
from pathlib import Path

PRIMITIVES = {
    "string": "str",
    "number": "float",
    "integer": "int",
    "boolean": "bool",
}

def to_camel(name: str) -> str:
    name = re.sub(r'[^0-9a-zA-Z]+', ' ', name).strip()
    parts = name.split()
    head = ''.join(p[:1].upper() + p[1:] for p in parts if p)
    if not head:
        head = "Model"
    if head[0].isdigit():
        head = "N" + head
    return head

def ensure_pkg(out_dir: Path, pkg: str):
    pkg_dir = out_dir / pkg
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("# generated package\n", encoding="utf-8")
    return pkg_dir

def type_of(schema, name_hint, models):
    # enums
    if isinstance(schema, dict) and "enum" in schema:
        vals = schema["enum"]
        lit = "Literal[" + ", ".join([repr(v) for v in vals]) + "]"
        return lit, set(["Literal"])
    # type can be list (nullable)
    t = schema.get("type")
    if isinstance(t, list):
        # ex: ["string","null"]
        non_null = [x for x in t if x != "null"]
        if len(non_null)==1:
            inner, imps = type_of({"type": non_null[0], **{k:v for k,v in schema.items() if k!='type'}}, name_hint, models)
            return f"Optional[{inner}]", imps | set(["Optional"])
        else:
            return "dict", set()  # fallback
    if t in PRIMITIVES:
        return PRIMITIVES[t], set()
    if t == "array":
        items = schema.get("items", {})
        inner, imps = type_of(items, name_hint+"Item", models)
        return f"List[{inner}]", imps | set(["List"])
    if t == "object":
        # Inline object -> create nested model
        title = to_camel(name_hint)
        # Avoid duplicate class names
        base_title = title
        i=2
        while title in models:
            title = f"{base_title}{i}"
            i+=1
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        fields = []
        imports = set()
        for k, v in props.items():
            pytype, imps = type_of(v, f"{title}_{k}", models)
            optional = (k not in required)
            if optional and not pytype.startswith("Optional["):
                pytype = f"Optional[{pytype}]"
                imps |= set(["Optional"])
            default = " = None" if optional else ""
            fields.append((k, pytype, default))
            imports |= imps
        models[title] = {"fields": fields, "imports": imports}
        return title, imports
    # default
    return "dict", set()

def generate_model_code(models: dict, package: str):
    # Order: as inserted
    from_imports = set(["BaseModel"])
    for m in models.values():
        from_imports |= m["imports"]
    head_imports = []
    if from_imports - set(["BaseModel"]):
        head_imports.append("from typing import " + ", ".join(sorted(from_imports - set(["BaseModel"]))))
    head_imports.append("from pydantic import BaseModel")
    code = "\n".join(head_imports) + "\n\n"
    for name, spec in models.items():
        code += f"class {name}(BaseModel):\n"
        if not spec["fields"]:
            code += "    pass\n\n"
            continue
        for k, t, d in spec["fields"]:
            code += f"    {k}: {t}{d}\n"
        code += "\n"
    return code

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemas", default="schemas")
    ap.add_argument("--out-py", default="src/models_py")
    ap.add_argument("--package", default="nww_models")
    args = ap.parse_args()

    schemas_dir = Path(args.schemas)
    out_dir = Path(args.out_py)
    pkg_dir = ensure_pkg(out_dir, args.package)

    for fn in sorted(os.listdir(schemas_dir)):
        if not fn.endswith(".schema.json"):
            continue
        schema = json.load(open(schemas_dir / fn, encoding="utf-8"))
        title = schema.get("title") or fn.replace(".schema.json","")
        main_name = to_camel(title)
        models = {}
        # top-level object only
        t = schema.get("type")
        if t != "object":
            # wrap as object with single field 'value'
            models[main_name] = {"fields":[("value","str","")],"imports":set()}
        else:
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            fields = []
            imports = set()
            for k, v in props.items():
                pytype, imps = type_of(v, f"{main_name}_{k}", models)
                optional = (k not in required)
                if optional and not pytype.startswith("Optional["):
                    pytype = f"Optional[{pytype}]"
                    imps |= set(["Optional"])
                default = " = None" if optional else ""
                fields.append((k, pytype, default))
                imports |= imps
            models[main_name] = {"fields": fields, "imports": imports}

        code = generate_model_code(models, args.package)
        out_path = pkg_dir / f"{fn.replace('.schema.json','')}.py"
        out_path.write_text(code, encoding="utf-8")
        print("generated:", out_path)

    # write package __init__ exports
    init_lines = ["# generated exports\n"]
    for fn in sorted(os.listdir(pkg_dir)):
        if fn.endswith(".py") and fn != "__init__.py":
            mod = fn[:-3]
            init_lines.append(f"from .{mod} import *\n")
    (pkg_dir / "__init__.py").write_text("".join(init_lines), encoding="utf-8")
    print("done:", pkg_dir)

if __name__ == "__main__":
    main()
