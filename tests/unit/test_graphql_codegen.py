import os, subprocess, sys, shutil, pathlib

def test_codegen_imports(tmp_path):
    # simulate repo with SDL
    root = tmp_path
    (root/"api").mkdir()
    (root/"api"/"NWW.graphql").write_text("type Query { ok: Boolean, ping: String }", encoding="utf-8")
    (root/"scripts").mkdir()
    # copy codegen
    pack_root = pathlib.Path(__file__).resolve().parents[2]
    src = pack_root / "scripts" / "codegen_graphql.py"
    dst = root / "scripts" / "codegen_graphql.py"
    dst.write_text(src.read_text("utf-8"), encoding="utf-8")

    # run codegen
    subprocess.check_call([sys.executable, str(dst), "--sdl", str(root/"api"/"NWW.graphql"), "--out", str(root/"api"/"graphql_gen")])
    assert (root/"api"/"graphql_gen"/"schema.py").exists()

    # try importing generated schema
    sys.path.insert(0, str(root))
    mod = __import__("api.graphql_gen.schema", fromlist=["schema"])
    assert hasattr(mod, "schema")
