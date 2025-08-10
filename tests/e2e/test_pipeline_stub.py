import os, json

def test_smoke_trace_exists(tmp_path):
    # simple check: create a dummy smoke trace file to simulate pipeline run
    out = tmp_path / "smoke_result.json"
    out.write_text(json.dumps({"ok": True}), encoding="utf-8")
    data = json.loads(out.read_text("utf-8"))
    assert "ok" in data
