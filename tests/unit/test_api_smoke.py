from fastapi.testclient import TestClient
from api.app.main import app

def test_health_ok():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_events_stub():
    c = TestClient(app)
    r = c.post("/v1/events", json={"id":"E_TEST_1","schema":"TEST","time":"2025-08-10T00:00:00Z"})
    assert r.status_code == 200
    assert r.json()["id"] == "E_TEST_1"
