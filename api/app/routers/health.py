from fastapi import APIRouter
router = APIRouter()

@router.get("", summary="Liveness probe")
def liveness():
    return {"status":"ok"}

@router.get("/ready", summary="Readiness probe")
def readiness():
    # Extend with checks: vector DB, TSDB, config, etc.
    return {"ready": True, "deps": {"vector_db": "stub", "tsdb": "stub"}}
