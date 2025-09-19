import json
import os
from datetime import datetime

def record_ledger(out_dir="data/bundles/sample"):
    """Record pipeline steps into ledger"""
    os.makedirs(os.path.join(out_dir, "ledger"), exist_ok=True)
    record = {"ts": datetime.utcnow().isoformat(), "status": "pipeline completed"}
    path = os.path.join(out_dir, "ledger", f"{record['ts']}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
