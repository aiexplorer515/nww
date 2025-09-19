"""
Ledger - simple audit log for pipeline execution
"""

import os
import json
from datetime import datetime

class LedgerOps:
    def __init__(self, bundle_dir: str):
        self.bundle_dir = bundle_dir
        self.ledger_path = os.path.join(bundle_dir, "ledger.jsonl")

    def log_event(self, step: str, description: str, status: str = "Completed"):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "step": step,
            "description": description,
            "status": status
        }
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry
