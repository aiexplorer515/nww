"""
AlertDecider - decide whether to trigger alerts based on scores
"""

import os
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class AlertDecider:
    def __init__(self, bundle_dir: str, threshold: float = 0.7, hysteresis: float = 0.1):
        self.bundle_dir = bundle_dir
        self.threshold = threshold
        self.hysteresis = hysteresis
        self.alerts_path = os.path.join(bundle_dir, "alerts.jsonl")

    def run(self, scores_file: str):
        """
        Decide alerts based on scores file.
        Saves results to alerts.jsonl
        """
        if not os.path.exists(scores_file):
            logger.warning(f"No scores file found: {scores_file}")
            return []

        alerts = []
        with open(scores_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    score = record.get("score", 0)
                    if score >= self.threshold:
                        alerts.append({
                            "id": record.get("id"),
                            "score": score,
                            "stage": record.get("stage"),
                            "region": record.get("region", "Unknown"),
                            "domain": record.get("domain", "Unknown"),
                            "status": "ALERT"
                        })
                except Exception as e:
                    logger.error(f"Error parsing score line: {e}")
                    continue

        if alerts:
            with open(self.alerts_path, "a", encoding="utf-8") as f:
                for alert in alerts:
                    f.write(json.dumps(alert, ensure_ascii=False) + "\n")

        logger.info(f"[AlertDecider] Generated {len(alerts)} alerts")
        return alerts
