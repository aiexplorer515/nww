"""
scenario_builder.py
Builds scenarios from matched blocks and scoring results
"""

import logging
import json
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ScenarioBuilder:
    """Construct scenarios based on EDS blocks and scoring."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def run(self, input_path: str, output_path: str):
        """
        Build scenarios from processed blocks.

        Args:
            input_path (str): Path to blocks.jsonl
            output_path (str): Path to scenarios.jsonl
        """
        if not os.path.exists(input_path):
            logger.warning(f"No input found: {input_path}")
            return

        scenarios = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    block = json.loads(line.strip())
                    scenarios.append(self._build_scenario(block))
                except Exception as e:
                    logger.error(f"Error parsing block: {e}")

        with open(output_path, "w", encoding="utf-8") as f:
            for s in scenarios:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        logger.info(f"[ScenarioBuilder] Completed → {output_path}")

    def _build_scenario(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """Convert block into a scenario dict (basic placeholder)."""
        return {
            "id": f"scn_{block.get('id', 'unknown')}",
            "summary": f"Scenario from block {block.get('id', 'N/A')}",
            "indicators": block.get("indicators", []),
            "score": block.get("score", 0.0),
        }
