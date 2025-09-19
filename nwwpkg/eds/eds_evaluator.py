"""
EDS Block Evaluator
Evaluates effectiveness of block matching results
"""

import json
from typing import List, Dict, Any

class EDSEvaluator:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def evaluate(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter matches based on score threshold
        """
        return [m for m in matches if m["score"] >= self.threshold]

    def run(self, input_path: str, output_path: str) -> None:
        with open(input_path, "r", encoding="utf-8") as f:
            matches = [json.loads(line) for line in f]

        filtered = self.evaluate(matches)

        with open(output_path, "w", encoding="utf-8") as f:
            for r in filtered:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
