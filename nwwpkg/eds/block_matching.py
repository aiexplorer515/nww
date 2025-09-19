"""
EDS Block Matching
Matches analyzed news data against EDS blocks
"""

import json
from typing import List, Dict, Any
from .eds_registry import EDSRegistry
from .eds_block import EDSBlock

class EDSBlockMatcher:
    def __init__(self, registry: EDSRegistry = None):
        self.registry = registry or EDSRegistry()

    def match(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Match article indicators against registered EDS blocks
        """
        matches = []
        indicators = article.get("indicators", {})

        for block in self.registry.list_all():
            score = sum(
                indicators.get(ind, 0) * weight
                for ind, weight in block.indicators.items()
            )
            if score > 0:
                matches.append({
                    "block_id": block.block_id,
                    "block_name": block.name,
                    "score": score,
                })
        return matches

    def run(self, input_path: str, output_path: str) -> None:
        with open(input_path, "r", encoding="utf-8") as f:
            articles = [json.loads(line) for line in f]

        results = []
        for article in articles:
            results.extend(self.match(article))

        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")



