"""
EDS Block Definition
Defines the structure of an Expert Data System (EDS) block
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class EDSBlock:
    block_id: str
    name: str
    domain: str
    indicators: Dict[str, float]  # indicator_name → weight
    description: str = ""
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "name": self.name,
            "domain": self.domain,
            "indicators": self.indicators,
            "description": self.description,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
