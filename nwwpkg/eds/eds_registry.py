"""
EDS Block Registry
Maintains a collection of registered EDS blocks
"""

from typing import Dict
from .eds_block import EDSBlock

class EDSRegistry:
    def __init__(self):
        self.blocks: Dict[str, EDSBlock] = {}

    def register(self, block: EDSBlock) -> None:
        self.blocks[block.block_id] = block

    def get(self, block_id: str) -> EDSBlock:
        return self.blocks.get(block_id)

    def list_all(self):
        return list(self.blocks.values())

    def remove(self, block_id: str) -> None:
        if block_id in self.blocks:
            del self.blocks[block_id]
