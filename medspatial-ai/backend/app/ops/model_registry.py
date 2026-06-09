import json
from pathlib import Path


class ModelRegistry:
    def __init__(self, path: str = "../models/registry.json") -> None:
        self.path = Path(path)

    def list(self) -> dict:
        if not self.path.exists():
            return {"architecture": "AWM-JGEM", "models": []}
        return json.loads(self.path.read_text(encoding="utf-8"))
