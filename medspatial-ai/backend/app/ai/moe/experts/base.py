"""Expert provider contract."""

from abc import ABC, abstractmethod
from typing import Any


class Expert(ABC):
    name: str
    tasks: frozenset[str]
    modalities: frozenset[str]
    regions: frozenset[str]
    estimated_memory_mb: int = 128

    @abstractmethod
    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
