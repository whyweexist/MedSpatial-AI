from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TrainingSample:
    volume: np.ndarray
    metadata: dict[str, Any]
    provenance: dict[str, Any]


class DatasetAdapter(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> TrainingSample:
        raise NotImplementedError
