"""Promptable segmentation provider interface."""

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field


class SegmentationPrompt(BaseModel):
    prompt_type: Literal["point", "box", "text", "slice", "correction"]
    structure: str | None = None
    points: list[tuple[float, float, float]] = Field(default_factory=list)
    box: tuple[float, float, float, float, float, float] | None = None
    slice_index: int | None = None
    previous_mask_uri: str | None = None


class SegmentationResult(BaseModel):
    provider: str
    structure: str
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class SegmenterProvider(ABC):
    name: str

    @abstractmethod
    def segment(
        self, volume: np.ndarray, prompt: SegmentationPrompt, metadata: dict[str, Any]
    ) -> tuple[np.ndarray, SegmentationResult]:
        raise NotImplementedError
