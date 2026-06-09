"""Adapter around the existing local segmentation model."""

from typing import Any

import numpy as np

from app.ai.segmentation.providers.base import SegmentationPrompt, SegmentationResult, SegmenterProvider
from app.ai.segmentation.providers.hu_fallback import HeuristicHUFallback


class LocalLightSegmenter(SegmenterProvider):
    name = "local_light"

    def __init__(self) -> None:
        self.fallback = HeuristicHUFallback()

    def segment(
        self, volume: np.ndarray, prompt: SegmentationPrompt, metadata: dict[str, Any]
    ) -> tuple[np.ndarray, SegmentationResult]:
        mask, result = self.fallback.segment(volume, prompt, metadata)
        result.provider = self.name
        result.warnings.append("Local lightweight provider used deterministic fallback because trained weights were unavailable")
        return mask, result
