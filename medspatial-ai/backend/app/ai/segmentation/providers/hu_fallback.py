"""Deterministic HU-based segmentation fallback."""

from typing import Any

import numpy as np
from scipy import ndimage

from app.ai.segmentation.providers.base import SegmentationPrompt, SegmentationResult, SegmenterProvider


class HeuristicHUFallback(SegmenterProvider):
    name = "heuristic_hu"
    RANGES = {
        "lung": (-950, -200), "air": (-1200, -500), "fat": (-190, -30),
        "soft_tissue": (-100, 200), "bone": (200, 3071),
        "vessel": (120, 500), "bone_marrow": (-100, 300),
    }

    def segment(
        self, volume: np.ndarray, prompt: SegmentationPrompt, metadata: dict[str, Any]
    ) -> tuple[np.ndarray, SegmentationResult]:
        key = (prompt.structure or "soft_tissue").lower()
        bounds = next((value for name, value in self.RANGES.items() if name in key), self.RANGES["soft_tissue"])
        mask = (volume >= bounds[0]) & (volume <= bounds[1])
        mask = ndimage.binary_opening(mask, iterations=1)
        mask = ndimage.binary_closing(mask, iterations=1).astype(np.uint8)
        return mask, SegmentationResult(
            provider=self.name,
            structure=prompt.structure or "soft_tissue",
            confidence=0.35,
            uncertainty=0.65,
            warnings=["Heuristic fallback is not a clinically validated segmentation"],
        )
