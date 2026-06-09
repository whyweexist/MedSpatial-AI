"""Optional MedSAM2 adapter that never triggers downloads implicitly."""

from typing import Any

import numpy as np

from app.ai.segmentation.providers.base import SegmentationPrompt, SegmentationResult, SegmenterProvider


class MedSAM2Adapter(SegmenterProvider):
    name = "medsam2"

    def __init__(self, provider: Any | None = None) -> None:
        self.provider = provider

    def segment(
        self, volume: np.ndarray, prompt: SegmentationPrompt, metadata: dict[str, Any]
    ) -> tuple[np.ndarray, SegmentationResult]:
        if self.provider is None:
            raise RuntimeError("MedSAM2 is not configured; select local_light or heuristic_hu")
        mask = self.provider.segment(volume=volume, prompt=prompt.model_dump(), metadata=metadata)
        return np.asarray(mask, dtype=np.uint8), SegmentationResult(
            provider=self.name, structure=prompt.structure or "prompted_region",
            confidence=0.8, uncertainty=0.2,
        )
