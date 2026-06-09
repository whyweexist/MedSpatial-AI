"""Deterministic imaging quality checks."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualityResult:
    accepted: bool
    score: float
    warnings: list[str] = field(default_factory=list)


class QualityGate:
    def evaluate(self, metadata: dict[str, Any], file_count: int) -> QualityResult:
        warnings: list[str] = []
        score = 1.0
        if file_count == 0:
            return QualityResult(False, 0.0, ["No image files were supplied"])
        if not metadata.get("modality"):
            score -= 0.2
            warnings.append("Modality was inferred because metadata was missing")
        if not metadata.get("rows") or not metadata.get("columns"):
            score -= 0.2
            warnings.append("Image dimensions are incomplete")
        if file_count == 1 and str(metadata.get("modality", "")).upper() == "CT":
            score -= 0.2
            warnings.append("Single-frame CT may not support volumetric reconstruction")
        score = max(0.0, score)
        return QualityResult(score >= 0.4, score, warnings)
