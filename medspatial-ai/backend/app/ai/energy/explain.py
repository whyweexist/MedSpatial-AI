"""Rank energy contributions for explanations."""

from app.ai.energy.scorer import EnergyResult


def dominant_terms(result: EnergyResult, limit: int = 3) -> list[tuple[str, float]]:
    return sorted(result.explanation_terms.items(), key=lambda item: item[1], reverse=True)[:limit]
