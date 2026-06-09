"""Weighted, interpretable energy model."""

from dataclasses import dataclass, field

from app.ai.energy.energy_terms import EnergyTerms


@dataclass
class EnergyModel:
    weights: dict[str, float] = field(default_factory=lambda: {
        "jepa_residual": 1.0,
        "graph_inconsistency": 0.8,
        "topology": 0.8,
        "segmentation_boundary": 0.5,
        "density_outlier": 0.7,
        "uncertainty_penalty": 1.0,
    })

    def forward(self, terms: EnergyTerms) -> tuple[float, dict[str, float]]:
        values = terms.model_dump()
        contributions = {key: values[key] * self.weights[key] for key in self.weights}
        return sum(contributions.values()), contributions
