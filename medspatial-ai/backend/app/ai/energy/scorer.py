"""Public EBM scoring facade."""

from pydantic import BaseModel, Field

from app.ai.energy.calibration import energy_to_plausibility
from app.ai.energy.ebm_model import EnergyModel
from app.ai.energy.energy_terms import EnergyTerms


class EnergyResult(BaseModel):
    energy_score: float = Field(ge=0.0)
    plausibility_score: float = Field(ge=0.0, le=1.0)
    anomaly_likelihood_contribution: float = Field(ge=0.0, le=1.0)
    explanation_terms: dict[str, float]
    evidence_ids: list[str]


class EnergyScorer:
    def __init__(self, model: EnergyModel | None = None) -> None:
        self.model = model or EnergyModel()

    def score(self, terms: EnergyTerms, evidence_ids: list[str] | None = None) -> EnergyResult:
        energy, contributions = self.model.forward(terms)
        plausibility = energy_to_plausibility(energy)
        return EnergyResult(
            energy_score=energy,
            plausibility_score=plausibility,
            anomaly_likelihood_contribution=1.0 - plausibility,
            explanation_terms=contributions,
            evidence_ids=evidence_ids or [],
        )
