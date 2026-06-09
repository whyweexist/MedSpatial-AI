"""Typed energy terms."""

from pydantic import BaseModel, Field


class EnergyTerms(BaseModel):
    jepa_residual: float = Field(default=0.0, ge=0.0)
    graph_inconsistency: float = Field(default=0.0, ge=0.0)
    topology: float = Field(default=0.0, ge=0.0)
    segmentation_boundary: float = Field(default=0.0, ge=0.0)
    density_outlier: float = Field(default=0.0, ge=0.0)
    uncertainty_penalty: float = Field(default=0.0, ge=0.0)
