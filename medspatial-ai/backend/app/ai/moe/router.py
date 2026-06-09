"""Deterministic lightweight MoE router."""

from pydantic import BaseModel, Field

from app.ai.moe.budget import InferenceBudget
from app.ai.moe.experts.local import DEFAULT_EXPERTS, LocalExpert


class RouteRequest(BaseModel):
    modality: str
    body_region: str
    task: str
    study_quality: float = Field(default=1.0, ge=0.0, le=1.0)
    available_model_weights: set[str] = Field(default_factory=set)
    required_output_type: str = "json"
    budget: InferenceBudget = Field(default_factory=InferenceBudget)


class RouteDecision(BaseModel):
    selected_experts: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    fallback_route: list[str]
    estimated_memory_mb: int
    policy_constraints: list[str]


class ExpertRouter:
    def __init__(self, experts: tuple[LocalExpert, ...] = DEFAULT_EXPERTS) -> None:
        self.experts = experts

    def route(self, request: RouteRequest) -> RouteDecision:
        candidates = []
        for expert in self.experts:
            modality_match = request.modality in expert.modalities or "*" in expert.modalities
            region_match = request.body_region in expert.regions or "*" in expert.regions
            task_match = request.task in expert.tasks
            weights_match = not request.available_model_weights or expert.name in request.available_model_weights
            if modality_match and region_match and task_match and weights_match:
                candidates.append(expert)
        candidates.sort(key=lambda expert: expert.estimated_memory_mb)
        selected = []
        used = 0
        for expert in candidates:
            if used + expert.estimated_memory_mb <= request.budget.available_memory_mb:
                selected.append(expert)
                used += expert.estimated_memory_mb
        if not selected:
            fallback = ["atlas"] if request.task in {"atlas", "mesh"} else ["qa_evidence"]
        else:
            fallback = ["qa_evidence"] if request.task == "ask" else ["atlas"]
        constraints = ["local_only"] if not request.budget.allow_remote else []
        if request.modality in {"XR", "DX", "CR"}:
            constraints.append("estimated_3d_only")
        return RouteDecision(
            selected_experts=[expert.name for expert in selected],
            confidence=min(0.95, 0.5 + 0.1 * len(selected)) * request.study_quality,
            fallback_route=fallback,
            estimated_memory_mb=used,
            policy_constraints=constraints,
        )
