"""Metadata for locally available small experts."""

from typing import Any

from app.ai.moe.experts.base import Expert


class LocalExpert(Expert):
    def __init__(
        self, name: str, tasks: set[str], modalities: set[str],
        regions: set[str], estimated_memory_mb: int,
    ) -> None:
        self.name = name
        self.tasks = frozenset(tasks)
        self.modalities = frozenset(modalities)
        self.regions = frozenset(regions)
        self.estimated_memory_mb = estimated_memory_mb

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"expert": self.name, "status": "delegated", "payload": payload}


DEFAULT_EXPERTS = (
    LocalExpert("ct_lung", {"analyze", "segment"}, {"CT"}, {"chest"}, 512),
    LocalExpert("ct_brain", {"analyze", "segment"}, {"CT"}, {"head", "brain"}, 512),
    LocalExpert("mri", {"analyze", "segment"}, {"MR", "NIFTI"}, {"*"}, 768),
    LocalExpert("xray_chest", {"analyze"}, {"XR", "DX", "CR"}, {"chest"}, 256),
    LocalExpert("segmentation", {"segment"}, {"*"}, {"*"}, 384),
    LocalExpert("anomaly", {"analyze"}, {"*"}, {"*"}, 384),
    LocalExpert("atlas", {"atlas", "mesh"}, {"SYNTHETIC", "*"}, {"*"}, 128),
    LocalExpert("qa_evidence", {"ask"}, {"*"}, {"*"}, 128),
    LocalExpert("mesh_geometry", {"mesh"}, {"*"}, {"*"}, 256),
)
