"""Deterministic JEPA inference facade."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from app.ai.jepa.context_encoder import ContextEncoder3D
from app.ai.jepa.predictor import LatentPredictor
from app.ai.jepa.world_memory import WorldStateMemory


@dataclass
class JEPAInferenceOutput:
    latent_tokens: torch.Tensor
    slot_embeddings: dict[str, torch.Tensor]
    anomaly_residuals: torch.Tensor
    continuity_score: float
    structure_confidence: float
    uncertainty: float
    embedding_deviation_score: float


class JEPAInferenceEngine:
    def __init__(self, embed_dim: int = 128, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.encoder = ContextEncoder3D(embed_dim).to(self.device).eval()
        self.predictor = LatentPredictor(embed_dim).to(self.device).eval()
        self.memory = WorldStateMemory(embed_dim).to(self.device).eval()

    @torch.inference_mode()
    def infer(self, volume: torch.Tensor) -> JEPAInferenceOutput:
        volume = volume.to(self.device, dtype=torch.float32)
        _, tokens = self.encoder(volume)
        predicted = F.normalize(self.predictor(tokens), dim=-1)
        residuals = (predicted - tokens).pow(2).mean(dim=-1)
        slots = self.memory(tokens)
        slots.pop("attention_weights", None)
        continuity = float(torch.exp(-tokens.diff(dim=1).pow(2).mean()).item()) if tokens.shape[1] > 1 else 1.0
        deviation = float(residuals.mean().item())
        uncertainty = max(0.0, min(1.0, deviation))
        return JEPAInferenceOutput(
            latent_tokens=tokens.cpu(),
            slot_embeddings={key: value.cpu() for key, value in slots.items()},
            anomaly_residuals=residuals.cpu(),
            continuity_score=continuity,
            structure_confidence=1.0 - uncertainty,
            uncertainty=uncertainty,
            embedding_deviation_score=deviation,
        )
