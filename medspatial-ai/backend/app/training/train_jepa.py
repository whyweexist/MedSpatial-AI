"""Minimal optional JEPA training loop entry point."""

import torch

from app.ai.jepa.context_encoder import ContextEncoder3D
from app.ai.jepa.losses import masked_latent_loss
from app.ai.jepa.predictor import LatentPredictor
from app.ai.jepa.target_encoder import TargetEncoder3D


def training_step(volume: torch.Tensor, context: ContextEncoder3D, target: TargetEncoder3D, predictor: LatentPredictor) -> torch.Tensor:
    _, context_tokens = context(volume)
    with torch.no_grad():
        _, target_tokens = target(volume)
    return masked_latent_loss(predictor(context_tokens), target_tokens)
