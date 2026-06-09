"""EMA target encoder used for self-supervised JEPA training."""

from copy import deepcopy

import torch
from torch import nn

from app.ai.jepa.context_encoder import ContextEncoder3D


class TargetEncoder3D(nn.Module):
    def __init__(self, context_encoder: ContextEncoder3D) -> None:
        super().__init__()
        self.encoder = deepcopy(context_encoder)
        for parameter in self.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def update(self, context_encoder: ContextEncoder3D, momentum: float = 0.996) -> None:
        for target, source in zip(self.encoder.parameters(), context_encoder.parameters()):
            target.data.mul_(momentum).add_(source.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def forward(self, volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(volume)
