"""Predict held-out anatomical latent tokens from context."""

import torch
from torch import nn


class LatentPredictor(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        return self.network(context_tokens)
