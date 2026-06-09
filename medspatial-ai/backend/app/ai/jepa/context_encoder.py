"""CPU-compatible 3D context encoder."""

import torch
from torch import nn
from torch.nn import functional as F


class ContextEncoder3D(nn.Module):
    def __init__(self, embed_dim: int = 128, base_channels: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, base_channels, 5, stride=2, padding=2),
            nn.GroupNorm(4, base_channels),
            nn.GELU(),
            nn.Conv3d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(4, base_channels * 2),
            nn.GELU(),
            nn.Conv3d(base_channels * 2, embed_dim, 3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(volume)
        tokens = features.flatten(2).transpose(1, 2)
        global_token = F.normalize(tokens.mean(dim=1), dim=-1)
        return global_token, F.normalize(tokens, dim=-1)
