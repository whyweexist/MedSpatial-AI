"""
MedSpatial AI — Depth Lifter Module
Converts a single 2D X-ray into a pseudo-3D depth representation
using a lightweight monocular depth CNN (4 residual blocks).
Inspired by Monodepth2 applied to chest radiographs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Single 2D residual block: Conv → BN → GELU → Conv → BN + skip."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual)


class DepthLifter(nn.Module):
    """
    Depth Lifting Module for 2D X-ray → pseudo-3D tensor.

    Input:  (B, 1, H, W) — grayscale X-ray / radiograph
    Output: (B, 1, D, H, W) — pseudo-depth volume

    Architecture:
      1. Stem: 1→32 channels
      2. 4 × ResidualBlock
      3. Depth head: predict D depth planes in a single forward pass
         via a transposed conv / reshape
    """

    def __init__(self, out_depth: int = 64, base_channels: int = 32) -> None:
        super().__init__()
        self.out_depth = out_depth

        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, 7, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )

        self.encoder = nn.Sequential(
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            ResidualBlock(base_channels * 2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
        )

        # Depth head: predicts D feature maps → each becomes one depth plane
        self.depth_head = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, out_depth, 1),
            nn.Sigmoid(),  # Normalized depth values [0, 1]
        )

    def forward(self, xray: torch.Tensor) -> torch.Tensor:
        """
        xray: (B, 1, H, W) grayscale radiograph, normalized [0, 1]
        Returns: (B, 1, D, H, W) pseudo-depth volume
        """
        B, C, H, W = xray.shape
        x = self.stem(xray)
        x = self.encoder(x)
        depth_planes = self.depth_head(x)  # (B, D, H', W')
        # Upsample to match input spatial dimensions
        depth_planes = F.interpolate(depth_planes, size=(H, W), mode="bilinear",
                                     align_corners=False)  # (B, D, H, W)
        # Reshape to volume: (B, 1, D, H, W)
        volume = depth_planes.unsqueeze(1)  # (B, 1, D, H, W)
        return volume

    def lift_to_hu(self, xray: torch.Tensor, min_hu: float = -1024.0,
                   max_hu: float = 400.0) -> torch.Tensor:
        """
        Lift X-ray to pseudo-HU volume.
        Maps depth [0,1] to HU range appropriate for chest.
        """
        volume = self.forward(xray)  # (B, 1, D, H, W) in [0,1]
        # X-ray intensity inversely correlates with density
        # Bright = air/lung (-1000 HU), Dark = bone (+400 HU)
        # Invert: bright pixels → low HU
        volume_hu = (1.0 - volume) * (max_hu - min_hu) + min_hu
        return volume_hu


def create_depth_lifter(out_depth: int = 64) -> DepthLifter:
    """Create a DepthLifter with given depth planes."""
    return DepthLifter(out_depth=out_depth, base_channels=32)
