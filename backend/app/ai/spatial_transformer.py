"""
MedSpatial AI — SpatialTransformer3D
A novel 3D Vision Transformer architecture designed for volumetric medical imaging.

This is NOT a standard ViT. Key innovations:
  1. 3D Patch Embedding with overlapping patches for volumetric data
  2. Cross-Plane Attention: learns inter-slice relationships across axial/coronal/sagittal
  3. Volumetric Positional Encoding: 3D sinusoidal position encoding
  4. Multi-Scale Feature Pyramid: extracts features at multiple resolutions
  5. Spatial Attention Gates: learnable gating that focuses on anatomically relevant regions

Architecture flow:
  Input (B, 1, D, H, W) → 3D Patch Embed → Cross-Plane Attention Blocks → 
  Multi-scale Feature Pyramid → Output features
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class VolumetricPositionEncoding(nn.Module):
    """
    Learnable + sinusoidal 3D positional encoding for volumetric patches.
    Combines sinusoidal base encoding with learnable residual for adaptability.
    """

    def __init__(self, embed_dim: int, max_positions: int = 512):
        super().__init__()
        self.embed_dim = embed_dim

        # Sinusoidal base
        pe = torch.zeros(max_positions, embed_dim)
        position = torch.arange(0, max_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: embed_dim // 2])
        self.register_buffer("sinusoidal_pe", pe)

        # Learnable residual
        self.learnable_pe = nn.Parameter(torch.randn(1, max_positions, embed_dim) * 0.02)

        # 3D spatial embedding networks for depth, height, width
        self.depth_embed = nn.Embedding(64, embed_dim // 3 + embed_dim % 3)
        self.height_embed = nn.Embedding(64, embed_dim // 3)
        self.width_embed = nn.Embedding(64, embed_dim // 3)

    def forward(self, x: torch.Tensor, grid_shape: tuple[int, int, int] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) patch embeddings
            grid_shape: (depth_patches, height_patches, width_patches)
        """
        B, N, D = x.shape

        if grid_shape is not None:
            gd, gh, gw = grid_shape

            d_idx = torch.arange(gd, device=x.device).clamp(max=63)
            h_idx = torch.arange(gh, device=x.device).clamp(max=63)
            w_idx = torch.arange(gw, device=x.device).clamp(max=63)

            d_emb = self.depth_embed(d_idx)  # (gd, D/3)
            h_emb = self.height_embed(h_idx)  # (gh, D/3)
            w_emb = self.width_embed(w_idx)  # (gw, D/3)

            # Determine if CLS token is present in x (common in transformer pipelines)
            has_cls = (N == gd * gh * gw + 1)

            # Create positional encoding for patch tokens and optional CLS token.
            pos_3d = torch.zeros(N, D, device=x.device)
            for i in range(gd):
                for j in range(gh):
                    for k in range(gw):
                        patch_idx = i * gh * gw + j * gw + k
                        token_idx = patch_idx + 1 if has_cls else patch_idx
                        if token_idx < N:
                            pos_3d[token_idx] = torch.cat([d_emb[i], h_emb[j], w_emb[k]])

            # Keep CLS token at zero if present (no positional bias)
            x = x + pos_3d.unsqueeze(0)

        # Add sinusoidal + learnable encoding
        if N > self.sinusoidal_pe.shape[0]:
            # Expand or wrap-around positional encodings for very large grids.
            repeat_count = (N + self.sinusoidal_pe.shape[0] - 1) // self.sinusoidal_pe.shape[0]
            sinusoidal = self.sinusoidal_pe.repeat(repeat_count, 1)[:N]
        else:
            sinusoidal = self.sinusoidal_pe[:N]

        if N > self.learnable_pe.shape[1]:
            repeat_count = (N + self.learnable_pe.shape[1] - 1) // self.learnable_pe.shape[1]
            learnable = self.learnable_pe.repeat(1, repeat_count, 1)[:, :N, :]
        else:
            learnable = self.learnable_pe[:, :N, :]

        x = x + sinusoidal.unsqueeze(0).to(x.device) + learnable.to(x.device)
        return x


class CrossPlaneAttention(nn.Module):
    """
    Cross-Plane Multi-Head Attention: attends across axial, coronal, and sagittal planes.
    This allows the model to learn 3D spatial relationships that standard 2D attention misses.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Cross-plane projection layers
        self.axial_proj = nn.Linear(embed_dim, embed_dim)
        self.coronal_proj = nn.Linear(embed_dim, embed_dim)
        self.sagittal_proj = nn.Linear(embed_dim, embed_dim)
        self.plane_gate = nn.Linear(embed_dim * 3, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        grid_shape: tuple[int, int, int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) token embeddings
            grid_shape: (depth, height, width) grid dimensions
        """
        B, N, D = x.shape

        # Standard multi-head self-attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, D)

        # Cross-plane gating: project through axial/coronal/sagittal pathways
        if grid_shape is not None and grid_shape[0] * grid_shape[1] * grid_shape[2] <= N:
            gd, gh, gw = grid_shape
            n_patches = gd * gh * gw

            x_vol = x_attn[:, :n_patches]  # only volumetric tokens

            # Reshape to 3D grid
            x_3d = x_vol.reshape(B, gd, gh, gw, D)

            # Axial pooling (along depth)
            axial = x_3d.mean(dim=1)  # (B, gh, gw, D)
            axial = self.axial_proj(axial).reshape(B, gh * gw, D)
            axial_expanded = repeat(axial, "b hw d -> b gd hw d", gd=gd).reshape(B, n_patches, D)

            # Coronal pooling (along height)
            coronal = x_3d.mean(dim=2)  # (B, gd, gw, D)
            coronal = self.coronal_proj(coronal).reshape(B, gd * gw, D)
            coronal_expanded = repeat(coronal, "b dw d -> b gh dw d", gh=gh).reshape(B, n_patches, D)

            # Sagittal pooling (along width)
            sagittal = x_3d.mean(dim=3)  # (B, gd, gh, D)
            sagittal = self.sagittal_proj(sagittal).reshape(B, gd * gh, D)
            sagittal_expanded = repeat(sagittal, "b dh d -> b gw dh d", gw=gw).reshape(B, n_patches, D)

            # Gating
            cross_plane = torch.cat([axial_expanded, coronal_expanded, sagittal_expanded], dim=-1)
            gate = torch.sigmoid(self.plane_gate(cross_plane))
            x_attn[:, :n_patches] = x_attn[:, :n_patches] * gate

        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        return x_attn


class SpatialAttentionGate(nn.Module):
    """
    Spatial Attention Gate: learns which 3D regions are most relevant.
    Used between transformer blocks to dynamically focus on anatomically important areas.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.Sigmoid(),
        )
        self.spatial_conv = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel + spatial attention gating."""
        # Channel attention
        channel_weights = self.channel_attn(x.mean(dim=1, keepdim=True))
        x = x * channel_weights

        # Spatial attention
        spatial_weights = self.spatial_conv(x)
        x = x * spatial_weights

        return x


class TransformerBlock(nn.Module):
    """Single transformer block with Cross-Plane Attention and FFN."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CrossPlaneAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )
        self.gate = SpatialAttentionGate(embed_dim)

    def forward(self, x: torch.Tensor, grid_shape: tuple = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), grid_shape=grid_shape)
        x = x + self.mlp(self.norm2(x))
        x = self.gate(x)
        return x


class PatchEmbedding3D(nn.Module):
    """
    3D Patch Embedding with overlapping patches for volumetric data.
    Uses 3D convolutions to create overlapping patches, capturing local context.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 512,
        patch_size: int = 16,
        overlap: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap
        stride = patch_size - overlap

        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim // 4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(embed_dim // 4),
            nn.GELU(),
            nn.Conv3d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(embed_dim // 2),
            nn.GELU(),
            nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            x: (B, C, D, H, W) volume

        Returns:
            embeddings: (B, N, embed_dim)
            grid_shape: (gd, gh, gw)
        """
        x = self.proj(x)  # (B, embed_dim, gd, gh, gw)
        B, C, gd, gh, gw = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, gd*gh*gw, embed_dim)
        return x, (gd, gh, gw)


class MultiScaleFeaturePyramid(nn.Module):
    """
    Multi-Scale Feature Pyramid Network for extracting features at multiple resolutions.
    Processes the transformer output at different scales for fine-grained spatial understanding.
    """

    def __init__(self, embed_dim: int, num_scales: int = 3):
        super().__init__()
        self.scales = nn.ModuleList()
        for i in range(num_scales):
            scale_dim = embed_dim // (2 ** i)
            self.scales.append(
                nn.Sequential(
                    nn.Linear(embed_dim, scale_dim),
                    nn.LayerNorm(scale_dim),
                    nn.GELU(),
                    nn.Linear(scale_dim, embed_dim),
                )
            )
        self.fusion = nn.Linear(embed_dim * num_scales, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-scale feature extraction and fusion."""
        scale_features = [scale(x) for scale in self.scales]
        fused = torch.cat(scale_features, dim=-1)
        return self.fusion(fused)


class SpatialTransformer3D(nn.Module):
    """
    SpatialTransformer3D — Custom 3D Vision Transformer for volumetric medical imaging.

    This is the core architecture of MedSpatial AI. It combines:
    - 3D overlapping patch embedding
    - Volumetric positional encoding (sinusoidal + learnable)
    - Cross-Plane Attention blocks (axial/coronal/sagittal aware)
    - Spatial Attention Gates
    - Multi-Scale Feature Pyramid

    Usage:
        model = SpatialTransformer3D(embed_dim=512, num_heads=8, num_layers=6)
        features = model(volume)  # volume: (B, 1, D, H, W)
        # features: (B, N, embed_dim) — spatial feature tokens
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        patch_size: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes: int = 0,  # 0 = feature extractor mode
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # 3D Patch Embedding
        self.patch_embed = PatchEmbedding3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Positional encoding
        self.pos_encoding = VolumetricPositionEncoding(embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Feature pyramid
        self.feature_pyramid = MultiScaleFeaturePyramid(embed_dim)

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head (optional)
        if num_classes > 0:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, num_classes),
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with truncated normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract spatial features from a 3D volume.

        Args:
            x: (B, 1, D, H, W) input volume

        Returns:
            cls_features: (B, embed_dim) global features from CLS token
            spatial_features: (B, N, embed_dim) per-patch spatial features
        """
        B = x.shape[0]

        # Patch embedding
        x, grid_shape = self.patch_embed(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Positional encoding
        x = self.pos_encoding(x, grid_shape)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, grid_shape)

        # Multi-scale feature pyramid
        x = self.feature_pyramid(x)

        # Normalize
        x = self.norm(x)

        cls_features = x[:, 0]  # (B, embed_dim)
        spatial_features = x[:, 1:]  # (B, N, embed_dim)

        return cls_features, spatial_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 1, D, H, W) input volume

        Returns:
            If num_classes > 0: (B, num_classes) classification logits
            Else: (B, N+1, embed_dim) all features including CLS
        """
        cls_features, spatial_features = self.extract_features(x)

        if self.num_classes > 0:
            return self.head(cls_features)

        # Return all features
        return torch.cat([cls_features.unsqueeze(1), spatial_features], dim=1)


def create_spatial_transformer(
    embed_dim: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    num_classes: int = 0,
    pretrained_path: str = None,
) -> SpatialTransformer3D:
    """Factory function to create a SpatialTransformer3D model."""
    model = SpatialTransformer3D(
        in_channels=1,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
    )

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    return model
