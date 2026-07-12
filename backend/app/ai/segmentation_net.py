"""
MedSpatial AI — 3D Segmentation Network
3D U-Net variant with Attention Gates for organ/tissue segmentation.
Receives spatial features from SpatialTransformer3D for enhanced segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Double 3D convolution block with batch norm and GELU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate3D(nn.Module):
    """
    Attention Gate for 3D data.
    Helps the decoder focus on relevant spatial regions by gating skip connections.
    """

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv3d(gate_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_ch),
        )
        self.W_skip = nn.Sequential(
            nn.Conv3d(skip_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_ch, 1, kernel_size=1, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.GELU()

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_gate(gate)
        s = self.W_skip(skip)

        # Match spatial dimensions
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(g, size=s.shape[2:], mode="trilinear", align_corners=False)

        combined = self.relu(g + s)
        alpha = self.psi(combined)
        return skip * alpha


class EncoderBlock(nn.Module):
    """Encoder block: ConvBlock + MaxPool downsampling."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock3D(in_ch, out_ch)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        pooled = self.pool(features)
        return pooled, features


class DecoderBlock(nn.Module):
    """Decoder block: upsample + attention gate + skip connection + ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.attention_gate = AttentionGate3D(
            gate_ch=in_ch // 2, skip_ch=skip_ch, inter_ch=skip_ch // 2
        )
        self.conv = ConvBlock3D(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Adjust spatial dimensions if needed
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

        # Attention-gated skip connection
        skip = self.attention_gate(x, skip)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SegmentationNet3D(nn.Module):
    """
    3D U-Net with Attention Gates for organ/tissue segmentation.

    Architecture:
        - Encoder: 4 levels with increasing channels (32→64→128→256)
        - Bottleneck: 512 channels
        - Decoder: 4 levels with attention-gated skip connections
        - Optional spatial transformer feature injection at bottleneck

    Outputs:
        - Segmentation mask: (B, num_classes, D, H, W)
        - Deep supervision outputs for training
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,  # bg, bone, soft_tissue, air, vessel, anomaly
        base_channels: int = 32,
        transformer_dim: int = 0,  # if > 0, inject transformer features
    ):
        super().__init__()
        self.num_classes = num_classes
        ch = base_channels

        # Encoder path
        self.enc1 = EncoderBlock(in_channels, ch)       # 32
        self.enc2 = EncoderBlock(ch, ch * 2)             # 64
        self.enc3 = EncoderBlock(ch * 2, ch * 4)         # 128
        self.enc4 = EncoderBlock(ch * 4, ch * 8)         # 256

        # Bottleneck
        bottleneck_in = ch * 8
        if transformer_dim > 0:
            self.transformer_proj = nn.Sequential(
                nn.Linear(transformer_dim, ch * 8),
                nn.GELU(),
            )
            bottleneck_in = ch * 16  # concatenated with transformer features
        else:
            self.transformer_proj = None

        self.bottleneck = ConvBlock3D(bottleneck_in, ch * 16)  # 512

        # Decoder path
        self.dec4 = DecoderBlock(ch * 16, ch * 8, ch * 8)   # 256
        self.dec3 = DecoderBlock(ch * 8, ch * 4, ch * 4)     # 128
        self.dec2 = DecoderBlock(ch * 4, ch * 2, ch * 2)     # 64
        self.dec1 = DecoderBlock(ch * 2, ch, ch)              # 32

        # Output head
        self.output = nn.Conv3d(ch, num_classes, kernel_size=1)

        # Deep supervision heads
        self.ds3 = nn.Conv3d(ch * 4, num_classes, kernel_size=1)
        self.ds2 = nn.Conv3d(ch * 2, num_classes, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        transformer_features: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 1, D, H, W) input volume
            transformer_features: optional (B, N, D) spatial features from SpatialTransformer3D

        Returns:
            dict with 'segmentation': (B, num_classes, D, H, W) and optional deep supervision outputs
        """
        input_shape = x.shape[2:]

        # Encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        # Inject transformer features at bottleneck
        if self.transformer_proj is not None and transformer_features is not None:
            B, N, D = transformer_features.shape
            tf = self.transformer_proj(transformer_features)  # (B, N, ch*8)

            # Reshape to volume
            vol_size = x.shape[2:]
            n_voxels = vol_size[0] * vol_size[1] * vol_size[2]

            if N >= n_voxels:
                tf = tf[:, :n_voxels]
            else:
                tf = F.interpolate(
                    tf.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1),
                    size=(n_voxels, 1, 1),
                    mode="nearest",
                ).squeeze(-1).squeeze(-1).permute(0, 2, 1)
                tf = tf[:, :n_voxels]

            tf_vol = tf.permute(0, 2, 1).reshape(B, -1, *vol_size)
            x = torch.cat([x, tf_vol], dim=1)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        ds3_out = self.ds3(x)

        x = self.dec2(x, skip2)
        ds2_out = self.ds2(x)

        x = self.dec1(x, skip1)

        # Final output
        seg = self.output(x)

        # Resize to original dimensions
        if seg.shape[2:] != input_shape:
            seg = F.interpolate(seg, size=input_shape, mode="trilinear", align_corners=False)
            ds3_out = F.interpolate(ds3_out, size=input_shape, mode="trilinear", align_corners=False)
            ds2_out = F.interpolate(ds2_out, size=input_shape, mode="trilinear", align_corners=False)

        return {
            "segmentation": seg,
            "deep_sup_3": ds3_out,
            "deep_sup_2": ds2_out,
        }


def create_segmentation_net(
    num_classes: int = 6,
    transformer_dim: int = 512,
    pretrained_path: str = None,
) -> SegmentationNet3D:
    """Factory function to create a SegmentationNet3D."""
    model = SegmentationNet3D(
        in_channels=1,
        num_classes=num_classes,
        base_channels=32,
        transformer_dim=transformer_dim,
    )

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    return model
