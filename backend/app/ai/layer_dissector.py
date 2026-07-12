"""
MedSpatial AI — Layer Dissector
Decomposes 3D medical volumes into tissue layers (bone, soft tissue, air, vessels)
using Hounsfield Unit thresholding refined by a learned neural network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from typing import Optional


class LayerRefinementNet(nn.Module):
    """
    Small CNN that refines HU-based tissue masks using learned features.
    Takes the raw volume + initial HU mask and outputs refined boundary masks.
    """

    def __init__(self, num_layers: int = 5):
        super().__init__()
        self.num_layers = num_layers

        # Takes volume (1 ch) + initial masks (num_layers ch)
        in_ch = 1 + num_layers

        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, groups=4),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, num_layers, kernel_size=1),
        )

    def forward(self, volume: torch.Tensor, initial_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: (B, 1, D, H, W)
            initial_masks: (B, num_layers, D, H, W)
        Returns:
            refined_masks: (B, num_layers, D, H, W) with softmax probabilities
        """
        x = torch.cat([volume, initial_masks], dim=1)
        logits = self.net(x)
        return F.softmax(logits, dim=1)


class LayerDissector:
    """
    Decomposes 3D volumes into tissue layers using HU thresholds + optional neural refinement.

    Tissue layers (standard CT Hounsfield Units):
        0 - Background/Air:    HU < -500
        1 - Lung/Fat:          -500 ≤ HU < -100
        2 - Soft Tissue/Water: -100 ≤ HU < 200
        3 - Bone:              200 ≤ HU < 3000
        4 - Metal/Contrast:    HU ≥ 3000
    """

    LAYER_DEFINITIONS = {
        "air": {"index": 0, "hu_min": -1024.0, "hu_max": -500.0, "color": [0.2, 0.2, 0.8, 0.3]},
        "lung_fat": {"index": 1, "hu_min": -500.0, "hu_max": -100.0, "color": [0.9, 0.6, 0.7, 0.4]},
        "soft_tissue": {"index": 2, "hu_min": -100.0, "hu_max": 200.0, "color": [0.9, 0.7, 0.6, 0.6]},
        "bone": {"index": 3, "hu_min": 200.0, "hu_max": 3000.0, "color": [0.95, 0.95, 0.85, 0.9]},
        "contrast": {"index": 4, "hu_min": 3000.0, "hu_max": 5000.0, "color": [1.0, 1.0, 1.0, 1.0]},
    }

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.refinement_net = LayerRefinementNet(num_layers=len(self.LAYER_DEFINITIONS))
        self.refinement_net.to(self.device)
        self.refinement_net.eval()

    def threshold_decomposition(self, volume: np.ndarray) -> dict[str, np.ndarray]:
        """
        Decompose volume into binary masks using HU thresholds.

        Args:
            volume: 3D numpy array in Hounsfield Units

        Returns:
            dict mapping layer name → binary mask (same shape as volume)
        """
        masks = {}
        for name, defn in self.LAYER_DEFINITIONS.items():
            mask = ((volume >= defn["hu_min"]) & (volume < defn["hu_max"])).astype(np.float32)

            # Clean up: remove tiny isolated regions
            if mask.sum() > 50:
                labeled, num_feat = ndimage.label(mask)
                if num_feat > 1:
                    sizes = ndimage.sum(mask, labeled, range(1, num_feat + 1))
                    # Keep components larger than 1% of the biggest
                    max_size = max(sizes)
                    for i, size in enumerate(sizes):
                        if size < max_size * 0.01:
                            mask[labeled == (i + 1)] = 0

                # Smooth boundaries
                mask = ndimage.gaussian_filter(mask, sigma=0.5)
                mask = (mask > 0.5).astype(np.float32)

            masks[name] = mask

        return masks

    def refine_with_network(
        self,
        volume: np.ndarray,
        initial_masks: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        Use the refinement network to improve boundary delineation.

        Args:
            volume: 3D HU volume
            initial_masks: dict of HU-threshold masks
        """
        # Prepare input tensors
        vol_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

        # Stack masks in layer order
        mask_list = []
        layer_names = sorted(self.LAYER_DEFINITIONS.keys(), key=lambda n: self.LAYER_DEFINITIONS[n]["index"])
        for name in layer_names:
            mask_list.append(initial_masks.get(name, np.zeros_like(volume)))
        masks_tensor = torch.from_numpy(np.stack(mask_list)).float().unsqueeze(0)  # (1,N,D,H,W)

        # Downsample if too large for GPU
        original_shape = vol_tensor.shape[2:]
        max_dim = 96
        if max(original_shape) > max_dim:
            scale = max_dim / max(original_shape)
            vol_small = F.interpolate(vol_tensor, scale_factor=scale, mode="trilinear", align_corners=False)
            masks_small = F.interpolate(masks_tensor, scale_factor=scale, mode="trilinear", align_corners=False)
        else:
            vol_small = vol_tensor
            masks_small = masks_tensor

        # Run refinement
        with torch.no_grad():
            vol_small = vol_small.to(self.device)
            masks_small = masks_small.to(self.device)
            refined = self.refinement_net(vol_small, masks_small)

            # Upsample back
            if max(original_shape) > max_dim:
                refined = F.interpolate(refined, size=original_shape, mode="trilinear", align_corners=False)

        refined_np = refined.cpu().numpy()[0]  # (num_layers, D, H, W)

        # Convert back to dict
        refined_masks = {}
        for name in layer_names:
            idx = self.LAYER_DEFINITIONS[name]["index"]
            refined_masks[name] = refined_np[idx]

        return refined_masks

    def decompose(
        self, volume: np.ndarray, use_refinement: bool = True
    ) -> dict[str, dict]:
        """
        Full layer decomposition pipeline.

        Args:
            volume: 3D HU volume
            use_refinement: whether to use neural refinement

        Returns:
            dict mapping layer name → {mask, volume_fraction, voxel_count, color, hu_stats}
        """
        # Step 1: HU threshold decomposition
        initial_masks = self.threshold_decomposition(volume)

        # Step 2: Optional neural refinement
        if use_refinement:
            try:
                masks = self.refine_with_network(volume, initial_masks)
            except Exception:
                masks = initial_masks
        else:
            masks = initial_masks

        # Step 3: Compute statistics per layer
        total_voxels = volume.size
        results = {}

        for name, mask in masks.items():
            binary_mask = (mask > 0.5).astype(np.float32)
            voxel_count = int(binary_mask.sum())
            masked_values = volume[binary_mask > 0.5]

            results[name] = {
                "mask": binary_mask,
                "voxel_count": voxel_count,
                "volume_fraction": float(voxel_count / total_voxels),
                "color": self.LAYER_DEFINITIONS[name]["color"],
                "hu_stats": {
                    "mean": float(masked_values.mean()) if len(masked_values) > 0 else 0.0,
                    "std": float(masked_values.std()) if len(masked_values) > 0 else 0.0,
                    "min": float(masked_values.min()) if len(masked_values) > 0 else 0.0,
                    "max": float(masked_values.max()) if len(masked_values) > 0 else 0.0,
                },
            }

        return results

    def decompose_without_images(
        self, shape: tuple[int, int, int], tissue_type: str = "chest"
    ) -> dict[str, dict]:
        """
        Generate synthetic layer decomposition without actual DICOM images.
        Creates anatomically plausible synthetic volumes for demonstration.

        Args:
            shape: (D, H, W) volume shape
            tissue_type: 'chest', 'head', 'abdomen' etc.
        """
        D, H, W = shape
        volume = np.zeros(shape, dtype=np.float32)

        # Generate synthetic anatomy based on tissue type
        z, y, x = np.ogrid[:D, :H, :W]
        cz, cy, cx = D // 2, H // 2, W // 2

        if tissue_type == "chest":
            # Outer body contour (soft tissue)
            body = ((y - cy) ** 2 / (cy * 0.9) ** 2 + (x - cx) ** 2 / (cx * 0.8) ** 2) < 1
            volume[body] = 40  # soft tissue HU

            # Lung fields
            left_lung = ((y - cy) ** 2 / (cy * 0.4) ** 2 + (x - cx + cx * 0.3) ** 2 / (cx * 0.3) ** 2) < 1
            right_lung = ((y - cy) ** 2 / (cy * 0.4) ** 2 + (x - cx - cx * 0.3) ** 2 / (cx * 0.3) ** 2) < 1
            volume[left_lung] = -700  # lung tissue HU
            volume[right_lung] = -700

            # Spine
            spine = ((y - cy - cy * 0.5) ** 2 / (cy * 0.08) ** 2 + (x - cx) ** 2 / (cx * 0.06) ** 2) < 1
            volume[spine] = 800  # bone HU

            # Ribs (simplified as arcs)
            for rib_z in range(D // 6, D - D // 6, D // 8):
                rib_mask = np.zeros(shape, dtype=bool)
                for angle in np.linspace(0, np.pi, 30):
                    ry = int(cy + cy * 0.7 * np.cos(angle))
                    rx = int(cx + cx * 0.7 * np.sin(angle))
                    if 0 <= ry < H and 0 <= rx < W:
                        volume[max(0, rib_z - 1):min(D, rib_z + 2),
                               max(0, ry - 2):min(H, ry + 3),
                               max(0, rx - 2):min(W, rx + 3)] = 700

            # Heart
            heart = ((z - cz) ** 2 / (cz * 0.3) ** 2 +
                     (y - cy + cy * 0.1) ** 2 / (cy * 0.25) ** 2 +
                     (x - cx + cx * 0.15) ** 2 / (cx * 0.2) ** 2) < 1
            volume[heart] = 45

            # Air outside body
            volume[~body] = -1000

        elif tissue_type == "head":
            # Skull
            skull_outer = ((y - cy) ** 2 + (x - cx) ** 2 + (z - cz) ** 2) < (min(D, H, W) * 0.4) ** 2
            skull_inner = ((y - cy) ** 2 + (x - cx) ** 2 + (z - cz) ** 2) < (min(D, H, W) * 0.35) ** 2
            volume[skull_outer & ~skull_inner] = 1000  # bone
            volume[skull_inner] = 35  # brain tissue
            volume[~skull_outer] = -1000  # air

        else:
            # Generic abdomen
            body = ((y - cy) ** 2 / (cy * 0.85) ** 2 + (x - cx) ** 2 / (cx * 0.75) ** 2) < 1
            volume[body] = 40
            volume[~body] = -1000

            # Spine
            spine = ((y - cy - cy * 0.5) ** 2 / (cy * 0.08) ** 2 + (x - cx) ** 2 / (cx * 0.06) ** 2) < 1
            volume[spine] = 800

        # Add noise for realism
        volume += np.random.normal(0, 5, shape).astype(np.float32)

        return self.decompose(volume, use_refinement=False)
