"""
MedSpatial AI — Surface Processor
Tissue-specific marching cubes with per-tissue colors, shared coordinate frame,
Laplacian smoothing, largest-connected-component filtering, and GLB export.
Produces anatomically meaningful 3D surfaces from HU volumes.
"""

import gc
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger
from scipy import ndimage
from skimage import measure

from app.core.mesh_generator import MeshGenerator


@dataclass
class TissueConfig:
    """Configuration for a single tissue layer."""
    name: str
    label_index: int
    hu_min: float
    hu_max: float
    iso_level: float
    color_rgb: tuple[float, float, float]
    opacity: float
    smoothing_sigma: float = 1.0
    min_component_size: int = 50
    description: str = ""


@dataclass
class TissueResult:
    """Result from processing a single tissue layer."""
    name: str
    label_index: int
    mesh_path: Optional[str] = None
    vertex_count: int = 0
    face_count: int = 0
    volume_mm3: float = 0.0
    color_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)
    opacity: float = 0.8
    centroid_mm: Optional[tuple[float, float, float]] = None
    bounds_mm: Optional[dict] = None
    voxel_count: int = 0
    mean_hu: float = 0.0


# 8 key anatomical layers for chest CT
CHEST_TISSUE_CONFIGS: list[TissueConfig] = [
    TissueConfig(
        name="skin",
        label_index=1,
        hu_min=-200.0,
        hu_max=100.0,
        iso_level=-50.0,
        color_rgb=(0.90, 0.75, 0.65),
        opacity=0.3,
        smoothing_sigma=1.5,
        min_component_size=200,
        description="Skin and subcutaneous tissue",
    ),
    TissueConfig(
        name="bone",
        label_index=3,
        hu_min=200.0,
        hu_max=3071.0,
        iso_level=300.0,
        color_rgb=(0.95, 0.92, 0.80),
        opacity=0.9,
        smoothing_sigma=0.8,
        min_component_size=30,
        description="Skeletal structures (ribs, spine, sternum, clavicles)",
    ),
    TissueConfig(
        name="left_lung",
        label_index=5,
        hu_min=-950.0,
        hu_max=-200.0,
        iso_level=-500.0,
        color_rgb=(0.40, 0.65, 0.85),
        opacity=0.4,
        smoothing_sigma=1.2,
        min_component_size=500,
        description="Left lung parenchyma",
    ),
    TissueConfig(
        name="right_lung",
        label_index=6,
        hu_min=-950.0,
        hu_max=-200.0,
        iso_level=-500.0,
        color_rgb=(0.30, 0.55, 0.80),
        opacity=0.4,
        smoothing_sigma=1.2,
        min_component_size=500,
        description="Right lung parenchyma",
    ),
    TissueConfig(
        name="heart",
        label_index=9,
        hu_min=-50.0,
        hu_max=200.0,
        iso_level=30.0,
        color_rgb=(0.90, 0.40, 0.40),
        opacity=0.7,
        smoothing_sigma=1.5,
        min_component_size=200,
        description="Heart and cardiac silhouette",
    ),
    TissueConfig(
        name="vessels",
        label_index=8,
        hu_min=150.0,
        hu_max=500.0,
        iso_level=200.0,
        color_rgb=(0.85, 0.20, 0.20),
        opacity=0.7,
        smoothing_sigma=0.8,
        min_component_size=20,
        description="Pulmonary and great vessels",
    ),
    TissueConfig(
        name="soft_tissue",
        label_index=2,
        hu_min=-100.0,
        hu_max=200.0,
        iso_level=40.0,
        color_rgb=(0.90, 0.70, 0.60),
        opacity=0.4,
        smoothing_sigma=1.5,
        min_component_size=100,
        description="Musculature and soft tissue",
    ),
    TissueConfig(
        name="pathology",
        label_index=11,
        hu_min=-100.0,
        hu_max=400.0,
        iso_level=50.0,
        color_rgb=(1.00, 0.15, 0.00),
        opacity=0.9,
        smoothing_sigma=0.5,
        min_component_size=10,
        description="Abnormality / pathology regions",
    ),
]


class SurfaceProcessor:
    """
    Generates tissue-specific 3D surface meshes from HU volumes.
    All meshes share a common coordinate frame so they align.
    """

    def __init__(self, step_size: int = 2, max_volume_dim: int = 192) -> None:
        self.step_size = step_size
        self.max_volume_dim = max_volume_dim
        self.mesh_gen = MeshGenerator()

    def process_all_tissues(
        self,
        volume: np.ndarray,
        scan_id: str,
        output_dir: str,
        voxel_spacing: np.ndarray,
        tissue_configs: Optional[list[TissueConfig]] = None,
        segmentation_mask: Optional[np.ndarray] = None,
    ) -> list[TissueResult]:
        """
        Generate meshes for all tissue layers from a single volume.

        Args:
            volume: 3D float32 array in Hounsfield Units
            scan_id: unique scan identifier
            output_dir: directory to save GLB files
            voxel_spacing: [z, y, x] spacing in mm
            tissue_configs: tissue layer definitions (defaults to CHEST_TISSUE_CONFIGS)
            segmentation_mask: optional label volume from neural segmentation

        Returns:
            list of TissueResult with mesh paths and statistics
        """
        if tissue_configs is None:
            tissue_configs = CHEST_TISSUE_CONFIGS

        volume = volume.astype(np.float32, copy=False)
        results: list[TissueResult] = []

        # Compute shared bounding box from body mask for alignment
        body_mask = volume > -400.0
        body_indices = np.argwhere(body_mask)
        if len(body_indices) == 0:
            logger.warning("No body voxels found in volume")
            return results

        shared_origin = body_indices.min(axis=0).astype(float)
        shared_extent = body_indices.max(axis=0).astype(float)
        shared_center = (shared_origin + shared_extent) / 2.0
        shared_max_extent = float(
            np.max((shared_extent - shared_origin) * voxel_spacing)
        )
        if shared_max_extent < 1.0:
            shared_max_extent = 1.0

        # Downsample volume if too large
        work_volume, work_spacing, zoom_factors = self._downsample_if_needed(
            volume, voxel_spacing
        )
        work_center = shared_center * zoom_factors if zoom_factors is not None else shared_center

        # Separate lungs by spatial position (left=x>center, right=x<center)
        lung_separated = self._separate_lungs(work_volume, work_center)

        for tissue_cfg in tissue_configs:
            try:
                result = self._process_single_tissue(
                    work_volume=work_volume,
                    tissue_cfg=tissue_cfg,
                    scan_id=scan_id,
                    output_dir=output_dir,
                    work_spacing=work_spacing,
                    shared_center=work_center,
                    shared_max_extent=shared_max_extent,
                    segmentation_mask=segmentation_mask,
                    lung_masks=lung_separated,
                )
                results.append(result)
            except Exception as exc:
                logger.error(f"Failed to process tissue '{tissue_cfg.name}': {exc}")
                results.append(
                    TissueResult(
                        name=tissue_cfg.name,
                        label_index=tissue_cfg.label_index,
                        color_rgb=tissue_cfg.color_rgb,
                        opacity=tissue_cfg.opacity,
                    )
                )
            gc.collect()

        logger.info(
            f"Surface processing complete: {len([r for r in results if r.mesh_path])} "
            f"of {len(tissue_configs)} tissues generated meshes"
        )
        return results

    def _process_single_tissue(
        self,
        work_volume: np.ndarray,
        tissue_cfg: TissueConfig,
        scan_id: str,
        output_dir: str,
        work_spacing: np.ndarray,
        shared_center: np.ndarray,
        shared_max_extent: float,
        segmentation_mask: Optional[np.ndarray],
        lung_masks: Optional[dict[str, np.ndarray]],
    ) -> TissueResult:
        """Process a single tissue layer into a GLB mesh."""

        result = TissueResult(
            name=tissue_cfg.name,
            label_index=tissue_cfg.label_index,
            color_rgb=tissue_cfg.color_rgb,
            opacity=tissue_cfg.opacity,
        )

        # Create tissue mask
        if tissue_cfg.name == "left_lung" and lung_masks and "left" in lung_masks:
            tissue_mask = lung_masks["left"]
        elif tissue_cfg.name == "right_lung" and lung_masks and "right" in lung_masks:
            tissue_mask = lung_masks["right"]
        elif segmentation_mask is not None and tissue_cfg.label_index < segmentation_mask.max() + 1:
            tissue_mask = (segmentation_mask == tissue_cfg.label_index).astype(np.float32)
        else:
            tissue_mask = (
                (work_volume >= tissue_cfg.hu_min) & (work_volume < tissue_cfg.hu_max)
            ).astype(np.float32)

        # For skin: create outer shell only (subtract interior)
        if tissue_cfg.name == "skin":
            body = work_volume > -400.0
            body_filled = ndimage.binary_fill_holes(body)
            eroded = ndimage.binary_erosion(body_filled, iterations=3)
            tissue_mask = (body_filled & ~eroded).astype(np.float32)

        # For heart: use central volume region
        if tissue_cfg.name == "heart":
            cz, cy, cx = work_volume.shape[0] // 2, work_volume.shape[1] // 2, work_volume.shape[2] // 2
            z, y, x = np.ogrid[:work_volume.shape[0], :work_volume.shape[1], :work_volume.shape[2]]
            cardiac_region = (
                ((z - cz) / max(cz * 0.35, 1)) ** 2
                + ((y - cy + cy * 0.1) / max(cy * 0.3, 1)) ** 2
                + ((x - cx + cx * 0.15) / max(cx * 0.25, 1)) ** 2
            ) < 1.0
            tissue_mask = tissue_mask * cardiac_region.astype(np.float32)

        voxel_count = int(tissue_mask.sum())
        result.voxel_count = voxel_count

        if voxel_count < tissue_cfg.min_component_size:
            logger.info(f"Tissue '{tissue_cfg.name}': {voxel_count} voxels, skipping (min={tissue_cfg.min_component_size})")
            return result

        # Compute HU stats
        masked_hu = work_volume[tissue_mask > 0.5]
        result.mean_hu = float(masked_hu.mean()) if len(masked_hu) > 0 else 0.0

        # Compute volume in mm³
        voxel_vol_mm3 = float(np.prod(work_spacing))
        result.volume_mm3 = voxel_count * voxel_vol_mm3

        # Compute centroid in mm
        centroid_voxel = ndimage.center_of_mass(tissue_mask)
        centroid_mm = np.array(centroid_voxel) * work_spacing
        result.centroid_mm = (float(centroid_mm[0]), float(centroid_mm[1]), float(centroid_mm[2]))

        # Largest connected component filtering
        tissue_mask = self._largest_components(tissue_mask, tissue_cfg.min_component_size)

        # Smooth the mask
        tissue_mask = ndimage.gaussian_filter(tissue_mask, sigma=tissue_cfg.smoothing_sigma)

        # Create volume for marching cubes
        layer_vol = np.where(tissue_mask > 0.3, work_volume, tissue_cfg.hu_min - 100)
        layer_vol = ndimage.gaussian_filter(layer_vol.astype(np.float64), sigma=tissue_cfg.smoothing_sigma)

        # Run marching cubes
        try:
            verts, faces, normals, _ = measure.marching_cubes(
                layer_vol,
                level=tissue_cfg.iso_level,
                step_size=self.step_size,
                allow_degenerate=False,
            )
        except (ValueError, RuntimeError):
            # Try auto threshold
            threshold = np.percentile(layer_vol[tissue_mask > 0.3], 50) if (tissue_mask > 0.3).any() else tissue_cfg.iso_level
            try:
                verts, faces, normals, _ = measure.marching_cubes(
                    layer_vol,
                    level=threshold,
                    step_size=self.step_size,
                    allow_degenerate=False,
                )
            except (ValueError, RuntimeError):
                logger.warning(f"Marching cubes failed for '{tissue_cfg.name}'")
                return result

        if len(verts) < 3 or len(faces) < 1:
            return result

        # Apply voxel spacing
        verts = verts * work_spacing

        # Center using shared coordinate frame
        verts -= shared_center * work_spacing

        # Normalize using shared extent
        if shared_max_extent > 0:
            verts = verts / shared_max_extent * 100.0

        # Laplacian smoothing (simple)
        verts = self._laplacian_smooth(verts, faces, iterations=3, factor=0.3)

        # Generate per-vertex colors
        r, g, b = tissue_cfg.color_rgb
        vertex_colors = np.zeros((len(verts), 4), dtype=np.float32)
        vertex_colors[:, 0] = r
        vertex_colors[:, 1] = g
        vertex_colors[:, 2] = b
        vertex_colors[:, 3] = tissue_cfg.opacity

        # Save GLB
        from pathlib import Path
        mesh_path = str(Path(output_dir) / f"{scan_id}_{tissue_cfg.name}.glb")
        self.mesh_gen.save_glb(verts, faces, normals, mesh_path, vertex_colors=vertex_colors)

        result.mesh_path = mesh_path
        result.vertex_count = len(verts)
        result.face_count = len(faces)

        # Compute bounds
        result.bounds_mm = {
            "min": {"x": float(verts[:, 2].min()), "y": float(verts[:, 1].min()), "z": float(verts[:, 0].min())},
            "max": {"x": float(verts[:, 2].max()), "y": float(verts[:, 1].max()), "z": float(verts[:, 0].max())},
        }

        logger.info(
            f"Tissue '{tissue_cfg.name}': {len(verts)} verts, {len(faces)} faces, "
            f"vol={result.volume_mm3 / 1000:.1f} cm³"
        )
        return result

    def _downsample_if_needed(
        self, volume: np.ndarray, spacing: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Downsample volume if dimensions exceed max_volume_dim."""
        if max(volume.shape) <= self.max_volume_dim:
            return volume, spacing, None

        zoom_factors = np.array([self.max_volume_dim / s for s in volume.shape])
        zoom_factors = np.minimum(zoom_factors, 1.0)
        work_volume = ndimage.zoom(volume, zoom_factors, order=1)
        work_spacing = spacing / zoom_factors
        logger.info(f"Downsampled volume {volume.shape} → {work_volume.shape}")
        return work_volume, work_spacing, zoom_factors

    def _separate_lungs(
        self, volume: np.ndarray, center: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Separate left and right lungs by spatial position."""
        lung_mask = ((volume >= -950.0) & (volume < -200.0)).astype(np.float32)

        if lung_mask.sum() < 100:
            return {}

        cx = int(center[2]) if len(center) > 2 else volume.shape[2] // 2

        left_mask = np.zeros_like(lung_mask)
        right_mask = np.zeros_like(lung_mask)
        left_mask[:, :, cx:] = lung_mask[:, :, cx:]
        right_mask[:, :, :cx] = lung_mask[:, :, :cx]

        return {"left": left_mask, "right": right_mask}

    def _largest_components(
        self, mask: np.ndarray, min_size: int
    ) -> np.ndarray:
        """Keep only connected components larger than min_size."""
        binary = (mask > 0.5).astype(np.int32)
        labeled, num_features = ndimage.label(binary)

        if num_features <= 1:
            return mask

        sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        filtered = np.zeros_like(mask)

        for i, size in enumerate(sizes):
            if size >= min_size:
                filtered[labeled == (i + 1)] = mask[labeled == (i + 1)]

        return filtered

    def _laplacian_smooth(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        iterations: int = 3,
        factor: float = 0.3,
    ) -> np.ndarray:
        """Simple Laplacian mesh smoothing."""
        if len(verts) == 0 or len(faces) == 0:
            return verts

        num_verts = len(verts)
        smoothed = verts.copy()

        # Build adjacency
        adjacency: list[set[int]] = [set() for _ in range(num_verts)]
        for face in faces:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        vi, vj = int(face[i]), int(face[j])
                        if vi < num_verts and vj < num_verts:
                            adjacency[vi].add(vj)

        for _ in range(iterations):
            new_verts = smoothed.copy()
            for i in range(num_verts):
                neighbors = adjacency[i]
                if len(neighbors) > 0:
                    avg = np.mean(smoothed[list(neighbors)], axis=0)
                    new_verts[i] = smoothed[i] + factor * (avg - smoothed[i])
            smoothed = new_verts

        return smoothed
