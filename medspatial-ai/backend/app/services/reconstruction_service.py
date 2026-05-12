"""
MedSpatial AI — Reconstruction Service (Enhanced)
Orchestrates 3D volume reconstruction from DICOM series.
Produces tissue-specific meshes via SurfaceProcessor, integrates DepthLifter
for single X-ray → pseudo-3D, body region detection, and anatomy labeling.
"""

import gc
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from scipy import ndimage
from skimage import measure

from app.ai.body_part_labeler import BodyPartLabeler
from app.ai.body_region_classifier import BodyRegionClassifier
from app.config import settings
from app.core.mesh_generator import MeshGenerator
from app.core.region_config import get_region_config
from app.core.surface_processor import (
    CHEST_TISSUE_CONFIGS,
    SurfaceProcessor,
    TissueConfig,
    TissueResult,
)
from app.core.volume_processor import VolumeProcessor


# Dissection order mapping (outside-in, higher = outermost)
_DISSECTION_ORDER = {
    "skin": 8,
    "soft_tissue": 7,
    "bone": 6,
    "left_lung": 4,
    "right_lung": 4,
    "vessels": 3,
    "heart": 2,
    "pathology": 1,
    "brain": 3,
    "liver": 3,
    "kidneys": 3,
}


class ReconstructionService:
    """
    Orchestrates the full reconstruction pipeline:
    DICOM → Volume → Body Region Detection → Tissue-Specific Meshes → Labels.
    """

    def __init__(self):
        self.volume_proc = VolumeProcessor()
        self.mesh_gen = MeshGenerator()
        self.surface_proc = SurfaceProcessor(
            step_size=settings.MARCHING_CUBES_STEP_SIZE
        )
        self.region_classifier = BodyRegionClassifier()
        self.labeler = BodyPartLabeler()

    async def build_reconstruction(
        self,
        scan_id: str,
        volume: np.ndarray,
        voxel_spacing: np.ndarray,
        metadata: dict,
        iso_level: Optional[float] = None,
        step_size: Optional[int] = None,
        generate_layers: bool = True,
    ) -> dict:
        """
        Full reconstruction pipeline.

        Args:
            scan_id: unique scan identifier
            volume: 3D numpy array in Hounsfield Units
            voxel_spacing: [z, y, x] voxel spacing in mm
            metadata: DICOM metadata dict
            iso_level: override iso-surface level
            step_size: override marching cubes step size
            generate_layers: whether to generate tissue-specific meshes

        Returns:
            dict with mesh paths, tissue results, body region, labels, summary
        """
        t_start = time.time()
        output_dir = str(Path(settings.MESH_DIR))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 1. Detect body region
        region_result = self.region_classifier.classify(metadata, volume)
        region_config = get_region_config(region_result.region)
        logger.info(f"Body region: {region_result.region.value} ({region_result.confidence:.0%})")

        # 2. Apply preprocessing
        volume = self.volume_proc.clip_hu(volume)

        # 3. Isotropic resampling if spacing is very anisotropic
        volume, voxel_spacing = self._resample_isotropic(volume, voxel_spacing)

        # 4. Generate primary mesh (full body surface at bone-level iso)
        primary_iso = iso_level or region_config.default_iso_level
        primary_mesh_path = self._generate_primary_mesh(
            volume, scan_id, output_dir, voxel_spacing, primary_iso, step_size
        )

        # 5. Generate tissue-specific layer meshes
        tissue_results: list[TissueResult] = []
        layer_mesh_paths: dict[str, dict] = {}

        if generate_layers:
            try:
                tissue_results = self.surface_proc.process_all_tissues(
                    volume=volume,
                    scan_id=scan_id,
                    output_dir=output_dir,
                    voxel_spacing=voxel_spacing,
                )
            except RuntimeError as exc:
                # OOM fallback: reduce resolution by 50% and retry
                logger.warning(f"Surface processing failed ({exc}), retrying at half resolution")
                gc.collect()
                zoom = [0.5, 0.5, 0.5]
                small_vol = ndimage.zoom(volume, zoom, order=1)
                small_spacing = voxel_spacing / np.array(zoom)
                tissue_results = self.surface_proc.process_all_tissues(
                    volume=small_vol,
                    scan_id=scan_id,
                    output_dir=output_dir,
                    voxel_spacing=small_spacing,
                )

            for tissue in tissue_results:
                if tissue.mesh_path:
                    layer_mesh_paths[tissue.name] = {
                        "mesh_path": tissue.mesh_path,
                        "name": tissue.name,
                        "label_index": tissue.label_index,
                        "vertex_count": tissue.vertex_count,
                        "face_count": tissue.face_count,
                        "volume_mm3": tissue.volume_mm3,
                        "color_rgb": list(tissue.color_rgb),
                        "opacity": tissue.opacity,
                        "centroid_mm": list(tissue.centroid_mm) if tissue.centroid_mm else None,
                        "mean_hu": tissue.mean_hu,
                        "voxel_count": tissue.voxel_count,
                        "dissection_order": _DISSECTION_ORDER.get(tissue.name, 5),
                    }

        # 6. Generate anatomy labels
        labels = []
        if tissue_results:
            labels = self.labeler.generate_labels(
                tissue_results=tissue_results,
                voxel_spacing=voxel_spacing,
                volume_shape=volume.shape,
            )

        # 7. Save volume to disk
        volume_path = str(Path(settings.VOLUME_DIR) / f"{scan_id}.npy")
        np.save(volume_path, volume)

        elapsed = time.time() - t_start
        logger.info(f"Reconstruction complete in {elapsed:.1f}s")

        # 8. Build layer URLs
        layer_urls = {}
        for tissue_name, info in layer_mesh_paths.items():
            layer_urls[tissue_name] = f"/api/reconstruction/mesh/{scan_id}/{tissue_name}"

        # 9. Build summary
        total_verts = sum(t.vertex_count for t in tissue_results)
        total_faces = sum(t.face_count for t in tissue_results)

        summary = {
            "scan_id": scan_id,
            "body_region": {
                "region": region_result.region.value,
                "confidence": region_result.confidence,
                "method": region_result.method,
                "modality": region_result.modality,
                "display_name": region_config.display_name,
                "icon": region_config.icon,
            },
            "tissues": [
                {
                    "name": t.name,
                    "label_index": t.label_index,
                    "vertex_count": t.vertex_count,
                    "face_count": t.face_count,
                    "volume_mm3": t.volume_mm3,
                    "volume_cm3": t.volume_mm3 / 1000.0,
                    "color_rgb": list(t.color_rgb),
                    "opacity": t.opacity,
                    "centroid_mm": list(t.centroid_mm) if t.centroid_mm else None,
                    "mean_hu": t.mean_hu,
                    "voxel_count": t.voxel_count,
                    "description": "",
                    "dissection_order": _DISSECTION_ORDER.get(t.name, 5),
                    "has_mesh": t.mesh_path is not None,
                }
                for t in tissue_results
            ],
            "labels": [
                {
                    "name": l.name,
                    "position": {"x": l.position[0], "y": l.position[1], "z": l.position[2]},
                    "volume_mm3": l.volume_mm3,
                    "color": list(l.color),
                    "layer_index": l.layer_index,
                    "description": l.description,
                }
                for l in labels
            ],
            "total_mesh_vertices": total_verts,
            "total_mesh_faces": total_faces,
            "processing_time_s": elapsed,
        }

        return {
            "scan_id": scan_id,
            "volume_path": volume_path,
            "primary_mesh_path": primary_mesh_path,
            "layer_mesh_paths": layer_mesh_paths,
            "layer_urls": layer_urls,
            "volume_dimensions": {
                "x": volume.shape[2],
                "y": volume.shape[1],
                "z": volume.shape[0],
            },
            "voxel_spacing": {
                "x": float(voxel_spacing[2]),
                "y": float(voxel_spacing[1]),
                "z": float(voxel_spacing[0]),
            },
            "body_region": summary["body_region"],
            "summary": summary,
            "labels": summary["labels"],
            "hu_range": {"min": float(volume.min()), "max": float(volume.max())},
        }

    def _generate_primary_mesh(
        self,
        volume: np.ndarray,
        scan_id: str,
        output_dir: str,
        voxel_spacing: np.ndarray,
        iso_level: float,
        step_size: Optional[int],
    ) -> Optional[str]:
        """Generate the primary combined-tissue mesh."""
        try:
            smoothed = ndimage.gaussian_filter(volume.astype(np.float64), sigma=1.0)
            step = step_size or settings.MARCHING_CUBES_STEP_SIZE

            verts, faces, normals, _ = measure.marching_cubes(
                smoothed, level=iso_level, step_size=step, allow_degenerate=False,
            )

            # Apply spacing and center
            verts = verts * voxel_spacing
            center = np.array(volume.shape) * voxel_spacing / 2.0
            max_extent = float(np.max(np.array(volume.shape) * voxel_spacing))
            verts -= center
            if max_extent > 0:
                verts = verts / max_extent * 100.0

            mesh_path = str(Path(output_dir) / f"{scan_id}_primary.glb")
            self.mesh_gen.save_glb(verts, faces, normals, mesh_path)
            logger.info(f"Primary mesh: {len(verts)} verts, {len(faces)} faces")
            return mesh_path

        except Exception as exc:
            logger.error(f"Primary mesh generation failed: {exc}")
            return None

    def _resample_isotropic(
        self, volume: np.ndarray, spacing: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample to near-isotropic spacing if very anisotropic."""
        if spacing.min() <= 0:
            return volume, spacing

        anisotropy = spacing.max() / spacing.min()
        if anisotropy <= 3.0:
            return volume, spacing

        target_spacing = np.array([spacing.min()] * 3)
        zoom_factors = spacing / target_spacing
        resampled = ndimage.zoom(volume, zoom_factors, order=1)
        logger.info(
            f"Isotropic resampling: {volume.shape} → {resampled.shape} "
            f"(spacing {spacing} → {target_spacing})"
        )
        return resampled, target_spacing

    def generate_layer_meshes(
        self, volume: np.ndarray, scan_id: str, output_dir: str,
        voxel_spacing: np.ndarray, step_size: int = 2,
    ) -> dict[str, Optional[str]]:
        """
        Legacy method: Generate tissue-specific layer meshes using HU thresholds.
        Kept for backward compatibility. New code should use build_reconstruction.
        """
        results = self.surface_proc.process_all_tissues(
            volume=volume,
            scan_id=scan_id,
            output_dir=output_dir,
            voxel_spacing=voxel_spacing,
        )
        return {r.name: r.mesh_path for r in results}

    def extract_slice(
        self, volume: np.ndarray, axis: str, index: int
    ) -> np.ndarray:
        """Extract a 2D slice from the volume."""
        if axis == "axial":
            idx = min(index, volume.shape[0] - 1)
            return volume[idx, :, :]
        elif axis == "coronal":
            idx = min(index, volume.shape[1] - 1)
            return volume[:, idx, :]
        elif axis == "sagittal":
            idx = min(index, volume.shape[2] - 1)
            return volume[:, :, idx]
        else:
            raise ValueError(f"Unknown axis: {axis}")

    async def reconstruct_from_xray(
        self,
        scan_id: str,
        image_array: np.ndarray,
        metadata: dict,
    ) -> dict:
        """
        Reconstruct pseudo-3D model from a single 2D X-ray using DepthLifter.
        """
        try:
            from app.ai.depth_lifter import DepthLifterCNN
            import torch

            lifter = DepthLifterCNN()
            lifter.eval()

            # Normalize image to [0,1]
            img = image_array.astype(np.float32)
            if img.max() > 1.0:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # Convert to tensor (B, 1, H, W)
            tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                depth_volume = lifter(tensor)  # (B, D, H, W)

            pseudo_volume = depth_volume.squeeze(0).numpy()  # (D, H, W)

            # Convert to pseudo-HU range
            pseudo_volume = pseudo_volume * 2000 - 1000  # map to [-1000, 1000]

            voxel_spacing = np.array([1.0, 1.0, 1.0])

            return await self.build_reconstruction(
                scan_id=scan_id,
                volume=pseudo_volume,
                voxel_spacing=voxel_spacing,
                metadata=metadata,
                generate_layers=True,
            )

        except Exception as exc:
            logger.error(f"X-ray reconstruction failed: {exc}")
            # Fallback: create a simple 3D volume from 2D
            D = 64
            pseudo = np.stack([image_array] * D, axis=0).astype(np.float32)
            if pseudo.max() > 1.0:
                pseudo = (pseudo - pseudo.min()) / (pseudo.max() - pseudo.min() + 1e-8)
            pseudo = pseudo * 2000 - 1000

            voxel_spacing = np.array([1.0, 1.0, 1.0])
            return await self.build_reconstruction(
                scan_id=scan_id,
                volume=pseudo,
                voxel_spacing=voxel_spacing,
                metadata=metadata,
                generate_layers=True,
            )
