"""
MedSpatial AI — Reconstruction Service
Converts DICOM volumes into 3D meshes using marching cubes and surface extraction.
"""

import base64
import io
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image
from scipy import ndimage
from skimage import measure

from app.config import settings
from app.core.mesh_generator import MeshGenerator
from app.core.volume_processor import VolumeProcessor
from app.schemas import SliceResponse
from app.services.dicom_service import DicomService


class ReconstructionService:
    """Orchestrates 3D volume reconstruction from DICOM series."""

    def __init__(self):
        self.dicom_svc = DicomService()
        self.volume_proc = VolumeProcessor()
        self.mesh_gen = MeshGenerator()

    def build_volume(self, upload_dir: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load DICOM files from upload directory and construct a 3D HU volume.

        Returns:
            volume: 3D numpy array (z, y, x) in Hounsfield units
            voxel_spacing: array [z, y, x] spacing in mm
        """
        volume, voxel_spacing = self.dicom_svc.load_dicom_series(upload_dir)

        # Enforce 3D volume format (in case a 4D stack is produced unintentionally)
        if volume.ndim == 4 and volume.shape[0] == 1:
            volume = volume[0]

        if volume.ndim != 3:
            raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")

        # Keep memory lower precision where feasible.
        volume = volume.astype(np.float32, copy=False)

        # Clip to plausible HU range before smoothing and meshing.
        volume = self.volume_proc.clip_hu(volume)
        volume = self.volume_proc.denoise(volume)

        logger.info(f"Volume built: {volume.shape}, range [{volume.min():.0f}, {volume.max():.0f}] HU")
        return volume, voxel_spacing

    def save_volume(self, volume: np.ndarray, path: str) -> None:
        """Save volume as compressed numpy file."""
        np.save(path, volume)
        logger.info(f"Volume saved to {path}")

    def generate_mesh(
        self,
        volume: np.ndarray,
        output_path: str,
        iso_level: float = 300.0,
        step_size: int = 2,
        voxel_spacing: np.ndarray = None,
    ) -> str:
        """
        Generate a 3D mesh from the volume using marching cubes.

        Args:
            volume: 3D HU volume
            output_path: path to save .glb mesh
            iso_level: Hounsfield unit threshold for surface extraction
            step_size: marching cubes step size (higher = faster, lower resolution)
            voxel_spacing: [z,y,x] spacing for correct aspect ratio
        """
        # Validate volume shape + type to avoid OOM from accidental extra dims.
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume for mesh generation, got {volume.shape}")

        vol = volume.astype(np.float32, copy=False)

        # Downsample for performance if volume is large
        if max(vol.shape) > 256:
            zoom_factors = [256 / s for s in vol.shape]
            vol = ndimage.zoom(vol, zoom_factors, order=1)
            if voxel_spacing is not None:
                voxel_spacing = voxel_spacing / np.array(zoom_factors)

        # Apply gaussian smoothing for better surface
        vol = ndimage.gaussian_filter(vol, sigma=1.0)

        # Run marching cubes
        try:
            verts, faces, normals, values = measure.marching_cubes(
                vol,
                level=iso_level,
                step_size=step_size,
                allow_degenerate=False,
            )
        except ValueError:
            # If iso_level produces no surface, try with auto threshold
            threshold = np.percentile(vol, 75)
            logger.warning(f"Iso level {iso_level} failed, retrying with auto threshold {threshold:.0f}")
            verts, faces, normals, values = measure.marching_cubes(
                vol,
                level=threshold,
                step_size=step_size,
                allow_degenerate=False,
            )

        # Apply voxel spacing to vertices
        if voxel_spacing is not None:
            verts = verts * voxel_spacing

        # Center the mesh
        center = (verts.max(axis=0) + verts.min(axis=0)) / 2
        verts -= center

        # Normalize to reasonable scale
        max_extent = np.abs(verts).max()
        if max_extent > 0:
            verts = verts / max_extent * 100.0

        # Generate and save mesh
        self.mesh_gen.save_glb(verts, faces, normals, output_path)
        logger.info(f"Mesh saved: {output_path} ({len(verts)} verts, {len(faces)} faces)")
        return output_path

    def generate_layer_meshes(
        self,
        volume: np.ndarray,
        scan_id: str,
        mesh_dir: str,
        voxel_spacing: np.ndarray = None,
    ) -> dict[str, str]:
        """
        Generate separate meshes for different tissue layers using HU thresholds.

        Standard HU ranges:
            Air:         -1000 to -500
            Lung/Fat:     -500 to -100
            Soft tissue:  -100 to  300
            Bone:          300 to 3000+
            Vessel:        200 to  600 (contrast-enhanced)
        """
        layer_configs = {
            "air": {"hu_min": -1000.0, "hu_max": -500.0, "iso_offset": 0.5},
            "soft_tissue": {"hu_min": -100.0, "hu_max": 300.0, "iso_offset": 0.5},
            "bone": {"hu_min": 300.0, "hu_max": 3000.0, "iso_offset": 0.5},
            "vessel": {"hu_min": 200.0, "hu_max": 600.0, "iso_offset": 0.5},
        }

        layer_paths = {}
        mesh_dir_path = Path(mesh_dir)

        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume for layer generation, got {volume.shape}")

        volume = volume.astype(np.float32, copy=False)

        for layer_name, config in layer_configs.items():
            try:
                # Create binary mask for this tissue type
                mask = (volume >= config["hu_min"]) & (volume < config["hu_max"])

                # Skip if too few voxels
                voxel_count = mask.sum()
                if voxel_count < 100:
                    logger.warning(f"Layer '{layer_name}': only {voxel_count} voxels, skipping")
                    continue

                # Create layer volume preserving HU values
                layer_vol = np.where(mask, volume, config["hu_min"] - 1)

                # Smooth the layer
                layer_vol = ndimage.gaussian_filter(layer_vol.astype(np.float64), sigma=1.5)

                # Downsample
                if max(layer_vol.shape) > 192:
                    zoom_factors = [192 / s for s in layer_vol.shape]
                    layer_vol = ndimage.zoom(layer_vol, zoom_factors, order=1)
                    local_spacing = voxel_spacing / np.array(zoom_factors) if voxel_spacing is not None else None
                else:
                    local_spacing = voxel_spacing

                # Extract surface
                iso_level = config["hu_min"] + (config["hu_max"] - config["hu_min"]) * config["iso_offset"]
                try:
                    verts, faces, normals, _ = measure.marching_cubes(
                        layer_vol,
                        level=iso_level,
                        step_size=settings.MARCHING_CUBES_STEP_SIZE,
                        allow_degenerate=False,
                    )
                except ValueError:
                    logger.warning(f"Marching cubes failed for layer '{layer_name}', skipping")
                    continue

                if local_spacing is not None:
                    verts = verts * local_spacing

                center = (verts.max(axis=0) + verts.min(axis=0)) / 2
                verts -= center
                max_extent = np.abs(verts).max()
                if max_extent > 0:
                    verts = verts / max_extent * 100.0

                layer_path = str(mesh_dir_path / f"{scan_id}_{layer_name}.glb")
                self.mesh_gen.save_glb(verts, faces, normals, layer_path)
                layer_paths[layer_name] = layer_path
                logger.info(f"Layer mesh '{layer_name}': {len(verts)} verts")

            except Exception as exc:
                logger.error(f"Failed to generate layer '{layer_name}': {exc}")
                continue

        return layer_paths

    def extract_slice(self, volume_path: str, axis: str, index: int) -> SliceResponse:
        """
        Extract a 2D slice from the volume and return as base64-encoded PNG.
        """
        volume = np.load(volume_path)

        axis_map = {"axial": 0, "coronal": 1, "sagittal": 2}
        axis_idx = axis_map.get(axis, 0)
        total_slices = volume.shape[axis_idx]

        index = max(0, min(index, total_slices - 1))

        if axis_idx == 0:
            slice_2d = volume[index, :, :]
        elif axis_idx == 1:
            slice_2d = volume[:, index, :]
        else:
            slice_2d = volume[:, :, index]

        # Apply windowing for display
        window_center = 40.0
        window_width = 400.0
        img = self.volume_proc.apply_window(slice_2d, window_center, window_width)

        # Convert to 8-bit image
        img_normalized = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_normalized, mode="L")

        # Encode as PNG base64
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return SliceResponse(
            image_data=img_base64,
            axis=axis,
            index=index,
            total_slices=total_slices,
            window_center=window_center,
            window_width=window_width,
        )
