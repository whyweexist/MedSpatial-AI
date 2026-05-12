"""
MedSpatial AI — DICOM Service
Parses DICOM files, extracts metadata, and constructs 3D numpy volumes.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from loguru import logger


class DicomService:
    """Handles DICOM file parsing, metadata extraction, and volume assembly."""

    def extract_metadata(self, file_paths: list[Path]) -> dict[str, Any]:
        """Extract representative metadata from a set of DICOM files."""
        if not file_paths:
            return {}

        # # Read the first file for series-level metadata
        # ds = pydicom.dcmread(str(file_paths[0]), stop_before_pixels=True)
        try:
            # Read the first file for series-level metadata
            ds = pydicom.dcmread(str(file_paths[0]), stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError:
            # Fallback for generic images (PNG, JPEG, etc.)
            import skimage.io
            try:
                img = skimage.io.imread(str(file_paths[0]))
                return {
                    "modality": "XR",
                    "body_part": "Unknown",
                    "rows": int(img.shape[0]),
                    "columns": int(img.shape[1]),
                    "patient_id": "image_upload",
                    "study_description": "Generic Image Upload",
                    "series_description": "Image",
                    "slice_thickness": 1.0,
                    "bits_stored": 8,
                }
            except Exception:
                return {}

        metadata = {
            "patient_id": str(getattr(ds, "PatientID", "")),
            "patient_name": str(getattr(ds, "PatientName", "")),
            "study_description": str(getattr(ds, "StudyDescription", "")),
            "series_description": str(getattr(ds, "SeriesDescription", "")),
            "modality": str(getattr(ds, "Modality", "")),
            "body_part": str(getattr(ds, "BodyPartExamined", "")),
            "study_date": str(getattr(ds, "StudyDate", "")),
            "institution": str(getattr(ds, "InstitutionName", "")),
            "manufacturer": str(getattr(ds, "Manufacturer", "")),
            "rows": int(getattr(ds, "Rows", 0)),
            "columns": int(getattr(ds, "Columns", 0)),
        }

        # Pixel spacing
        pixel_spacing = getattr(ds, "PixelSpacing", None)
        if pixel_spacing:
            metadata["pixel_spacing_x"] = float(pixel_spacing[0])
            metadata["pixel_spacing_y"] = float(pixel_spacing[1])

        # Slice thickness
        metadata["slice_thickness"] = float(getattr(ds, "SliceThickness", 1.0))

        # Window center/width for display
        metadata["window_center"] = float(getattr(ds, "WindowCenter", 0))
        metadata["window_width"] = float(getattr(ds, "WindowWidth", 1000))

        # Bits stored
        metadata["bits_stored"] = int(getattr(ds, "BitsStored", 16))

        logger.info(f"Extracted metadata: modality={metadata['modality']}, body_part={metadata['body_part']}")
        return metadata

    def load_dicom_series(self, directory: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load all DICOM files from a directory, sort by slice position,
        and assemble into a 3D numpy volume.

        Returns:
            volume: 3D numpy array in Hounsfield Units (HU)
            voxel_spacing: array [z_spacing, y_spacing, x_spacing]
        """
        dicom_files = []
        dir_path = Path(directory)

        for fname in os.listdir(directory):
            fpath = dir_path / fname
            if fpath.is_file():
                try:
                    ds = pydicom.dcmread(str(fpath))
                    if hasattr(ds, "pixel_array"):
                        dicom_files.append(ds)
                except Exception:
                    continue

        if not dicom_files:
            # raise ValueError(f"No valid DICOM files found in {directory}")
            image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            image_files = []
            for fname in os.listdir(directory):
                if Path(fname).suffix.lower() in image_extensions:
                    image_files.append(dir_path / fname)

            if not image_files:
                raise ValueError(f"No valid DICOM or Image files found in {directory}")

            import skimage.io
            from skimage.color import rgb2gray
            
            # Sort images to maintain slice order if it's a volume
            image_files.sort()
            
            slices = []
            for item in image_files:
                img = skimage.io.imread(str(item))
                if img.ndim == 3:
                    # Convert RGB to grayscale (values 0-1)
                    img = rgb2gray(img)
                elif img.max() > 1.0:
                    # Normalize if already grayscale but outside 0-1
                    img = img.astype(np.float32) / 255.0

                # Map to pseudo-HU for X-ray / general image [-1000 to +1000 or similar]
                # Dark areas in typical images might be air (low HU), bright areas bone (high HU)
                # If it's a photo, medical defaults might invert or shift this, but let's map [0,1] → [-1000, 1000]
                img_hu = img * 2000.0 - 1000.0
                slices.append(img_hu.astype(np.float32))

            volume = np.stack(slices, axis=0)
            voxel_spacing = np.array([1.0, 1.0, 1.0])
            logger.info(f"Loaded {len(slices)} generic image(s) into volume: shape={volume.shape}")
            return volume, voxel_spacing

        # Sort by ImagePositionPatient[2] (z-axis) or InstanceNumber
        def sort_key(ds):
            if hasattr(ds, "ImagePositionPatient"):
                return float(ds.ImagePositionPatient[2])
            return int(getattr(ds, "InstanceNumber", 0))

        dicom_files.sort(key=sort_key)

        # Extract voxel spacing
        first = dicom_files[0]
        pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
        slice_thickness = float(getattr(first, "SliceThickness", 1.0))

        # If we have more than one slice, compute actual z-spacing from positions
        if len(dicom_files) > 1 and hasattr(dicom_files[0], "ImagePositionPatient"):
            z0 = float(dicom_files[0].ImagePositionPatient[2])
            z1 = float(dicom_files[1].ImagePositionPatient[2])
            z_spacing = abs(z1 - z0)
            if z_spacing > 0:
                slice_thickness = z_spacing

        voxel_spacing = np.array([
            slice_thickness,
            float(pixel_spacing[0]),
            float(pixel_spacing[1]),
        ])

        # Build 3D volume. Handle single-frame (2D) and multi-frame (3D) DICOMs safely.
        slices = []
        for ds in dicom_files:
            pixel_array = ds.pixel_array
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))

            if pixel_array.ndim == 2:
                hu_slice = pixel_array.astype(np.float32) * slope + intercept
                slices.append(hu_slice)

            elif pixel_array.ndim == 3:
                # Multi-frame DICOM may already contain a 3D volume in one file.
                # Flatten into a consistent list of 2D slices.
                for frame in pixel_array:
                    hu_frame = frame.astype(np.float32) * slope + intercept
                    slices.append(hu_frame)

            else:
                raise ValueError(
                    f"Unsupported pixel array dimensions {pixel_array.shape} in DICOM series"
                )

        if not slices:
            raise ValueError(f"No image frames available in DICOM series at {directory}")

        # volume = np.stack(slices, axis=0).astype(np.float32)
        # Filter out scout/localizer slices by enforcing the most common shape
        from collections import Counter
        shapes = [s.shape for s in slices]
        most_common_shape = Counter(shapes).most_common(1)[0][0]
        
        filtered_slices = [s for s in slices if s.shape == most_common_shape]
        
        if len(filtered_slices) < len(slices):
            logger.warning(f"Discarded {len(slices) - len(filtered_slices)} slices with non-matching shapes (e.g. scouts).")

        volume = np.stack(filtered_slices, axis=0).astype(np.float32)


        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume but got shape {volume.shape}")

        logger.info(f"Built volume: shape={volume.shape}, spacing={voxel_spacing}")
        return volume, voxel_spacing

    def load_single_image(self, file_path: str) -> tuple[np.ndarray, dict]:
        """
        Load a single DICOM image (e.g., X-ray) and return as 2D array + metadata.
        For X-rays, we create a pseudo-3D volume by repeating the image.
        """
        ds = pydicom.dcmread(file_path)
        pixel_array = ds.pixel_array.astype(np.float64)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        image_hu = pixel_array * slope + intercept

        metadata = self.extract_metadata([Path(file_path)])

        # For X-rays, create a pseudo-volume by stacking the image
        if len(image_hu.shape) == 2:
            # Generate pseudo-depth by creating gradient-based layers
            depth = 64
            pseudo_volume = np.zeros((depth, image_hu.shape[0], image_hu.shape[1]), dtype=np.float64)
            for i in range(depth):
                # Create depth attenuation to simulate 3D structure
                attenuation = np.exp(-0.05 * abs(i - depth // 2))
                pseudo_volume[i] = image_hu * attenuation
            return pseudo_volume, metadata

        return image_hu, metadata
