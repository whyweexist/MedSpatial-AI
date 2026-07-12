"""
MedSpatial AI — Slice Sorter
Robust DICOM slice ordering: sorts by ImagePositionPatient, detects scouts,
removes duplicates, and computes accurate z-spacing.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
from loguru import logger


class SliceSorter:
    """Sorts DICOM slices into a consistent spatial order for volume assembly."""

    def sort_datasets(
        self, datasets: list[pydicom.Dataset]
    ) -> tuple[list[pydicom.Dataset], float]:
        """
        Sort DICOM datasets by slice position and compute z-spacing.

        Args:
            datasets: list of pydicom Dataset objects with pixel data.

        Returns:
            sorted_datasets: spatially ordered datasets (inferior→superior or vice versa)
            z_spacing: computed inter-slice spacing in mm
        """
        if not datasets:
            raise ValueError("No DICOM datasets provided for sorting")

        if len(datasets) == 1:
            z_spacing = float(getattr(datasets[0], "SliceThickness", 1.0))
            return datasets, z_spacing

        # Remove scouts / localizer series
        datasets = self._remove_scouts(datasets)

        if not datasets:
            raise ValueError("All slices were identified as scouts/localizers")

        # Remove duplicates
        datasets = self._remove_duplicates(datasets)

        # Sort by position
        datasets = self._sort_by_position(datasets)

        # Compute z-spacing
        z_spacing = self._compute_z_spacing(datasets)

        logger.info(
            f"SliceSorter: {len(datasets)} slices sorted, z-spacing={z_spacing:.3f} mm"
        )
        return datasets, z_spacing

    def _remove_scouts(
        self, datasets: list[pydicom.Dataset]
    ) -> list[pydicom.Dataset]:
        """Remove scout/localizer images that differ in dimensions or are tagged."""
        if len(datasets) <= 2:
            return datasets

        # Detect by ImageType tag containing LOCALIZER
        filtered: list[pydicom.Dataset] = []
        for ds in datasets:
            image_type = getattr(ds, "ImageType", [])
            if isinstance(image_type, pydicom.multival.MultiValue):
                image_type = list(image_type)
            elif isinstance(image_type, str):
                image_type = [image_type]
            else:
                image_type = list(image_type) if image_type else []

            type_str = " ".join(str(t).upper() for t in image_type)
            if "LOCALIZER" in type_str or "SCOUT" in type_str:
                logger.info(f"Removed scout/localizer: ImageType={image_type}")
                continue
            filtered.append(ds)

        if not filtered:
            return datasets  # don't remove everything

        # Also detect by frame size mismatch (scouts often have different dims)
        row_counts: dict[tuple[int, int], int] = {}
        for ds in filtered:
            dims = (int(getattr(ds, "Rows", 0)), int(getattr(ds, "Columns", 0)))
            row_counts[dims] = row_counts.get(dims, 0) + 1

        if len(row_counts) > 1:
            # Keep only the most common frame size
            most_common_dims = max(row_counts, key=row_counts.get)
            before = len(filtered)
            filtered = [
                ds
                for ds in filtered
                if (
                    int(getattr(ds, "Rows", 0)),
                    int(getattr(ds, "Columns", 0)),
                )
                == most_common_dims
            ]
            if len(filtered) < before:
                logger.info(
                    f"Removed {before - len(filtered)} slices with mismatched dimensions"
                )

        return filtered

    def _remove_duplicates(
        self, datasets: list[pydicom.Dataset]
    ) -> list[pydicom.Dataset]:
        """Remove slices with duplicate z-positions, keeping highest InstanceNumber."""
        if not any(hasattr(ds, "ImagePositionPatient") for ds in datasets):
            return datasets

        position_map: dict[float, list[pydicom.Dataset]] = {}
        for ds in datasets:
            if hasattr(ds, "ImagePositionPatient"):
                z_pos = round(float(ds.ImagePositionPatient[2]), 3)
            else:
                z_pos = float(getattr(ds, "InstanceNumber", 0))
            if z_pos not in position_map:
                position_map[z_pos] = []
            position_map[z_pos].append(ds)

        unique: list[pydicom.Dataset] = []
        duplicates_removed = 0
        for z_pos, group in position_map.items():
            if len(group) == 1:
                unique.append(group[0])
            else:
                # Keep the one with highest InstanceNumber (most recent acquisition)
                best = max(
                    group, key=lambda d: int(getattr(d, "InstanceNumber", 0))
                )
                unique.append(best)
                duplicates_removed += len(group) - 1

        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate slice positions")

        return unique

    def _sort_by_position(
        self, datasets: list[pydicom.Dataset]
    ) -> list[pydicom.Dataset]:
        """Sort datasets by z-position (ImagePositionPatient) or InstanceNumber."""

        def sort_key(ds: pydicom.Dataset) -> float:
            if hasattr(ds, "ImagePositionPatient"):
                return float(ds.ImagePositionPatient[2])
            if hasattr(ds, "SliceLocation"):
                return float(ds.SliceLocation)
            return float(getattr(ds, "InstanceNumber", 0))

        return sorted(datasets, key=sort_key)

    def _compute_z_spacing(self, datasets: list[pydicom.Dataset]) -> float:
        """Compute actual inter-slice spacing from consecutive z-positions."""
        if len(datasets) < 2:
            return float(getattr(datasets[0], "SliceThickness", 1.0))

        # Try ImagePositionPatient first
        if hasattr(datasets[0], "ImagePositionPatient") and hasattr(
            datasets[1], "ImagePositionPatient"
        ):
            positions = [
                float(ds.ImagePositionPatient[2])
                for ds in datasets
                if hasattr(ds, "ImagePositionPatient")
            ]
            if len(positions) >= 2:
                spacings = [
                    abs(positions[i + 1] - positions[i])
                    for i in range(len(positions) - 1)
                ]
                spacings = [s for s in spacings if s > 0.01]  # filter zero gaps
                if spacings:
                    median_spacing = float(np.median(spacings))
                    if median_spacing > 0:
                        return median_spacing

        # Fallback to SliceThickness
        return float(getattr(datasets[0], "SliceThickness", 1.0))

    def detect_modality(self, datasets: list[pydicom.Dataset]) -> str:
        """Detect imaging modality from DICOM metadata."""
        if not datasets:
            return "UNKNOWN"
        modality = str(getattr(datasets[0], "Modality", "")).upper()
        if modality in ("CT", "MR", "XR", "CR", "DX", "MG", "PT", "NM"):
            return modality
        return "UNKNOWN"

    def is_single_frame(self, datasets: list[pydicom.Dataset]) -> bool:
        """Check if this is a single-frame study (e.g., X-ray)."""
        if len(datasets) == 1:
            ds = datasets[0]
            pixel_array = getattr(ds, "pixel_array", None)
            if pixel_array is not None and pixel_array.ndim == 2:
                return True
        return False
