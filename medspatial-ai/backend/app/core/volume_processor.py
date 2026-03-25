"""
MedSpatial AI — Volume Processor
Windowing, normalization, and preprocessing utilities for 3D HU volumes.
"""

import numpy as np
from scipy import ndimage


class VolumeProcessor:
    """Utilities for processing 3D medical image volumes in Hounsfield Units."""

    # Standard CT window presets (center, width)
    WINDOW_PRESETS = {
        "lung": (-600, 1500),
        "mediastinum": (40, 400),
        "bone": (400, 1800),
        "brain": (40, 80),
        "soft_tissue": (50, 350),
        "liver": (60, 150),
        "abdomen": (60, 400),
    }

    def apply_window(
        self,
        data: np.ndarray,
        window_center: float = 40.0,
        window_width: float = 400.0,
    ) -> np.ndarray:
        """
        Apply CT windowing to convert HU values to display values.

        Args:
            data: numpy array in HU
            window_center: center of the display window
            window_width: width of the display window

        Returns:
            Windowed array in [0, 1] range
        """
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        windowed = np.clip(data, lower, upper)
        windowed = (windowed - lower) / (upper - lower)
        return windowed

    def apply_preset(self, data: np.ndarray, preset_name: str) -> np.ndarray:
        """Apply a named window preset."""
        if preset_name not in self.WINDOW_PRESETS:
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(self.WINDOW_PRESETS.keys())}")
        center, width = self.WINDOW_PRESETS[preset_name]
        return self.apply_window(data, center, width)

    def normalize(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume to [0, 1] range."""
        vmin, vmax = volume.min(), volume.max()
        if vmax - vmin < 1e-8:
            return np.zeros_like(volume, dtype=np.float32)
        return ((volume - vmin) / (vmax - vmin)).astype(np.float32)

    def clip_hu(
        self, volume: np.ndarray, hu_min: float = -1024.0, hu_max: float = 3071.0
    ) -> np.ndarray:
        """Clip HU values to a valid range."""
        return np.clip(volume, hu_min, hu_max)

    def denoise(self, volume: np.ndarray, sigma: float = 0.5) -> np.ndarray:
        """Apply mild Gaussian denoising."""
        return ndimage.gaussian_filter(volume, sigma=sigma)

    def resample(
        self,
        volume: np.ndarray,
        current_spacing: np.ndarray,
        target_spacing: np.ndarray = np.array([1.0, 1.0, 1.0]),
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Resample volume to isotropic spacing.

        Args:
            volume: 3D array
            current_spacing: [z, y, x] voxel spacing in mm
            target_spacing: desired [z, y, x] spacing

        Returns:
            Resampled volume and new spacing
        """
        zoom_factors = current_spacing / target_spacing
        resampled = ndimage.zoom(volume, zoom_factors, order=1)
        return resampled, target_spacing

    def resize_volume(
        self, volume: np.ndarray, target_size: int = 128
    ) -> np.ndarray:
        """Resize volume to target_size^3 for model input."""
        zoom_factors = [target_size / s for s in volume.shape]
        return ndimage.zoom(volume, zoom_factors, order=1)

    def extract_body_mask(self, volume: np.ndarray, threshold: float = -400.0) -> np.ndarray:
        """
        Create a binary mask separating body from air/background.
        Useful for focusing analysis on the patient's body.
        """
        binary = volume > threshold
        # Fill holes
        binary = ndimage.binary_fill_holes(binary)
        # Remove small objects
        labeled, num_features = ndimage.label(binary)
        if num_features > 0:
            sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
            largest = np.argmax(sizes) + 1
            binary = labeled == largest
        return binary.astype(np.float32)

    def compute_histogram(
        self, volume: np.ndarray, bins: int = 256, mask: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute HU histogram, optionally within a mask."""
        data = volume[mask > 0.5] if mask is not None else volume.ravel()
        hist, edges = np.histogram(data, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        return hist, centers
