"""
MedSpatial AI — Synthetic Training Data Generator
Creates synthetic training pairs by deforming the atlas volume and injecting pathologies.
Enables model training without requiring real medical data.
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import ndimage


def random_elastic_deformation(volume: np.ndarray, sigma: float = 8.0,
                                alpha: float = 15.0, seed: int = None) -> np.ndarray:
    """Apply random elastic deformation to a 3D volume."""
    if seed is not None:
        np.random.seed(seed)
    D, H, W = volume.shape
    dz = ndimage.gaussian_filter(np.random.randn(D, H, W) * alpha, sigma)
    dy = ndimage.gaussian_filter(np.random.randn(D, H, W) * alpha, sigma)
    dx = ndimage.gaussian_filter(np.random.randn(D, H, W) * alpha, sigma)
    
    z, y, x = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    coords = [
        np.clip(z + dz, 0, D - 1),
        np.clip(y + dy, 0, H - 1),
        np.clip(x + dx, 0, W - 1),
    ]
    return ndimage.map_coordinates(volume, coords, order=1, mode='reflect')


def inject_nodule(volume: np.ndarray, labels: np.ndarray, lung_layers: list = [5, 6],
                  hu_value: float = 40.0, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Inject a spherical nodule into a random lung location."""
    if seed is not None:
        np.random.seed(seed)
    
    D, H, W = volume.shape
    lung_mask = np.zeros_like(labels, dtype=bool)
    for layer in lung_layers:
        lung_mask |= (labels == layer)
    
    lung_coords = np.argwhere(lung_mask)
    if len(lung_coords) == 0:
        return volume, labels
    
    center_idx = np.random.randint(len(lung_coords))
    center = lung_coords[center_idx]
    radius = np.random.uniform(3.0, 8.0)
    
    z, y, x = np.ogrid[:D, :H, :W]
    dist = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
    nodule_mask = dist <= radius
    
    # Smooth edges
    nodule_smooth = ndimage.gaussian_filter(nodule_mask.astype(np.float32), sigma=1.0)
    
    volume_out = volume.copy()
    labels_out = labels.copy()
    volume_out[nodule_smooth > 0.3] = hu_value + np.random.normal(0, 10, nodule_smooth[nodule_smooth > 0.3].shape)
    labels_out[nodule_smooth > 0.5] = 11  # Pathology layer
    
    return volume_out, labels_out


def inject_ground_glass(volume: np.ndarray, labels: np.ndarray,
                         lung_layers: list = [5, 6], seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Inject ground-glass opacity into a lung region."""
    if seed is not None:
        np.random.seed(seed)
    
    D, H, W = volume.shape
    lung_mask = np.zeros_like(labels, dtype=bool)
    for layer in lung_layers:
        lung_mask |= (labels == layer)
    
    lung_coords = np.argwhere(lung_mask)
    if len(lung_coords) == 0:
        return volume, labels
    
    center_idx = np.random.randint(len(lung_coords))
    center = lung_coords[center_idx]
    radius = np.random.uniform(8.0, 18.0)
    
    z, y, x = np.ogrid[:D, :H, :W]
    dist = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
    gg_region = dist <= radius
    gg_region = gg_region & lung_mask
    
    volume_out = volume.copy()
    labels_out = labels.copy()
    # Ground glass: partially increased density (-700 → -400 range)
    volume_out[gg_region] = volume[gg_region] + np.random.uniform(200, 400, gg_region.sum())
    labels_out[gg_region] = 11
    
    return volume_out, labels_out


def inject_pleural_effusion(volume: np.ndarray, labels: np.ndarray,
                             side: str = 'right', seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Inject fluid collection at the base of a lung."""
    if seed is not None:
        np.random.seed(seed)
    
    D, H, W = volume.shape
    target_layer = 6 if side == 'right' else 5
    lung_mask = labels == target_layer
    
    # Take the lower portion of the lung
    fluid_height = np.random.uniform(0.15, 0.35)
    z_threshold = int(D * (1 - fluid_height))
    fluid_region = lung_mask.copy()
    fluid_region[:z_threshold, :, :] = False
    
    volume_out = volume.copy()
    labels_out = labels.copy()
    volume_out[fluid_region] = np.random.uniform(5, 20, fluid_region.sum())
    labels_out[fluid_region] = 11
    
    return volume_out, labels_out


def generate_synthetic_dataset(atlas_path: str, output_dir: str, num_samples: int = 100,
                                include_normal: bool = True) -> None:
    """
    Generate a synthetic training dataset from the atlas volume.
    
    Args:
        atlas_path: path to atlas_chest_128.npz
        output_dir: directory to save synthetic samples
        num_samples: number of samples to generate
        include_normal: whether to include normal (no pathology) samples
    """
    print(f"Loading atlas from {atlas_path}...")
    atlas = np.load(atlas_path)
    atlas_hu = atlas['volume_hu']
    atlas_labels = atlas['labels']
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pathology_types = ['nodule', 'ground_glass', 'pleural_effusion', 'normal']
    disease_labels_map = {
        'nodule': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Lung Nodule
        'ground_glass': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Pneumonia
        'pleural_effusion': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Pleural Effusion
        'normal': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Normal
    }
    
    print(f"Generating {num_samples} synthetic samples...")
    
    for i in range(num_samples):
        seed = i * 42
        
        # Apply elastic deformation to atlas
        deformed_hu = random_elastic_deformation(atlas_hu, sigma=6, alpha=10, seed=seed)
        deformed_labels = random_elastic_deformation(
            atlas_labels.astype(np.float32), sigma=6, alpha=10, seed=seed
        ).round().astype(np.int32)
        deformed_labels = np.clip(deformed_labels, 0, 11)
        
        # Add intensity augmentation
        intensity_shift = np.random.uniform(-30, 30)
        intensity_scale = np.random.uniform(0.9, 1.1)
        deformed_hu = deformed_hu * intensity_scale + intensity_shift
        
        # Choose pathology type
        if include_normal and np.random.rand() < 0.3:
            path_type = 'normal'
        else:
            path_type = np.random.choice(['nodule', 'ground_glass', 'pleural_effusion'])
        
        volume = deformed_hu.copy()
        labels = deformed_labels.copy()
        
        if path_type == 'nodule':
            num_nodules = np.random.randint(1, 4)
            for j in range(num_nodules):
                volume, labels = inject_nodule(volume, labels, seed=seed + j)
        elif path_type == 'ground_glass':
            volume, labels = inject_ground_glass(volume, labels, seed=seed)
        elif path_type == 'pleural_effusion':
            side = np.random.choice(['left', 'right'])
            volume, labels = inject_pleural_effusion(volume, labels, side=side, seed=seed)
        
        # Add noise
        volume += np.random.normal(0, 5, volume.shape).astype(np.float32)
        
        sample_path = out_dir / f"sample_{i:04d}.npz"
        np.savez_compressed(
            str(sample_path),
            volume=volume,
            labels=labels,
            disease_labels=np.array(disease_labels_map[path_type], dtype=np.int32),
            pathology_type=path_type,
        )
        
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i + 1}/{num_samples}] {path_type} → {sample_path.name}")
    
    print(f"Dataset complete: {num_samples} samples saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--atlas", type=str, default="./data/atlas_chest_128.npz")
    parser.add_argument("--output", type=str, default="./data/synthetic_training")
    parser.add_argument("--num-samples", type=int, default=1000)
    args = parser.parse_args()
    
    generate_synthetic_dataset(args.atlas, args.output, args.num_samples)


if __name__ == "__main__":
    main()
