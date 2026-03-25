"""
MedSpatial AI — Atlas Generator (generate_atlas.py)
Creates a synthetic but anatomically plausible chest atlas volume using procedural generation.
Outputs a compressed atlas_chest_128.npz with 12 anatomical layer masks.

Layer Classes:
  0: Background/Air
  1: Skin/Subcutaneous tissue
  2: Musculature (intercostals, pectorals, diaphragm)
  3: Skeletal structure (ribs, sternum, clavicles, scapulae, vertebrae)
  4: Pleural membrane
  5: Lung parenchyma (left lung)
  6: Lung parenchyma (right lung)
  7: Bronchial tree / Airways
  8: Pulmonary vasculature
  9: Heart / Cardiac silhouette
  10: Mediastinum (great vessels, esophagus, trachea)
  11: Abnormality/Pathology regions (empty in normal atlas)
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import ndimage


def create_ellipsoid(shape: Tuple[int, ...], center: Tuple[float, ...],
                     radii: Tuple[float, ...]) -> np.ndarray:
    """Create a 3D ellipsoid binary mask."""
    D, H, W = shape
    z, y, x = np.ogrid[:D, :H, :W]
    cz, cy, cx = center
    rz, ry, rx = radii
    dist = ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2
    return (dist <= 1.0).astype(np.float32)


def create_cylinder(shape: Tuple[int, ...], center_yx: Tuple[float, float],
                    radius: float, z_range: Tuple[int, int]) -> np.ndarray:
    """Create a vertical cylinder mask."""
    D, H, W = shape
    mask = np.zeros(shape, dtype=np.float32)
    z, y, x = np.ogrid[:D, :H, :W]
    cy, cx = center_yx
    circle = ((y - cy) ** 2 + (x - cx) ** 2) <= radius ** 2
    mask[z_range[0]:z_range[1]] = circle[0, :, :]
    return mask


def create_curved_rib(shape: Tuple[int, ...], z_pos: int, rib_radius: float,
                      center_y: float, center_x: float, thickness: float = 2.5,
                      arc_angle: float = 2.8) -> np.ndarray:
    """Create a single curved rib as an arc in the XY plane at given z."""
    D, H, W = shape
    mask = np.zeros(shape, dtype=np.float32)
    num_points = 120
    for t in np.linspace(-arc_angle / 2, arc_angle / 2, num_points):
        ry = center_y + rib_radius * np.cos(t) - rib_radius * 0.3
        rx = center_x + rib_radius * np.sin(t)
        iy, ix = int(round(ry)), int(round(rx))
        for dz in range(max(0, z_pos - 1), min(D, z_pos + 2)):
            for dy in range(max(0, iy - int(thickness)), min(H, iy + int(thickness) + 1)):
                for dx in range(max(0, ix - int(thickness)), min(W, ix + int(thickness) + 1)):
                    dist = np.sqrt((dz - z_pos) ** 2 + (dy - iy) ** 2 + (dx - ix) ** 2)
                    if dist <= thickness:
                        mask[dz, dy, dx] = 1.0
    return mask


def create_bronchial_tree(shape: Tuple[int, ...], start: Tuple[int, int, int],
                          max_depth: int = 5) -> np.ndarray:
    """Generate a branching bronchial tree using L-system principles."""
    D, H, W = shape
    mask = np.zeros(shape, dtype=np.float32)
    
    branches = [(start, np.array([0.0, 1.0, 0.0]), 3.0, 0)]  # pos, direction, radius, depth
    
    while branches:
        pos, direction, radius, depth = branches.pop(0)
        if depth >= max_depth or radius < 0.8:
            continue
        
        length = int(8 * (0.7 ** depth))
        for step in range(length):
            z, y, x = int(pos[0]), int(pos[1]), int(pos[2])
            if not (0 <= z < D and 0 <= y < H and 0 <= x < W):
                break
            
            r = max(1, int(radius))
            for dz in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        dist = np.sqrt(dz ** 2 + dy ** 2 + dx ** 2)
                        if dist <= radius:
                            nz, ny, nx = z + dz, y + dy, x + dx
                            if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                                mask[nz, ny, nx] = 1.0
            
            pos = pos + direction * 1.5
        
        # Branch into two sub-branches with angle divergence
        if depth < max_depth - 1:
            angle = 0.4 + 0.15 * depth
            for sign in [-1, 1]:
                new_dir = direction.copy()
                new_dir[2] += sign * angle
                new_dir[0] += np.random.uniform(-0.1, 0.1)
                new_dir = new_dir / (np.linalg.norm(new_dir) + 1e-8)
                branches.append((pos.copy(), new_dir, radius * 0.7, depth + 1))
    
    return mask


def generate_atlas(size: int = 128) -> dict:
    """
    Generate the complete 12-layer chest atlas volume.
    
    Returns:
        dict with keys: 'labels' (D,H,W int), 'probabilities' (12,D,H,W float),
                        'volume_hu' (D,H,W float), 'metadata' (dict)
    """
    shape = (size, size, size)
    D, H, W = shape
    cz, cy, cx = D // 2, H // 2, W // 2
    
    # Initialize all layers as empty
    layers = {i: np.zeros(shape, dtype=np.float32) for i in range(12)}
    
    # ── Layer 1: Skin (outer ellipsoid shell) ──────────────────
    body_outer = create_ellipsoid(shape, (cz, cy, cx), (cz * 0.88, cy * 0.78, cx * 0.72))
    body_inner = create_ellipsoid(shape, (cz, cy, cx), (cz * 0.83, cy * 0.73, cx * 0.67))
    layers[1] = np.clip(body_outer - body_inner, 0, 1)
    
    # ── Layer 2: Musculature (inner shell beneath skin) ────────
    muscle_inner = create_ellipsoid(shape, (cz, cy, cx), (cz * 0.78, cy * 0.68, cx * 0.62))
    layers[2] = np.clip(body_inner - muscle_inner, 0, 1)
    
    # ── Layer 3: Skeletal structures ──────────────────────────
    skeleton = np.zeros(shape, dtype=np.float32)
    
    # Spine (vertebral column)
    spine = create_cylinder(shape, (cy + cy * 0.55, cx), radius=3.5,
                           z_range=(int(D * 0.1), int(D * 0.9)))
    skeleton += spine
    
    # Sternum
    sternum = create_cylinder(shape, (cy - cy * 0.55, cx), radius=2.5,
                             z_range=(int(D * 0.2), int(D * 0.7)))
    skeleton += sternum
    
    # Ribs (12 pairs)
    rib_positions = np.linspace(D * 0.15, D * 0.85, 12).astype(int)
    for z_pos in rib_positions:
        rib_r = cy * 0.55
        skeleton += create_curved_rib(shape, z_pos, rib_r, cy, cx, thickness=2.0)
    
    # Clavicles
    for sign in [-1, 1]:
        clav_mask = np.zeros(shape, dtype=np.float32)
        for t in np.linspace(0, 1, 40):
            y_pos = int(cy - cy * 0.4)
            x_pos = int(cx + sign * t * cx * 0.5)
            z_pos = int(D * 0.18 + t * 3)
            for dz in range(-1, 2):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        nz, ny, nx = z_pos + dz, y_pos + dy, x_pos + dx
                        if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                            clav_mask[nz, ny, nx] = 1.0
        skeleton += clav_mask
    
    layers[3] = np.clip(skeleton, 0, 1)
    
    # ── Layer 4: Pleural membrane (thin shell around lungs) ───
    lung_outer_left = create_ellipsoid(shape, (cz, cy, cx + cx * 0.28),
                                       (cz * 0.55, cy * 0.42, cx * 0.28))
    lung_outer_right = create_ellipsoid(shape, (cz, cy, cx - cx * 0.28),
                                        (cz * 0.57, cy * 0.44, cx * 0.30))
    lung_inner_left = create_ellipsoid(shape, (cz, cy, cx + cx * 0.28),
                                       (cz * 0.53, cy * 0.40, cx * 0.26))
    lung_inner_right = create_ellipsoid(shape, (cz, cy, cx - cx * 0.28),
                                        (cz * 0.55, cy * 0.42, cx * 0.28))
    layers[4] = np.clip((lung_outer_left - lung_inner_left) +
                        (lung_outer_right - lung_inner_right), 0, 1)
    
    # ── Layer 5: Left Lung ────────────────────────────────────
    layers[5] = lung_inner_left
    
    # ── Layer 6: Right Lung ───────────────────────────────────
    layers[6] = lung_inner_right
    
    # ── Layer 7: Bronchial tree ───────────────────────────────
    trachea_start = (int(D * 0.12), int(cy * 0.95), cx)
    layers[7] = create_bronchial_tree(shape, trachea_start, max_depth=5)
    # Clip to lung regions
    lung_mask = np.clip(layers[5] + layers[6], 0, 1)
    trachea_region = create_cylinder(shape, (cy, cx), radius=4,
                                     z_range=(int(D * 0.1), int(D * 0.35)))
    layers[7] = np.clip(layers[7] * (lung_mask + trachea_region), 0, 1)
    
    # ── Layer 8: Pulmonary vasculature ────────────────────────
    # Generate as a subset of the lung field with branching patterns
    vasc = np.zeros(shape, dtype=np.float32)
    np.random.seed(42)
    for _ in range(150):
        pos = np.array([
            np.random.randint(int(D * 0.2), int(D * 0.8)),
            np.random.randint(int(H * 0.2), int(H * 0.8)),
            np.random.randint(int(W * 0.2), int(W * 0.8)),
        ])
        if lung_mask[pos[0], pos[1], pos[2]] > 0.5:
            r = np.random.uniform(1.0, 2.5)
            for dz in range(-3, 4):
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        dist = np.sqrt(dz ** 2 + dy ** 2 + dx ** 2)
                        if dist <= r:
                            nz, ny, nx = pos[0] + dz, pos[1] + dy, pos[2] + dx
                            if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                                vasc[nz, ny, nx] = 1.0
    layers[8] = vasc * lung_mask
    
    # ── Layer 9: Heart ────────────────────────────────────────
    layers[9] = create_ellipsoid(shape, (cz + 2, cy + 3, cx + cx * 0.12),
                                 (cz * 0.22, cy * 0.25, cx * 0.20))
    
    # ── Layer 10: Mediastinum ─────────────────────────────────
    mediastinum = create_ellipsoid(shape, (cz, cy, cx), (cz * 0.6, cy * 0.2, cx * 0.12))
    # Remove heart overlap
    layers[10] = np.clip(mediastinum - layers[9], 0, 1)
    
    # ── Layer 11: Pathology (empty in normal atlas) ───────────
    layers[11] = np.zeros(shape, dtype=np.float32)
    
    # ── Layer 0: Background/Air (everything else) ─────────────
    all_tissue = np.zeros(shape, dtype=np.float32)
    for i in range(1, 12):
        all_tissue = np.clip(all_tissue + layers[i], 0, 1)
    layers[0] = 1.0 - np.clip(all_tissue, 0, 1)
    
    # ── Resolve overlaps via priority ─────────────────────────
    # Higher layer index = higher priority (pathology > organs > skeleton > muscle > skin > air)
    label_volume = np.zeros(shape, dtype=np.int32)
    for i in range(12):
        label_volume[layers[i] > 0.5] = i
    
    # ── Create probability volume (soft boundaries) ───────────
    prob_volume = np.zeros((12,) + shape, dtype=np.float32)
    for i in range(12):
        smoothed = ndimage.gaussian_filter(layers[i], sigma=1.0)
        prob_volume[i] = smoothed
    # Normalize to sum to 1
    total = prob_volume.sum(axis=0, keepdims=True)
    total = np.where(total < 1e-8, 1.0, total)
    prob_volume = prob_volume / total
    
    # ── Create HU volume ──────────────────────────────────────
    hu_map = {
        0: -1000.0,   # Air
        1: -50.0,     # Skin/fat
        2: 40.0,      # Muscle
        3: 800.0,     # Bone
        4: 20.0,      # Pleura
        5: -700.0,    # Left lung
        6: -700.0,    # Right lung
        7: -950.0,    # Airways
        8: 60.0,      # Vessels
        9: 45.0,      # Heart
        10: 35.0,     # Mediastinum
        11: 50.0,     # Pathology
    }
    volume_hu = np.zeros(shape, dtype=np.float32)
    for i in range(12):
        volume_hu[label_volume == i] = hu_map[i]
    # Add realistic noise
    volume_hu += np.random.normal(0, 8, shape).astype(np.float32)
    
    metadata = {
        "structure": "chest",
        "size": size,
        "num_layers": 12,
        "layer_names": [
            "Background/Air", "Skin", "Musculature", "Skeleton",
            "Pleural Membrane", "Left Lung", "Right Lung", "Bronchial Tree",
            "Pulmonary Vasculature", "Heart", "Mediastinum", "Pathology"
        ],
        "hu_ranges": hu_map,
    }
    
    return {
        "labels": label_volume,
        "probabilities": prob_volume,
        "volume_hu": volume_hu,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate MedSpatial AI chest atlas")
    parser.add_argument("--size", type=int, default=128, help="Atlas volume size (NxNxN)")
    parser.add_argument("--output", type=str, default="atlas_chest_128.npz", help="Output file")
    parser.add_argument("--output-dir", type=str, default="./data", help="Output directory")
    args = parser.parse_args()
    
    print(f"Generating {args.size}³ chest atlas...")
    atlas = generate_atlas(args.size)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output
    
    np.savez_compressed(
        str(out_path),
        labels=atlas["labels"],
        probabilities=atlas["probabilities"],
        volume_hu=atlas["volume_hu"],
    )
    
    file_size = out_path.stat().st_size / (1024 * 1024)
    print(f"Atlas saved: {out_path} ({file_size:.1f} MB)")
    print(f"Label volume shape: {atlas['labels'].shape}")
    print(f"Probability volume shape: {atlas['probabilities'].shape}")
    print(f"HU volume range: [{atlas['volume_hu'].min():.0f}, {atlas['volume_hu'].max():.0f}]")
    print(f"Layer distribution:")
    for i, name in enumerate(atlas["metadata"]["layer_names"]):
        count = (atlas["labels"] == i).sum()
        pct = count / atlas["labels"].size * 100
        print(f"  [{i:2d}] {name:25s}: {count:8d} voxels ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
