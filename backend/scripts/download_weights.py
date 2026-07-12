"""
MedSpatial AI — Model Weights Downloader
Downloads pre-trained weights on first run and caches locally.
Falls back gracefully when offline.
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# ── Weight manifest ──────────────────────────────────────────────
# Each entry: (filename, url, sha256, size_mb, required)
WEIGHT_MANIFEST = [
    # sentence-transformers all-MiniLM-L6-v2 tokenizer config (lightweight)
    (
        "minilm_config.json",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json",
        None,  # skip hash check for config files
        0.001,
        False,
    ),
]

# In a real deployment, large weights (EfficientViT, etc.) would be listed here.
# We pre-generate the memory bank from the atlas so no large downloads are needed.


def download_file(url: str, dest: Path, expected_sha256: str | None = None) -> bool:
    """Download a single file with progress reporting. Returns True on success."""
    try:
        print(f"  ⬇️  {dest.name} ...", end="", flush=True)
        
        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(100, block_num * block_size * 100 // total_size)
                print(f"\r  ⬇️  {dest.name} ... {pct}%", end="", flush=True)

        urllib.request.urlretrieve(url, str(dest), reporthook=show_progress)
        print(f"\r  ✅ {dest.name}")
        
        if expected_sha256:
            sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()
            if sha256 != expected_sha256:
                print(f"  ⚠️  Hash mismatch for {dest.name}, re-downloading may be needed")
                return False
        return True

    except Exception as exc:
        print(f"\r  ❌ Failed: {dest.name} — {exc}")
        return False


def generate_memory_bank(output_dir: Path, atlas_path: Path) -> None:
    """Generate the anomaly detector memory bank from the atlas volume."""
    memory_bank_path = output_dir / "memory_bank.npz"
    if memory_bank_path.exists():
        print(f"  ✅ memory_bank.npz (already exists)")
        return

    print("  🧠 Generating anomaly detector memory bank from atlas...")
    try:
        import numpy as np

        if atlas_path.exists():
            atlas = np.load(str(atlas_path))
            vol = atlas["volume_hu"]
        else:
            print("  ⚠️  Atlas not found, generating random memory bank (run 'make atlas' first)")
            vol = np.random.randn(128, 128, 128).astype(np.float32)

        # Normalize
        vol_norm = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)

        # Extract 8×8×8 patches (stride 8 = non-overlapping → 16³ = 4096 patches)
        patch_size = 8
        D, H, W = vol_norm.shape
        patches = []
        for dz in range(0, D - patch_size + 1, patch_size):
            for dy in range(0, H - patch_size + 1, patch_size):
                for dx in range(0, W - patch_size + 1, patch_size):
                    p = vol_norm[dz:dz+patch_size, dy:dy+patch_size, dx:dx+patch_size]
                    patches.append(p.flatten())

        patch_array = np.array(patches, dtype=np.float32)  # (N, 512)

        # K-means to get 1024 normal embeddings
        from scipy.cluster.vq import kmeans
        print(f"  🔄 K-means clustering {len(patches)} patches → 1024 centroids...")
        centroids, _ = kmeans(patch_array, min(1024, len(patches)), iter=20)

        np.savez_compressed(str(memory_bank_path), centroids=centroids)
        print(f"  ✅ memory_bank.npz  ({memory_bank_path.stat().st_size/1024:.0f} KB)")

    except Exception as exc:
        print(f"  ⚠️  Memory bank generation failed: {exc}")
        # Fallback: random normal embeddings
        import numpy as np
        centroids = np.random.randn(1024, 512).astype(np.float32)
        centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        np.savez_compressed(str(memory_bank_path), centroids=centroids)
        print(f"  ⚠️  Saved random memory bank as fallback")


def main():
    parser = argparse.ArgumentParser(description="MedSpatial AI weight downloader")
    parser.add_argument("--output-dir", type=str, default="./models", help="Directory to save weights")
    parser.add_argument("--atlas", type=str, default="./data/atlas_chest_128.npz")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atlas_path = Path(args.atlas)

    print("📦 MedSpatial AI — Weight Initialization")
    print(f"   Output: {out_dir.resolve()}")
    print()

    # Download optional remote weights
    downloaded = 0
    skipped = 0
    for filename, url, sha256, size_mb, required in WEIGHT_MANIFEST:
        dest = out_dir / filename
        if dest.exists() and not args.force:
            print(f"  ✅ {filename} (cached)")
            skipped += 1
            continue
        success = download_file(url, dest, sha256)
        if success:
            downloaded += 1
        elif required:
            print(f"  ❌ Required weight {filename} could not be downloaded")
            sys.exit(1)

    # Generate memory bank from atlas
    generate_memory_bank(out_dir, atlas_path)

    print()
    print(f"✅ Weights ready: {downloaded} downloaded, {skipped} cached")
    print(f"   Location: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
