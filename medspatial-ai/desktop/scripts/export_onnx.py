"""
MedSpatial AI — PyTorch → ONNX Export Script
=============================================
Exports all three PyTorch models to ONNX format suitable for CPU inference
via ONNX Runtime. No PyTorch required at runtime after this step.

Models exported:
  1. SpatialTransformer3D  → spatial_transformer.onnx
  2. SegmentationNet3D     → segmentation_net.onnx
  3. AnomalyDetector3D     → anomaly_detector.onnx  (inference-mode VAE)

Usage:
  # From repo root (medspatial-ai/)
  python desktop/scripts/export_onnx.py \
      --model-dir models/ \
      --output-dir desktop/onnx_models/ \
      --volume-size 128

  # Export a single model
  python desktop/scripts/export_onnx.py --only spatial_transformer

  # With FP16 conversion (smaller, still CPU-compatible via ORT)
  python desktop/scripts/export_onnx.py --fp16

Requirements (export machine only — not needed at runtime):
  pip install torch==2.2.0 onnx==1.16.0 onnxscript

Notes on ONNX dynamic shapes:
  - Batch dimension is always dynamic (dim=0).
  - Spatial dims (D/H/W) are FIXED at VOLUME_SIZE for production to ensure
    deterministic graph shapes. The pre-processing pipeline always resizes
    inputs to this size before calling the model.
  - The AnomalyDetector uses eval() mode which makes reparameterize()
    deterministic: eps=0 → z = mu. This is the standard inference-time
    convention for VAEs and gives stable, reproducible anomaly maps.
"""

import argparse
import sys
import os
import logging
from pathlib import Path

import torch
import torch.nn as nn
import onnx

# ---------------------------------------------------------------------------
# Ensure backend package is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.ai.spatial_transformer import create_spatial_transformer, SpatialTransformer3D
from app.ai.segmentation_net import create_segmentation_net, SegmentationNet3D
from app.ai.anomaly_detector import create_anomaly_detector, AnomalyDetector3D
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("export_onnx")


# ---------------------------------------------------------------------------
# Inference-mode wrapper: fixes the reparameterize trick for VAE
# ---------------------------------------------------------------------------
class AnomalyDetectorInference(nn.Module):
    """
    Wraps AnomalyDetector3D for deterministic ONNX export.

    At inference we want z = mu (no random sampling).
    Returns only the outputs needed at runtime:
      - anomaly_map  : (1, 1, D, H, W)
      - anomaly_score: (1,)
      - reconstruction: (1, 1, D, H, W)
    """

    def __init__(self, detector: AnomalyDetector3D):
        super().__init__()
        self.detector = detector

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.detector.encoder(x)
        # Deterministic: use mu directly (no eps sampling)
        z = mu
        reconstruction = self.detector.decoder(z)
        recon_error = (x - reconstruction) ** 2
        anomaly_map = recon_error
        density_score = -self.detector.density(mu)
        recon_score = recon_error.flatten(1).mean(dim=1)
        anomaly_score = 0.7 * recon_score + 0.3 * torch.sigmoid(density_score)
        return anomaly_map, anomaly_score, reconstruction


class SpatialTransformerInference(nn.Module):
    """
    Wraps SpatialTransformer3D to return a flat tuple instead of tensor
    so ONNX gets named outputs: (all_features,) or (cls_features, spatial_features).
    We export extract_features() which is what the service layer actually calls.
    """

    def __init__(self, transformer: SpatialTransformer3D):
        super().__init__()
        self.transformer = transformer

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls_features, spatial_features = self.transformer.extract_features(x)
        return cls_features, spatial_features


class SegmentationNetInference(nn.Module):
    """
    Wraps SegmentationNet3D — returns only the primary segmentation logits.
    Deep supervision outputs are training-only and not exported.
    transformer_features is passed as None → no transformer injection path.
    """

    def __init__(self, seg_net: SegmentationNet3D):
        super().__init__()
        self.seg_net = seg_net

    def forward(self, x: torch.Tensor, spatial_features: torch.Tensor) -> torch.Tensor:
        output = self.seg_net(x, transformer_features=spatial_features)
        return output["segmentation"]


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _make_dynamic_axes(input_names: list[str], output_names: list[str]) -> dict:
    """Batch dimension is dynamic for all inputs and outputs."""
    axes = {}
    for name in input_names + output_names:
        axes[name] = {0: "batch"}
    return axes


def export_spatial_transformer(
    model_path: str | None,
    output_path: Path,
    volume_size: int,
    opset: int,
    fp16: bool,
) -> None:
    log.info("─── Exporting SpatialTransformer3D ───")
    base = create_spatial_transformer(
        embed_dim=settings.EMBED_DIM,
        num_heads=settings.NUM_HEADS,
        num_layers=settings.NUM_LAYERS,
        pretrained_path=model_path,
    ).eval()

    wrapper = SpatialTransformerInference(base).eval()

    dummy_input = torch.zeros(1, 1, volume_size, volume_size, volume_size)

    input_names = ["volume"]
    output_names = ["cls_features", "spatial_features"]

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(output_path),
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=_make_dynamic_axes(input_names, output_names),
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

    log.info(f"  Saved → {output_path}")
    _verify_onnx(output_path)
    if fp16:
        _convert_fp16(output_path)


def export_segmentation_net(
    model_path: str | None,
    output_path: Path,
    volume_size: int,
    embed_dim: int,
    opset: int,
    fp16: bool,
) -> None:
    log.info("─── Exporting SegmentationNet3D ───")
    base = create_segmentation_net(
        num_classes=6,
        transformer_dim=embed_dim,
        pretrained_path=model_path,
    ).eval()

    wrapper = SegmentationNetInference(base).eval()

    # Compute spatial feature size: PatchEmbedding3D uses 3 conv with stride 2
    # so spatial dims are reduced by 2^3 = 8
    feat_spatial = volume_size // 8
    n_patches = feat_spatial ** 3

    dummy_volume = torch.zeros(1, 1, volume_size, volume_size, volume_size)
    dummy_spatial_feat = torch.zeros(1, n_patches, embed_dim)

    input_names = ["volume", "spatial_features"]
    output_names = ["segmentation"]

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_volume, dummy_spatial_feat),
            str(output_path),
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=_make_dynamic_axes(input_names, output_names),
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

    log.info(f"  Saved → {output_path}")
    _verify_onnx(output_path)
    if fp16:
        _convert_fp16(output_path)


def export_anomaly_detector(
    model_path: str | None,
    output_path: Path,
    volume_size: int,
    embed_dim: int,
    opset: int,
    fp16: bool,
) -> None:
    log.info("─── Exporting AnomalyDetector3D (inference mode) ───")
    base = create_anomaly_detector(
        latent_dim=256,
        input_size=volume_size,
        transformer_dim=embed_dim,
        pretrained_path=model_path,
    ).eval()

    wrapper = AnomalyDetectorInference(base).eval()

    dummy_input = torch.zeros(1, 1, volume_size, volume_size, volume_size)

    input_names = ["volume"]
    output_names = ["anomaly_map", "anomaly_score", "reconstruction"]

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(output_path),
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=_make_dynamic_axes(input_names, output_names),
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

    log.info(f"  Saved → {output_path}")
    _verify_onnx(output_path)
    if fp16:
        _convert_fp16(output_path)


# ---------------------------------------------------------------------------
# Verification and FP16 helpers
# ---------------------------------------------------------------------------

def _verify_onnx(path: Path) -> None:
    """Run onnx.checker — raises if graph is malformed."""
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    size_mb = path.stat().st_size / (1024 ** 2)
    log.info(f"  ✓ ONNX check passed  |  size: {size_mb:.1f} MB")


def _convert_fp16(fp32_path: Path) -> None:
    """Convert FP32 ONNX model to FP16 for ~50% size reduction."""
    try:
        from onnxconverter_common import float16  # pip install onnxconverter-common
        fp16_path = fp32_path.with_suffix(".fp16.onnx")
        model = onnx.load(str(fp32_path))
        model_fp16 = float16.convert_float_to_float16(
            model,
            keep_io_types=True,      # keep float32 I/O for compatibility
            disable_shape_infer=False,
        )
        onnx.save(model_fp16, str(fp16_path))
        size_mb = fp16_path.stat().st_size / (1024 ** 2)
        log.info(f"  ✓ FP16 saved → {fp16_path}  |  size: {size_mb:.1f} MB")
    except ImportError:
        log.warning("  onnxconverter-common not installed — skipping FP16 conversion.")
        log.warning("  Install with: pip install onnxconverter-common")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export MedSpatial AI models to ONNX")
    p.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="Directory containing .pth/.pt model weights",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "desktop" / "onnx_models",
        help="Output directory for .onnx files",
    )
    p.add_argument(
        "--volume-size",
        type=int,
        default=128,
        help="Spatial size of the input volume cube (default: 128)",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="Also produce FP16 variants for smaller file size",
    )
    p.add_argument(
        "--only",
        choices=["spatial_transformer", "segmentation_net", "anomaly_detector"],
        default=None,
        help="Export only one specific model",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Model weights dir : {args.model_dir}")
    log.info(f"ONNX output dir   : {args.output_dir}")
    log.info(f"Volume size       : {args.volume_size}³")
    log.info(f"ONNX opset        : {args.opset}")
    log.info(f"FP16 conversion   : {args.fp16}")
    log.info("")

    # Resolve optional weight paths
    def _weight(filename: str) -> str | None:
        p = args.model_dir / filename
        if p.exists():
            log.info(f"  Loading weights: {p}")
            return str(p)
        log.warning(f"  No weights found at {p} — exporting with random weights")
        return None

    run_all = args.only is None

    if run_all or args.only == "spatial_transformer":
        export_spatial_transformer(
            model_path=_weight("spatial_transformer.pth"),
            output_path=args.output_dir / "spatial_transformer.onnx",
            volume_size=args.volume_size,
            opset=args.opset,
            fp16=args.fp16,
        )

    if run_all or args.only == "segmentation_net":
        export_segmentation_net(
            model_path=_weight("segmentation_net.pth"),
            output_path=args.output_dir / "segmentation_net.onnx",
            volume_size=args.volume_size,
            embed_dim=settings.EMBED_DIM,
            opset=args.opset,
            fp16=args.fp16,
        )

    if run_all or args.only == "anomaly_detector":
        export_anomaly_detector(
            model_path=_weight("anomaly_detector.pth"),
            output_path=args.output_dir / "anomaly_detector.onnx",
            volume_size=args.volume_size,
            embed_dim=settings.EMBED_DIM,
            opset=args.opset,
            fp16=args.fp16,
        )

    log.info("")
    log.info("═══════════════════════════════════════")
    log.info("  All models exported successfully.")
    log.info(f"  Output directory: {args.output_dir}")
    log.info("  Next step: run validate_onnx.py")
    log.info("═══════════════════════════════════════")


if __name__ == "__main__":
    main()
