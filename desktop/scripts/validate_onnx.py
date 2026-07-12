"""
MedSpatial AI — ONNX Validation & PyTorch Comparison
=====================================================
Validates exported ONNX models against their PyTorch equivalents.

For each model this script:
  1. Loads the ONNX model and runs a forward pass with ONNX Runtime
  2. Loads the original PyTorch model and runs the same forward pass
  3. Computes per-output max absolute difference and RMSE
  4. Reports pass/fail with configurable tolerance
  5. Measures and compares inference latency

Usage:
  python desktop/scripts/validate_onnx.py \
      --onnx-dir  desktop/onnx_models/ \
      --model-dir models/ \
      --volume-size 128

  # Validate single model
  python desktop/scripts/validate_onnx.py --only anomaly_detector

  # Stricter tolerance
  python desktop/scripts/validate_onnx.py --atol 1e-4

Requirements:
  pip install onnxruntime torch numpy tabulate
"""

import argparse
import sys
import time
import logging
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.ai.spatial_transformer import create_spatial_transformer, SpatialTransformer3D
from app.ai.segmentation_net import create_segmentation_net
from app.ai.anomaly_detector import create_anomaly_detector
from app.config import settings

# Re-import the inference wrappers from the export script
sys.path.insert(0, str(REPO_ROOT / "desktop" / "scripts"))
from export_onnx import (
    SpatialTransformerInference,
    SegmentationNetInference,
    AnomalyDetectorInference,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("validate_onnx")

# ─── Default tolerance ────────────────────────────────────────────────────────
# FP32 ONNX vs PyTorch differences arise from op-level fused kernels and
# constant folding. 1e-3 absolute tolerance is industry-standard for this.
DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3


# ---------------------------------------------------------------------------
# ONNX Runtime session factory
# ---------------------------------------------------------------------------
def _make_ort_session(onnx_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)  # half cores
    opts.inter_op_num_threads = 1
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


import os  # noqa: E402 (after usage in _make_ort_session)


# ---------------------------------------------------------------------------
# Generic comparison helper
# ---------------------------------------------------------------------------
def _compare(
    model_name: str,
    ort_outputs: list[np.ndarray],
    torch_outputs: list[torch.Tensor],
    output_names: list[str],
    atol: float,
    rtol: float,
) -> bool:
    """Compare ORT and PyTorch outputs element-wise. Returns True if all pass."""
    all_passed = True
    rows = []

    for name, ort_out, torch_out in zip(output_names, ort_outputs, torch_outputs):
        np_torch = torch_out.detach().cpu().numpy()
        diff = np.abs(ort_out - np_torch)
        max_diff = float(diff.max())
        rmse = float(np.sqrt((diff ** 2).mean()))
        close = np.allclose(ort_out, np_torch, atol=atol, rtol=rtol)
        status = "✓ PASS" if close else "✗ FAIL"
        if not close:
            all_passed = False
        rows.append((name, f"{max_diff:.2e}", f"{rmse:.2e}", status))

    # Print table
    header = f"\n  {'Output':<25}  {'MaxAbsDiff':>12}  {'RMSE':>10}  {'Status':>8}"
    sep = "  " + "─" * 62
    log.info(f"\n{'═' * 66}")
    log.info(f"  {model_name}")
    log.info(header)
    log.info(sep)
    for row in rows:
        log.info(f"  {row[0]:<25}  {row[1]:>12}  {row[2]:>10}  {row[3]:>8}")
    log.info(sep)
    log.info(f"  Result: {'ALL PASSED' if all_passed else 'FAILURES DETECTED'}")

    return all_passed


# ---------------------------------------------------------------------------
# Per-model validation
# ---------------------------------------------------------------------------
def validate_spatial_transformer(
    onnx_path: Path,
    model_path: str | None,
    volume_size: int,
    atol: float,
    rtol: float,
    n_warmup: int = 2,
    n_bench: int = 5,
) -> bool:
    log.info("\n──────────────────────────────────────────────────────────────")
    log.info("  Validating: SpatialTransformer3D")
    log.info(f"  ONNX: {onnx_path}")

    session = _make_ort_session(onnx_path)

    # PyTorch reference
    base = create_spatial_transformer(
        embed_dim=settings.EMBED_DIM,
        num_heads=settings.NUM_HEADS,
        num_layers=settings.NUM_LAYERS,
        pretrained_path=model_path,
    ).eval()
    wrapper = SpatialTransformerInference(base).eval()

    # Fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    dummy = torch.randn(1, 1, volume_size, volume_size, volume_size)
    dummy_np = dummy.numpy()

    # PyTorch forward
    with torch.no_grad():
        pt_cls, pt_spatial = wrapper(dummy)

    # ORT forward
    ort_cls, ort_spatial = session.run(None, {"volume": dummy_np})

    passed = _compare(
        "SpatialTransformer3D",
        [ort_cls, ort_spatial],
        [pt_cls, pt_spatial],
        ["cls_features", "spatial_features"],
        atol, rtol,
    )

    # Latency benchmark
    _benchmark(session, {"volume": dummy_np}, "ORT", n_warmup, n_bench)
    _benchmark_torch(wrapper, (dummy,), "PyTorch", n_warmup, n_bench)
    return passed


def validate_segmentation_net(
    onnx_path: Path,
    model_path: str | None,
    volume_size: int,
    embed_dim: int,
    atol: float,
    rtol: float,
    n_warmup: int = 2,
    n_bench: int = 5,
) -> bool:
    log.info("\n──────────────────────────────────────────────────────────────")
    log.info("  Validating: SegmentationNet3D")
    log.info(f"  ONNX: {onnx_path}")

    session = _make_ort_session(onnx_path)

    base = create_segmentation_net(
        num_classes=6,
        transformer_dim=embed_dim,
        pretrained_path=model_path,
    ).eval()
    wrapper = SegmentationNetInference(base).eval()

    feat_spatial = volume_size // 8
    n_patches = feat_spatial ** 3

    torch.manual_seed(42)
    dummy_vol = torch.randn(1, 1, volume_size, volume_size, volume_size)
    dummy_feat = torch.randn(1, n_patches, embed_dim)
    dummy_vol_np = dummy_vol.numpy()
    dummy_feat_np = dummy_feat.numpy()

    with torch.no_grad():
        pt_seg = wrapper(dummy_vol, dummy_feat)

    ort_seg, = session.run(None, {
        "volume": dummy_vol_np,
        "spatial_features": dummy_feat_np,
    })

    passed = _compare(
        "SegmentationNet3D",
        [ort_seg],
        [pt_seg],
        ["segmentation"],
        atol, rtol,
    )

    _benchmark(
        session,
        {"volume": dummy_vol_np, "spatial_features": dummy_feat_np},
        "ORT", n_warmup, n_bench,
    )
    _benchmark_torch(wrapper, (dummy_vol, dummy_feat), "PyTorch", n_warmup, n_bench)
    return passed


def validate_anomaly_detector(
    onnx_path: Path,
    model_path: str | None,
    volume_size: int,
    embed_dim: int,
    atol: float,
    rtol: float,
    n_warmup: int = 2,
    n_bench: int = 5,
) -> bool:
    log.info("\n──────────────────────────────────────────────────────────────")
    log.info("  Validating: AnomalyDetector3D (inference mode)")
    log.info(f"  ONNX: {onnx_path}")

    session = _make_ort_session(onnx_path)

    base = create_anomaly_detector(
        latent_dim=256,
        input_size=volume_size,
        transformer_dim=embed_dim,
        pretrained_path=model_path,
    ).eval()
    wrapper = AnomalyDetectorInference(base).eval()

    torch.manual_seed(42)
    dummy = torch.randn(1, 1, volume_size, volume_size, volume_size)
    dummy_np = dummy.numpy()

    with torch.no_grad():
        pt_map, pt_score, pt_recon = wrapper(dummy)

    ort_map, ort_score, ort_recon = session.run(None, {"volume": dummy_np})

    passed = _compare(
        "AnomalyDetector3D",
        [ort_map, ort_score, ort_recon],
        [pt_map, pt_score, pt_recon],
        ["anomaly_map", "anomaly_score", "reconstruction"],
        atol, rtol,
    )

    _benchmark(session, {"volume": dummy_np}, "ORT", n_warmup, n_bench)
    _benchmark_torch(wrapper, (dummy,), "PyTorch", n_warmup, n_bench)
    return passed


# ---------------------------------------------------------------------------
# Latency helpers
# ---------------------------------------------------------------------------
def _benchmark(
    session: ort.InferenceSession,
    inputs: dict,
    label: str,
    n_warmup: int,
    n_runs: int,
) -> None:
    for _ in range(n_warmup):
        session.run(None, inputs)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, inputs)
        times.append((time.perf_counter() - t0) * 1000)
    log.info(
        f"  Latency [{label}]: "
        f"mean={np.mean(times):.0f}ms  "
        f"min={np.min(times):.0f}ms  "
        f"max={np.max(times):.0f}ms  "
        f"(n={n_runs})"
    )


def _benchmark_torch(
    model: torch.nn.Module,
    inputs: tuple,
    label: str,
    n_warmup: int,
    n_runs: int,
) -> None:
    with torch.no_grad():
        for _ in range(n_warmup):
            model(*inputs)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(*inputs)
            times.append((time.perf_counter() - t0) * 1000)
    log.info(
        f"  Latency [{label}]:  "
        f"mean={np.mean(times):.0f}ms  "
        f"min={np.min(times):.0f}ms  "
        f"max={np.max(times):.0f}ms  "
        f"(n={n_runs})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate MedSpatial AI ONNX models")
    p.add_argument("--onnx-dir",  type=Path, default=REPO_ROOT / "desktop" / "onnx_models")
    p.add_argument("--model-dir", type=Path, default=REPO_ROOT / "models")
    p.add_argument("--volume-size", type=int, default=128)
    p.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    p.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    p.add_argument(
        "--only",
        choices=["spatial_transformer", "segmentation_net", "anomaly_detector"],
        default=None,
    )
    p.add_argument("--n-bench", type=int, default=5, help="Latency benchmark iterations")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    def _w(name: str) -> str | None:
        p = args.model_dir / f"{name}.pth"
        return str(p) if p.exists() else None

    results: dict[str, bool] = {}
    run_all = args.only is None

    if run_all or args.only == "spatial_transformer":
        results["spatial_transformer"] = validate_spatial_transformer(
            onnx_path=args.onnx_dir / "spatial_transformer.onnx",
            model_path=_w("spatial_transformer"),
            volume_size=args.volume_size,
            atol=args.atol,
            rtol=args.rtol,
            n_bench=args.n_bench,
        )

    if run_all or args.only == "segmentation_net":
        results["segmentation_net"] = validate_segmentation_net(
            onnx_path=args.onnx_dir / "segmentation_net.onnx",
            model_path=_w("segmentation_net"),
            volume_size=args.volume_size,
            embed_dim=settings.EMBED_DIM,
            atol=args.atol,
            rtol=args.rtol,
            n_bench=args.n_bench,
        )

    if run_all or args.only == "anomaly_detector":
        results["anomaly_detector"] = validate_anomaly_detector(
            onnx_path=args.onnx_dir / "anomaly_detector.onnx",
            model_path=_w("anomaly_detector"),
            volume_size=args.volume_size,
            embed_dim=settings.EMBED_DIM,
            atol=args.atol,
            rtol=args.rtol,
            n_bench=args.n_bench,
        )

    # Final summary
    log.info("\n\n╔══════════════════════════════════════╗")
    log.info("║       VALIDATION SUMMARY             ║")
    log.info("╠══════════════════════════════════════╣")
    all_ok = True
    for model, passed in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        log.info(f"║  {model:<30}  {status}  ║")
        if not passed:
            all_ok = False
    log.info("╚══════════════════════════════════════╝")

    if not all_ok:
        log.error("One or more models failed validation. Check tolerance or re-export.")
        sys.exit(1)

    log.info("\nAll models validated. Safe to proceed with quantization or packaging.")


if __name__ == "__main__":
    main()
