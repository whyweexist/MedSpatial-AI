"""
MedSpatial AI — ONNX INT8 Static Quantization
==============================================
Applies post-training static INT8 quantization to the exported ONNX models.
INT8 reduces model size by ~75% and speeds up CPU inference by 2–4x on
modern CPUs with AVX2/AVX-512 support.

Quantization strategy:
  - Uses ONNX Runtime's QuantizationMode.QLinearOps (recommended for CPU)
  - Calibration with synthetic representative data (or real data if provided)
  - Per-channel quantization for Conv3d weights (better accuracy)
  - Asymmetric activation quantization (uint8, standard for CPU)
  - Accuracy comparison: quantized vs FP32 ONNX outputs

Usage:
  python desktop/scripts/quantize_onnx.py \
      --onnx-dir  desktop/onnx_models/ \
      --output-dir desktop/onnx_models/quantized/ \
      --volume-size 128 \
      --n-calib 8

  # Quantize single model
  python desktop/scripts/quantize_onnx.py --only anomaly_detector

  # Use real calibration data (directory of .npy volume files)
  python desktop/scripts/quantize_onnx.py --calib-data path/to/volumes/

Requirements (export machine only):
  pip install onnxruntime onnx numpy
  # For full quantization pipeline:
  pip install onnxruntime-extensions

Notes:
  - 3D convolution quantization support in ORT requires opset >= 13.
  - If quantized accuracy drops > 5% vs FP32, fallback to FP16 instead.
  - LayerNorm and Softmax ops are intentionally kept in FP32 for stability.
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    QuantFormat,
    QuantType,
    quantize_static,
    CalibrationDataReader,
    CalibrationMethod,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

REPO_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("quantize_onnx")


# ---------------------------------------------------------------------------
# Calibration data readers
# ---------------------------------------------------------------------------

class VolumeCalibrationReader(CalibrationDataReader):
    """
    Feeds synthetic (or real) 3D volumes to the ONNX model for calibration.
    Calibration determines the min/max range of each activation tensor,
    which is then used to compute INT8 quantization scales.
    """

    def __init__(
        self,
        input_names: list[str],
        input_shapes: dict[str, tuple],
        n_samples: int = 8,
        calib_data_dir: Path | None = None,
    ):
        self.input_names = input_names
        self.input_shapes = input_shapes
        self.n_samples = n_samples
        self.calib_data_dir = calib_data_dir
        self._samples = self._build_samples()
        self._index = 0

    def _build_samples(self) -> list[dict[str, np.ndarray]]:
        samples = []

        if self.calib_data_dir and self.calib_data_dir.exists():
            # Load real .npy calibration files
            npy_files = list(self.calib_data_dir.glob("*.npy"))[: self.n_samples]
            for f in npy_files:
                vol = np.load(str(f)).astype(np.float32)
                # Normalize to [0, 1]
                vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
                sample = {}
                for name in self.input_names:
                    shape = self.input_shapes[name]
                    if name == "volume":
                        # Resize if needed
                        if vol.shape != shape[1:]:
                            from scipy.ndimage import zoom
                            factors = [s / v for s, v in zip(shape[1:], vol.shape)]
                            vol = zoom(vol, [1] + factors, order=1)
                        sample[name] = vol.reshape(shape).astype(np.float32)
                    else:
                        # For feature inputs, use zeros during calibration
                        sample[name] = np.zeros(shape, dtype=np.float32)
                samples.append(sample)
            log.info(f"  Loaded {len(samples)} real calibration samples from {self.calib_data_dir}")

        # Pad with synthetic data up to n_samples
        np.random.seed(0)
        while len(samples) < self.n_samples:
            sample = {}
            for name, shape in self.input_shapes.items():
                if name == "volume":
                    # Simulate normalized medical volume: mostly zeros (air), few bright spots
                    vol = np.zeros(shape, dtype=np.float32)
                    # Add random anatomy-like blobs
                    rng = np.random.RandomState(len(samples))
                    for _ in range(rng.randint(3, 12)):
                        center = [rng.randint(10, s - 10) for s in shape[1:]]
                        radius = rng.randint(5, 20)
                        z, y, x = np.ogrid[
                            : shape[1], : shape[2], : shape[3]
                        ]
                        mask = (
                            (z - center[0]) ** 2
                            + (y - center[1]) ** 2
                            + (x - center[2]) ** 2
                        ) <= radius ** 2
                        vol[0][mask] = rng.uniform(0.2, 1.0)
                    sample[name] = vol
                else:
                    sample[name] = np.random.randn(*shape).astype(np.float32) * 0.1
            samples.append(sample)

        log.info(f"  Total calibration samples: {len(samples)}")
        return samples

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self._samples):
            return None
        sample = self._samples[self._index]
        self._index += 1
        return sample


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def _get_ops_to_exclude() -> list[str]:
    """
    Ops to keep in FP32 for numerical stability.
    LayerNorm and Softmax degrade significantly when quantized.
    """
    return ["LayerNormalization", "Softmax", "Sigmoid", "Gather"]


def quantize_model(
    fp32_path: Path,
    output_path: Path,
    calib_reader: CalibrationDataReader,
    per_channel: bool = True,
) -> None:
    """
    Run ONNX Runtime static quantization on a single model.

    Steps:
      1. Pre-process: shape inference + model optimization
      2. Static quantization with MinMax calibration
      3. Save quantized model
    """
    log.info(f"  Input  : {fp32_path}")
    log.info(f"  Output : {output_path}")

    # Step 1: Pre-process the model (required before static quantization)
    with tempfile.TemporaryDirectory() as tmpdir:
        preprocessed_path = Path(tmpdir) / "preprocessed.onnx"
        log.info("  Running quant_pre_process (shape inference + optimization)...")
        quant_pre_process(
            input_model_path=str(fp32_path),
            output_model_path=str(preprocessed_path),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
            auto_merge=True,
            verbose=0,
        )

        # Step 2: Static quantization
        log.info("  Running static quantization (INT8, MinMax calibration)...")
        quantize_static(
            model_input=str(preprocessed_path),
            model_output=str(output_path),
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QOperator,   # fused ops, fastest on CPU
            activation_type=QuantType.QUInt8,      # asymmetric, good for ReLU/GELU
            weight_type=QuantType.QInt8,           # symmetric for weights
            per_channel=per_channel,               # per-output-channel for Conv
            reduce_range=True,                     # avoid overflow on AVX2
            calibrate_method=CalibrationMethod.MinMax,
            nodes_to_exclude=_get_ops_to_exclude(),
            extra_options={
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
                "EnableSubgraph": True,
            },
        )

    size_fp32 = fp32_path.stat().st_size / (1024 ** 2)
    size_int8 = output_path.stat().st_size / (1024 ** 2)
    ratio = (1 - size_int8 / size_fp32) * 100
    log.info(f"  FP32 size : {size_fp32:.1f} MB")
    log.info(f"  INT8 size : {size_int8:.1f} MB  (reduction: {ratio:.0f}%)")


# ---------------------------------------------------------------------------
# Accuracy comparison
# ---------------------------------------------------------------------------

def _run_ort(session: ort.InferenceSession, inputs: dict) -> list[np.ndarray]:
    return session.run(None, inputs)


def compare_accuracy(
    fp32_path: Path,
    int8_path: Path,
    volume_size: int,
    embed_dim: int,
    model_key: str,
    n_samples: int = 10,
) -> dict:
    """
    Compare FP32 vs INT8 ONNX outputs on random inputs.
    Returns accuracy metrics dict.
    """
    log.info(f"  Comparing accuracy: FP32 vs INT8  ({model_key})")

    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    input_meta = sess_fp32.get_inputs()

    max_diffs = []
    rmses = []
    np.random.seed(99)

    for i in range(n_samples):
        # Build sample inputs matching the model's input schema
        feed = {}
        for meta in input_meta:
            shape = [d if isinstance(d, int) else 1 for d in meta.shape]
            feed[meta.name] = np.random.randn(*shape).astype(np.float32)
            # Clip to [0, 1] for volume inputs
            if meta.name == "volume":
                feed[meta.name] = np.clip(feed[meta.name], 0.0, 1.0)

        fp32_outs = _run_ort(sess_fp32, feed)
        int8_outs = _run_ort(sess_int8, feed)

        for fp32_o, int8_o in zip(fp32_outs, int8_outs):
            diff = np.abs(fp32_o.astype(np.float32) - int8_o.astype(np.float32))
            max_diffs.append(float(diff.max()))
            rmses.append(float(np.sqrt((diff ** 2).mean())))

    result = {
        "model": model_key,
        "n_samples": n_samples,
        "mean_max_diff": float(np.mean(max_diffs)),
        "mean_rmse": float(np.mean(rmses)),
        "max_max_diff": float(np.max(max_diffs)),
        "pass": float(np.mean(max_diffs)) < 0.1,  # <10% error threshold
    }

    log.info(f"    Mean MaxAbsDiff : {result['mean_max_diff']:.4f}")
    log.info(f"    Mean RMSE       : {result['mean_rmse']:.4f}")
    log.info(f"    Max  MaxAbsDiff : {result['max_max_diff']:.4f}")
    log.info(f"    Accuracy check  : {'✓ PASS' if result['pass'] else '⚠ WARN (consider FP16 instead)'}")

    return result


# ---------------------------------------------------------------------------
# Per-model quantization
# ---------------------------------------------------------------------------

def quantize_spatial_transformer(
    onnx_dir: Path,
    output_dir: Path,
    volume_size: int,
    embed_dim: int,
    n_calib: int,
    calib_data_dir: Path | None,
) -> None:
    log.info("\n─── Quantizing SpatialTransformer3D ───")
    fp32 = onnx_dir / "spatial_transformer.onnx"
    out  = output_dir / "spatial_transformer_int8.onnx"

    reader = VolumeCalibrationReader(
        input_names=["volume"],
        input_shapes={"volume": (1, 1, volume_size, volume_size, volume_size)},
        n_samples=n_calib,
        calib_data_dir=calib_data_dir,
    )
    quantize_model(fp32, out, reader)
    compare_accuracy(fp32, out, volume_size, embed_dim, "spatial_transformer")


def quantize_segmentation_net(
    onnx_dir: Path,
    output_dir: Path,
    volume_size: int,
    embed_dim: int,
    n_calib: int,
    calib_data_dir: Path | None,
) -> None:
    log.info("\n─── Quantizing SegmentationNet3D ───")
    fp32 = onnx_dir / "segmentation_net.onnx"
    out  = output_dir / "segmentation_net_int8.onnx"

    feat_spatial = volume_size // 8
    n_patches = feat_spatial ** 3

    reader = VolumeCalibrationReader(
        input_names=["volume", "spatial_features"],
        input_shapes={
            "volume": (1, 1, volume_size, volume_size, volume_size),
            "spatial_features": (1, n_patches, embed_dim),
        },
        n_samples=n_calib,
        calib_data_dir=calib_data_dir,
    )
    quantize_model(fp32, out, reader)
    compare_accuracy(fp32, out, volume_size, embed_dim, "segmentation_net")


def quantize_anomaly_detector(
    onnx_dir: Path,
    output_dir: Path,
    volume_size: int,
    embed_dim: int,
    n_calib: int,
    calib_data_dir: Path | None,
) -> None:
    log.info("\n─── Quantizing AnomalyDetector3D ───")
    fp32 = onnx_dir / "anomaly_detector.onnx"
    out  = output_dir / "anomaly_detector_int8.onnx"

    reader = VolumeCalibrationReader(
        input_names=["volume"],
        input_shapes={"volume": (1, 1, volume_size, volume_size, volume_size)},
        n_samples=n_calib,
        calib_data_dir=calib_data_dir,
    )
    quantize_model(fp32, out, reader)
    compare_accuracy(fp32, out, volume_size, embed_dim, "anomaly_detector")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantize MedSpatial AI ONNX models to INT8")
    p.add_argument("--onnx-dir",    type=Path, default=REPO_ROOT / "desktop" / "onnx_models")
    p.add_argument("--output-dir",  type=Path, default=REPO_ROOT / "desktop" / "onnx_models" / "quantized")
    p.add_argument("--volume-size", type=int,  default=128)
    p.add_argument("--embed-dim",   type=int,  default=512)
    p.add_argument("--n-calib",     type=int,  default=8,
                   help="Number of calibration samples (more = better but slower)")
    p.add_argument("--calib-data",  type=Path, default=None,
                   help="Optional directory of .npy calibration volumes")
    p.add_argument(
        "--only",
        choices=["spatial_transformer", "segmentation_net", "anomaly_detector"],
        default=None,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"ONNX source dir   : {args.onnx_dir}")
    log.info(f"Output dir        : {args.output_dir}")
    log.info(f"Calibration samples: {args.n_calib}")

    run_all = args.only is None

    if run_all or args.only == "spatial_transformer":
        quantize_spatial_transformer(
            args.onnx_dir, args.output_dir,
            args.volume_size, args.embed_dim,
            args.n_calib, args.calib_data,
        )

    if run_all or args.only == "segmentation_net":
        quantize_segmentation_net(
            args.onnx_dir, args.output_dir,
            args.volume_size, args.embed_dim,
            args.n_calib, args.calib_data,
        )

    if run_all or args.only == "anomaly_detector":
        quantize_anomaly_detector(
            args.onnx_dir, args.output_dir,
            args.volume_size, args.embed_dim,
            args.n_calib, args.calib_data,
        )

    log.info("\n═══════════════════════════════════════════════")
    log.info("  Quantization complete.")
    log.info(f"  Quantized models → {args.output_dir}")
    log.info("  ⚠ If accuracy check shows WARN, use FP32 or FP16 ONNX instead.")
    log.info("═══════════════════════════════════════════════")


if __name__ == "__main__":
    main()
