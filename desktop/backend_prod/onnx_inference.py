"""
MedSpatial AI — ONNX Runtime Inference Engine
==============================================
Drop-in replacement for the PyTorch AnomalyService.
Loads all three ONNX models once at startup and exposes the exact same
public API surface that the rest of the backend (services, API routes)
already calls — so no other file needs to change.

Public API (mirrors AnomalyService):
    engine = OnnxInferenceEngine(onnx_dir, volume_size)
    engine.detect_anomalies(volume_np)   -> dict
    engine.segment_organs(volume_np)     -> dict
    engine.dissect_layers(volume_np)     -> dict   (unchanged, no PyTorch)

Design goals:
    - Zero PyTorch import at runtime.
    - Thread-safe: each InferenceSession is created with intra_op threading
      sized to half available cores.  Multiple concurrent requests share
      the same session objects safely (ORT is thread-safe per session).
    - Lazy model loading with an explicit warm_up() call from the lifespan hook.
    - Identical numerical output to the PyTorch path (validated by
      validate_onnx.py within 1e-3 absolute tolerance).
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from scipy import ndimage

log = logging.getLogger("onnx_inference")

# ---------------------------------------------------------------------------
# Session factory — shared, thread-safe configuration
# ---------------------------------------------------------------------------

def _build_session(model_path: Path) -> ort.InferenceSession:
    """
    Create an optimised CPU InferenceSession.

    Graph optimisation level ORT_ENABLE_ALL fuses ops (e.g.
    Conv+BatchNorm+GELU → single kernel) which reduces both latency and
    peak memory compared to the default ORT_ENABLE_BASIC.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}\n"
            "Run desktop/scripts/export_onnx.py first."
        )

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Use half the logical CPUs for intra-op parallelism; leave the rest for
    # the FastAPI event loop and OS overhead.
    n_cores = max(1, (os.cpu_count() or 4) // 2)
    opts.intra_op_num_threads = n_cores
    opts.inter_op_num_threads = 1        # sequential op dispatch is faster for single-batch
    opts.enable_mem_pattern = True       # reuse activation buffers across runs
    opts.enable_cpu_mem_arena = True     # pre-allocate CPU arena to avoid malloc pressure
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    size_mb = model_path.stat().st_size / (1024 ** 2)
    log.info(f"  Loaded {model_path.name}  ({size_mb:.1f} MB)  threads={n_cores}")
    return session


# ---------------------------------------------------------------------------
# Volume pre-processing  (mirrors VolumeProcessor + AnomalyService._prepare_volume)
# ---------------------------------------------------------------------------

_HU_MIN = -1000.0
_HU_MAX = 3000.0


def _preprocess_volume(volume: np.ndarray, target_size: int) -> np.ndarray:
    """
    Clip HU, normalise to [0, 1], resize to (target_size)³, add batch+channel dims.

    Returns: float32 array of shape (1, 1, D, H, W)
    """
    vol = np.clip(volume.astype(np.float32), _HU_MIN, _HU_MAX)
    vol = (vol - _HU_MIN) / (_HU_MAX - _HU_MIN)

    if vol.shape != (target_size, target_size, target_size):
        zoom_factors = [target_size / s for s in vol.shape]
        vol = ndimage.zoom(vol, zoom_factors, order=1)

    return vol[np.newaxis, np.newaxis, :, :, :].astype(np.float32)   # (1,1,D,H,W)


def _resize_back(data: np.ndarray, target_shape: tuple, order: int = 1) -> np.ndarray:
    """Zoom a model output volume back to the original scan dimensions."""
    zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]
    return ndimage.zoom(data, zoom_factors, order=order)


# ---------------------------------------------------------------------------
# Main inference engine
# ---------------------------------------------------------------------------

class OnnxInferenceEngine:
    """
    Singleton-style inference engine backed by ONNX Runtime.

    Usage:
        engine = OnnxInferenceEngine(onnx_dir=Path("desktop/onnx_models"), volume_size=128)
        engine.warm_up()          # call once from FastAPI lifespan
        result = engine.detect_anomalies(volume_np)
    """

    def __init__(self, onnx_dir: Path, volume_size: int = 128):
        self._onnx_dir = Path(onnx_dir)
        self._volume_size = volume_size
        self._lock = threading.Lock()

        # Sessions — populated lazily by warm_up() or on first call
        self._sess_transformer:  Optional[ort.InferenceSession] = None
        self._sess_segmentation: Optional[ort.InferenceSession] = None
        self._sess_anomaly:      Optional[ort.InferenceSession] = None
        self._loaded = False

        # Layer dissector — pure numpy, no PyTorch
        self._layer_dissector = None

    # ------------------------------------------------------------------
    # Session loading
    # ------------------------------------------------------------------

    def _load_sessions(self) -> None:
        """Load all three ONNX sessions. Idempotent and thread-safe."""
        with self._lock:
            if self._loaded:
                return
            log.info("Loading ONNX Runtime sessions...")

            # Prefer INT8 quantized models if available, fall back to FP32
            def _pick(stem: str) -> Path:
                q = self._onnx_dir / "quantized" / f"{stem}_int8.onnx"
                if q.exists():
                    log.info(f"  Using quantized: {q.name}")
                    return q
                fp16 = self._onnx_dir / f"{stem}.fp16.onnx"
                if fp16.exists():
                    log.info(f"  Using FP16: {fp16.name}")
                    return fp16
                return self._onnx_dir / f"{stem}.onnx"

            self._sess_transformer  = _build_session(_pick("spatial_transformer"))
            self._sess_segmentation = _build_session(_pick("segmentation_net"))
            self._sess_anomaly      = _build_session(_pick("anomaly_detector"))
            self._loaded = True
            log.info("All ONNX sessions ready.")

    def warm_up(self) -> None:
        """
        Load sessions and run one forward pass to JIT-compile kernels.
        Call this from the FastAPI lifespan so the first user request is fast.
        """
        self._load_sessions()
        log.info("Running warm-up pass...")
        dummy = np.zeros(
            (1, 1, self._volume_size, self._volume_size, self._volume_size),
            dtype=np.float32,
        )
        try:
            self._run_transformer(dummy)
            log.info("  Warm-up complete.")
        except Exception as exc:
            log.warning(f"  Warm-up failed (non-fatal): {exc}")

    # ------------------------------------------------------------------
    # Internal runners
    # ------------------------------------------------------------------

    def _run_transformer(
        self, volume_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        SpatialTransformer3D forward pass.

        Args:
            volume_batch: (1, 1, D, H, W) float32

        Returns:
            cls_features:     (1, embed_dim)
            spatial_features: (1, N, embed_dim)
        """
        cls_feat, spatial_feat = self._sess_transformer.run(
            None, {"volume": volume_batch}
        )
        return cls_feat, spatial_feat

    def _run_segmentation(
        self,
        volume_batch: np.ndarray,
        spatial_features: np.ndarray,
    ) -> np.ndarray:
        """
        SegmentationNet3D forward pass.

        Returns:
            segmentation logits: (1, num_classes, D, H, W)
        """
        (seg_logits,) = self._sess_segmentation.run(
            None,
            {"volume": volume_batch, "spatial_features": spatial_features},
        )
        return seg_logits

    def _run_anomaly(
        self, volume_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        AnomalyDetector3D forward pass.

        Returns:
            anomaly_map:   (1, 1, D, H, W)
            anomaly_score: (1,)
            reconstruction:(1, 1, D, H, W)
        """
        anomaly_map, anomaly_score, reconstruction = self._sess_anomaly.run(
            None, {"volume": volume_batch}
        )
        return anomaly_map, anomaly_score, reconstruction

    # ------------------------------------------------------------------
    # Public API  (mirrors AnomalyService exactly)
    # ------------------------------------------------------------------

    def detect_anomalies(self, volume: np.ndarray) -> dict:
        """
        Run full anomaly detection pipeline on a 3-D numpy volume (HU values).

        Returns:
            heatmap:            np.ndarray (D, H, W) — per-voxel anomaly scores
            findings:           list[dict] — structured findings list
            overall_confidence: float      — scalar anomaly score ∈ [0, 1]
            summary:            str        — radiologist-readable summary
        """
        self._load_sessions()
        original_shape = volume.shape

        vol_batch = _preprocess_volume(volume, self._volume_size)

        # Stage 1: spatial transformer features
        cls_feat, _spatial_feat = self._run_transformer(vol_batch)

        # Stage 2: anomaly detection
        anomaly_map_batch, anomaly_score_batch, _recon = self._run_anomaly(vol_batch)

        # Squeeze model outputs
        anomaly_map = anomaly_map_batch[0, 0]                    # (D, H, W) model-space
        overall_score = float(anomaly_score_batch[0])

        # Resize heatmap to original scan dimensions
        heatmap = _resize_back(anomaly_map, original_shape, order=1)

        # Extract structured findings from heatmap
        findings = _extract_findings(heatmap, volume, overall_score)
        summary  = _generate_summary(findings, overall_score)

        return {
            "heatmap": heatmap,
            "findings": findings,
            "overall_confidence": overall_score,
            "summary": summary,
        }

    def segment_organs(self, volume: np.ndarray) -> dict:
        """
        Run organ segmentation. Returns integer label mask + per-class statistics.

        Returns:
            mask:        np.ndarray int (D, H, W) — class index per voxel
            class_names: list[str]
            statistics:  dict[str, dict]
        """
        self._load_sessions()
        original_shape = volume.shape

        vol_batch = _preprocess_volume(volume, self._volume_size)

        # Get spatial features for segmentation injection
        _cls_feat, spatial_feat = self._run_transformer(vol_batch)

        # Segmentation forward
        seg_logits = self._run_segmentation(vol_batch, spatial_feat)  # (1, C, D, H, W)

        # Argmax → class map in model space
        seg_mask_small = np.argmax(seg_logits[0], axis=0).astype(np.float32)  # (D, H, W)

        # Resize to original — nearest-neighbour to preserve label integers
        seg_mask_full = _resize_back(seg_mask_small, original_shape, order=0).astype(int)

        class_names = ["background", "bone", "soft_tissue", "air", "vessel", "anomaly"]
        statistics: dict[str, dict] = {}
        total_voxels = seg_mask_full.size
        for i, name in enumerate(class_names):
            count = int((seg_mask_full == i).sum())
            statistics[name] = {
                "voxel_count": count,
                "percentage": float(count / total_voxels * 100),
            }

        return {"mask": seg_mask_full, "class_names": class_names, "statistics": statistics}

    def dissect_layers(self, volume: np.ndarray, use_refinement: bool = True) -> dict:
        """
        Tissue layer decomposition.  Delegates to LayerDissector which is pure
        numpy/scipy — no PyTorch, no ONNX — so it works unchanged in production.
        """
        if self._layer_dissector is None:
            # Import lazily so we don't need to touch LayerDissector source
            import sys, pathlib
            backend_src = pathlib.Path(__file__).resolve().parents[2] / "backend"
            if str(backend_src) not in sys.path:
                sys.path.insert(0, str(backend_src))
            from app.ai.layer_dissector import LayerDissector
            self._layer_dissector = LayerDissector(device="cpu")
        return self._layer_dissector.decompose(volume, use_refinement=use_refinement)


# ---------------------------------------------------------------------------
# Finding extraction — identical logic to AnomalyService._extract_findings
# (duplicated here so the engine has no import dependency on AnomalyService)
# ---------------------------------------------------------------------------

def _extract_findings(
    heatmap: np.ndarray,
    volume: np.ndarray,
    overall_score: float,
    anomaly_threshold: float = 0.65,
) -> list[dict]:
    findings: list[dict] = []
    threshold = float(np.percentile(heatmap, 95))
    binary_mask = heatmap > threshold

    if not binary_mask.any():
        if overall_score < anomaly_threshold:
            findings.append({
                "region": "global",
                "description": (
                    "No significant anomalies detected. "
                    "The scan appears within normal limits."
                ),
                "confidence": float(1.0 - overall_score),
                "location": None,
                "severity": "normal",
            })
        return findings

    labeled, num_features = ndimage.label(binary_mask)

    for i in range(1, min(num_features + 1, 11)):
        component_mask = labeled == i
        component_size = int(component_mask.sum())
        if component_size < 10:
            continue

        center = ndimage.center_of_mass(component_mask)
        max_score = float(heatmap[component_mask].max())

        z_frac = center[0] / volume.shape[0]
        y_frac = center[1] / volume.shape[1]
        x_frac = center[2] / volume.shape[2]

        region   = _determine_region(z_frac, y_frac, x_frac)
        severity = _determine_severity(max_score, component_size, volume.size)
        mean_hu  = float(volume[component_mask].mean()) if component_mask.any() else 0.0
        description = _describe_finding(region, severity, mean_hu, component_size)

        findings.append({
            "region": region,
            "description": description,
            "confidence": float(min(max_score, 1.0)),
            "location": {
                "x": float(center[2]),
                "y": float(center[1]),
                "z": float(center[0]),
            },
            "severity": severity,
        })

    findings.sort(key=lambda f: f["confidence"], reverse=True)
    return findings


def _determine_region(z_frac: float, y_frac: float, x_frac: float) -> str:
    lr = "left"     if x_frac < 0.5 else "right"
    ud = "upper"    if z_frac < 0.4 else ("lower" if z_frac > 0.6 else "middle")
    ap = "anterior" if y_frac < 0.5 else "posterior"
    if 0.3 < x_frac < 0.7 and 0.3 < y_frac < 0.7:
        return f"{ud} central region"
    return f"{ud} {lr} {ap} region"


def _determine_severity(score: float, size: int, total_size: int) -> str:
    size_ratio = size / total_size
    if score > 0.85 or size_ratio > 0.05:
        return "severe"
    elif score > 0.70 or size_ratio > 0.02:
        return "moderate"
    elif score > 0.50:
        return "mild"
    return "minimal"


def _describe_finding(region: str, severity: str, mean_hu: float, size: int) -> str:
    if mean_hu < -500:
        tissue_type = "air-density"
        possible    = "emphysematous change or pneumothorax"
    elif mean_hu < -100:
        tissue_type = "low-density"
        possible    = "fatty or fluid-filled lesion"
    elif mean_hu < 200:
        tissue_type = "soft-tissue density"
        possible    = "mass, consolidation, or inflammation"
    else:
        tissue_type = "high-density"
        possible    = "calcification, bone lesion, or foreign body"

    return (
        f"{severity.capitalize()} {tissue_type} anomaly detected in the {region}. "
        f"Mean density: {mean_hu:.0f} HU. Size: {size} voxels. "
        f"Differential includes {possible}. Clinical correlation recommended."
    )


def _generate_summary(findings: list[dict], overall_score: float) -> str:
    if not findings:
        return "Analysis complete. No significant abnormalities identified."

    parts = [f"Analysis detected {len(findings)} region(s) of interest."]
    severe = [f for f in findings if f.get("severity") in ("severe", "moderate")]
    if severe:
        parts.append(f"{len(severe)} finding(s) warrant clinical attention.")
    for i, f in enumerate(findings[:3], 1):
        parts.append(f"Finding {i}: {f['description']}")
    parts.append(
        f"Overall anomaly confidence: {overall_score:.1%}. "
        "This is an AI-assisted analysis and should be reviewed by a qualified radiologist."
    )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Module-level singleton  (imported by config_prod.py and main_prod.py)
# ---------------------------------------------------------------------------

_engine_instance: Optional[OnnxInferenceEngine] = None
_engine_lock = threading.Lock()


def get_engine(onnx_dir: Path | None = None, volume_size: int = 128) -> OnnxInferenceEngine:
    """Return the module-level singleton, creating it on first call."""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                if onnx_dir is None:
                    raise RuntimeError(
                        "get_engine() called before engine was initialised. "
                        "Call get_engine(onnx_dir=...) once from lifespan."
                    )
                _engine_instance = OnnxInferenceEngine(
                    onnx_dir=onnx_dir,
                    volume_size=volume_size,
                )
    return _engine_instance
