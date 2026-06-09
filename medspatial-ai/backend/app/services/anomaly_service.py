"""
MedSpatial AI — Anomaly Service
Service layer that orchestrates the AI models for anomaly detection,
segmentation, and layer dissection on reconstructed volumes.
"""

import numpy as np
import torch
from pathlib import Path
from loguru import logger
from scipy import ndimage

from app.ai.anomaly_detector import AnomalyDetector3D, create_anomaly_detector
from app.ai.layer_dissector import LayerDissector
from app.ai.segmentation_net import SegmentationNet3D, create_segmentation_net
from app.ai.spatial_transformer import SpatialTransformer3D, create_spatial_transformer
from app.config import settings
from app.core.volume_processor import VolumeProcessor
from app.core.hardware_manager import get_hardware_manager


class AnomalyService:
    """
    Orchestrates AI-powered analysis of 3D medical volumes.
    Manages model loading, inference, and result formatting.
    """

    def __init__(self):
        hardware = get_hardware_manager()
        self.device = hardware.get_device()
        self.inference_size = hardware.adaptive_volume_size(settings.VOLUME_SIZE)
        if self.device.type == "cpu":
            self.inference_size = min(self.inference_size, 64)
        logger.info(f"Using device: {self.device}")
        if self.device.type == "cpu":
            logger.warning("GPU not available - analysis will be slow on CPU!")
        
        self.volume_proc = VolumeProcessor()
        self.layer_dissector = LayerDissector(device=str(self.device))

        # Lazy-load models on first use
        self._spatial_transformer: SpatialTransformer3D = None
        self._segmentation_net: SegmentationNet3D = None
        self._anomaly_detector: AnomalyDetector3D = None
        self._has_anomaly_weights = bool(
            settings.ANOMALY_DETECTOR_WEIGHTS
            and Path(settings.ANOMALY_DETECTOR_WEIGHTS).exists()
        )
        self._has_segmentation_weights = bool(
            settings.SEGMENTATION_WEIGHTS
            and Path(settings.SEGMENTATION_WEIGHTS).exists()
        )

    def _load_models(self):
        """Lazy-load AI models into memory."""
        if self._spatial_transformer is None:
            logger.info("Loading SpatialTransformer3D...")
            self._spatial_transformer = create_spatial_transformer(
                embed_dim=settings.EMBED_DIM,
                num_heads=settings.NUM_HEADS,
                num_layers=settings.NUM_LAYERS,
                pretrained_path=settings.SPATIAL_TRANSFORMER_WEIGHTS,
            ).to(self.device).eval()

        if self._anomaly_detector is None:
            logger.info("Loading AnomalyDetector3D...")
            self._anomaly_detector = create_anomaly_detector(
                latent_dim=256,
                input_size=self.inference_size,
                transformer_dim=settings.EMBED_DIM,
                pretrained_path=settings.ANOMALY_DETECTOR_WEIGHTS,
            ).to(self.device).eval()

        if self._segmentation_net is None:
            logger.info("Loading SegmentationNet3D...")
            self._segmentation_net = create_segmentation_net(
                num_classes=6,
                transformer_dim=settings.EMBED_DIM,
                pretrained_path=settings.SEGMENTATION_WEIGHTS,
            ).to(self.device).eval()

    def _prepare_volume(self, volume: np.ndarray, modality: str = "CT") -> torch.Tensor:
        """Normalize and resize volume for model input."""
        if modality.upper() == "CT":
            vol = self.volume_proc.clip_hu(volume)
        else:
            vol = np.asarray(volume, dtype=np.float32)
            finite = vol[np.isfinite(vol)]
            if finite.size:
                lower, upper = np.percentile(finite, [0.5, 99.5])
                vol = np.clip(vol, lower, upper)
        # Normalize to [0, 1]
        vol = self.volume_proc.normalize(vol)
        # Resize to model input size
        vol = self.volume_proc.resize_volume(vol, target_size=self.inference_size)
        # To tensor
        tensor = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        return tensor.to(self.device)

    def detect_anomalies(
        self,
        volume: np.ndarray,
        body_region: str = "unknown",
        modality: str = "CT",
    ) -> dict:
        """
        Run full anomaly detection pipeline.

        Args:
            volume: 3D numpy array in HU

        Returns:
            dict with heatmap, findings, overall_confidence, summary
        """
        if not self._has_anomaly_weights:
            return self._detect_anomalies_fallback(volume, body_region, modality)

        self._load_models()
        vol_tensor = self._prepare_volume(volume, modality)

        with torch.no_grad():
            # 1. Extract spatial features
            cls_features, spatial_features = self._spatial_transformer.extract_features(vol_tensor)

            # 2. Run anomaly detection
            anomaly_output = self._anomaly_detector(vol_tensor, transformer_features=cls_features)
            anomaly_map = anomaly_output["anomaly_map"].cpu().numpy()[0, 0]  # (D, H, W)
            anomaly_score = float(anomaly_output["anomaly_score"].cpu().item())

        # 3. Resize heatmap back to original volume dimensions
        from scipy import ndimage
        zoom_factors = [s / h for s, h in zip(volume.shape, anomaly_map.shape)]
        heatmap = ndimage.zoom(anomaly_map, zoom_factors, order=1)

        # 4. Extract findings from heatmap
        findings = self._extract_findings(
            heatmap, volume, anomaly_score, body_region, modality
        )

        # 5. Generate summary
        summary = self._generate_summary(findings, anomaly_score, volume)

        return {
            "heatmap": heatmap,
            "findings": findings,
            "overall_confidence": anomaly_score,
            "summary": summary,
        }

    def segment_organs(self, volume: np.ndarray, modality: str = "CT") -> dict:
        """
        Run organ segmentation on the volume.

        Returns:
            dict with mask (integer labels), class_names, statistics
        """
        if not self._has_segmentation_weights:
            return self._segment_fallback(volume, modality)

        self._load_models()
        vol_tensor = self._prepare_volume(volume, modality)

        with torch.no_grad():
            cls_features, spatial_features = self._spatial_transformer.extract_features(vol_tensor)
            seg_output = self._segmentation_net(vol_tensor, transformer_features=spatial_features)
            seg_logits = seg_output["segmentation"]
            seg_mask = torch.argmax(seg_logits, dim=1).cpu().numpy()[0]

        # Resize to original dimensions
        from scipy import ndimage
        zoom_factors = [s / m for s, m in zip(volume.shape, seg_mask.shape)]
        seg_mask_full = ndimage.zoom(seg_mask.astype(float), zoom_factors, order=0).astype(int)

        class_names = ["background", "bone", "soft_tissue", "air", "vessel", "anomaly"]
        stats = {}
        for i, name in enumerate(class_names):
            count = int((seg_mask_full == i).sum())
            stats[name] = {
                "voxel_count": count,
                "percentage": float(count / seg_mask_full.size * 100),
            }

        return {
            "mask": seg_mask_full,
            "class_names": class_names,
            "statistics": stats,
        }

    def analyze_full(
        self,
        volume: np.ndarray,
        body_region: str = "unknown",
        modality: str = "CT",
    ) -> dict:
        """Run anomaly and segmentation analysis using trained models or honest fallbacks."""
        if not self._has_anomaly_weights or not self._has_segmentation_weights:
            return {
                "anomaly": self.detect_anomalies(volume, body_region, modality),
                "segmentation": self.segment_organs(volume, modality),
                "methodology": "deterministic_fallback",
            }

        self._load_models()
        vol_tensor = self._prepare_volume(volume, modality)
        with torch.inference_mode():
            cls_features, spatial_features = self._spatial_transformer.extract_features(vol_tensor)
            anomaly_output = self._anomaly_detector(vol_tensor, transformer_features=cls_features)
            seg_output = self._segmentation_net(vol_tensor, transformer_features=spatial_features)

        anomaly_map = anomaly_output["anomaly_map"].cpu().numpy()[0, 0]
        anomaly_score = float(anomaly_output["anomaly_score"].cpu().item())
        heatmap = ndimage.zoom(
            anomaly_map,
            [size / current for size, current in zip(volume.shape, anomaly_map.shape)],
            order=1,
        )
        seg_mask = torch.argmax(seg_output["segmentation"], dim=1).cpu().numpy()[0]
        seg_mask = ndimage.zoom(
            seg_mask.astype(float),
            [size / current for size, current in zip(volume.shape, seg_mask.shape)],
            order=0,
        ).astype(np.uint8)
        findings = self._extract_findings(
            heatmap, volume, anomaly_score, body_region, modality
        )
        return {
            "anomaly": {
                "heatmap": heatmap,
                "findings": findings,
                "overall_confidence": anomaly_score,
                "summary": self._generate_summary(findings, anomaly_score, volume),
            },
            "segmentation": self._segmentation_result(seg_mask),
            "methodology": "trained_neural_shared_features",
        }

    def _detect_anomalies_fallback(
        self, volume: np.ndarray, body_region: str, modality: str
    ) -> dict:
        """Fast robust-residual baseline used when validated model weights are absent."""
        if modality.upper() == "CT":
            work = self.volume_proc.clip_hu(volume).astype(np.float32, copy=False)
        else:
            work = np.asarray(volume, dtype=np.float32)
            finite = work[np.isfinite(work)]
            if finite.size:
                lower, upper = np.percentile(finite, [0.5, 99.5])
                work = np.clip(work, lower, upper)
        scale = max(work.shape) / 96.0
        if scale > 1:
            small = ndimage.zoom(work, [1 / scale] * 3, order=1)
        else:
            small = work
        smooth = ndimage.gaussian_filter(small, sigma=1.25)
        residual = np.abs(small - smooth)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median))) + 1e-6
        robust_z = np.clip((residual - median) / (1.4826 * mad), 0, 12)
        heatmap_small = (robust_z / 12.0).astype(np.float32)
        heatmap = (
            ndimage.zoom(heatmap_small, [size / current for size, current in zip(work.shape, heatmap_small.shape)], order=1)
            if heatmap_small.shape != work.shape else heatmap_small
        )
        body_mask = work > -500 if modality.upper() == "CT" else np.isfinite(work)
        score = float(np.percentile(heatmap[body_mask], 99)) if body_mask.any() else 0.0
        findings = self._extract_findings(
            heatmap, work, score, body_region, modality
        )
        return {
            "heatmap": heatmap,
            "findings": findings,
            "overall_confidence": score,
            "summary": (
                self._generate_summary(findings, score, work)
                + (
                    " Method: deterministic robust local-density deviation; trained anomaly weights were not configured."
                    if modality.upper() == "CT"
                    else " Method: deterministic robust local-signal deviation; trained anomaly weights were not configured."
                )
            ),
        }

    def _segment_fallback(self, volume: np.ndarray, modality: str = "CT") -> dict:
        mask = np.zeros(volume.shape, dtype=np.uint8)
        if modality.upper() == "CT":
            mask[(volume >= 200)] = 1
            mask[(volume >= -100) & (volume < 200)] = 2
            mask[(volume < -500)] = 3
            mask[(volume >= 120) & (volume < 500)] = 4
        else:
            finite = volume[np.isfinite(volume)]
            if finite.size:
                q20, q50, q80, q95 = np.percentile(finite, [20, 50, 80, 95])
                mask[volume <= q20] = 1
                mask[(volume > q20) & (volume <= q50)] = 2
                mask[(volume > q50) & (volume <= q80)] = 3
                mask[(volume > q80) & (volume <= q95)] = 4
                mask[volume > q95] = 5
        return self._segmentation_result(mask)

    @staticmethod
    def _segmentation_result(mask: np.ndarray) -> dict:
        class_names = ["background", "bone", "soft_tissue", "air", "vessel", "anomaly"]
        stats = {
            name: {
                "voxel_count": int((mask == index).sum()),
                "percentage": float((mask == index).sum() / mask.size * 100),
            }
            for index, name in enumerate(class_names)
        }
        return {"mask": mask, "class_names": class_names, "statistics": stats}

    def dissect_layers(self, volume: np.ndarray, use_refinement: bool = True) -> dict:
        """
        Decompose volume into tissue layers.

        Returns:
            dict per layer with mask, statistics, color
        """
        return self.layer_dissector.decompose(volume, use_refinement=use_refinement)

    def generate_synthetic_layers(
        self, shape: tuple = (128, 128, 128), tissue_type: str = "chest"
    ) -> dict:
        """Generate layers without images (synthetic anatomy)."""
        return self.layer_dissector.decompose_without_images(shape, tissue_type)

    def _extract_findings(
        self,
        heatmap: np.ndarray,
        volume: np.ndarray,
        overall_score: float,
        body_region: str = "unknown",
        modality: str = "CT",
    ) -> list[dict]:
        """Extract structured findings from the anomaly heatmap."""
        from scipy import ndimage

        findings = []

        # Threshold the heatmap to find anomalous regions
        threshold = np.percentile(heatmap, 95)
        binary_mask = heatmap > threshold

        if not binary_mask.any():
            if overall_score < settings.ANOMALY_THRESHOLD:
                findings.append({
                    "region": "global",
                    "description": "No significant anomalies detected. The scan appears within normal limits.",
                    "confidence": float(1 - overall_score),
                    "location": None,
                    "severity": "normal",
                })
            return findings

        # Label connected components
        labeled, num_features = ndimage.label(binary_mask)

        for i in range(1, min(num_features + 1, 11)):  # max 10 findings
            component_mask = labeled == i
            component_size = component_mask.sum()

            if component_size < 10:
                continue

            # Get center of mass
            center = ndimage.center_of_mass(component_mask)
            max_score = float(heatmap[component_mask].max())

            # Determine anatomical region and severity
            z_frac = center[0] / volume.shape[0]
            y_frac = center[1] / volume.shape[1]
            x_frac = center[2] / volume.shape[2]

            region = self._determine_region(z_frac, y_frac, x_frac)
            severity = self._determine_severity(max_score, component_size, volume.size)

            # Get mean HU in region
            mean_hu = float(volume[component_mask].mean()) if component_mask.any() else 0.0
            description = self._describe_finding(
                region, severity, mean_hu, component_size, body_region, modality
            )

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

    def _determine_region(self, z_frac: float, y_frac: float, x_frac: float) -> str:
        """Determine anatomical region from normalized coordinates."""
        lr = "left" if x_frac < 0.5 else "right"
        ud = "upper" if z_frac < 0.4 else "lower" if z_frac > 0.6 else "middle"
        ap = "anterior" if y_frac < 0.5 else "posterior"

        if 0.3 < x_frac < 0.7 and 0.3 < y_frac < 0.7:
            return f"{ud} central region"
        return f"{ud} {lr} {ap} region"

    def _determine_severity(self, score: float, size: int, total_size: int) -> str:
        """Determine severity based on score and relative size."""
        size_ratio = size / total_size
        if score > 0.85 or size_ratio > 0.05:
            return "severe"
        elif score > 0.7 or size_ratio > 0.02:
            return "moderate"
        elif score > 0.5:
            return "mild"
        return "minimal"

    def _describe_finding(
        self,
        region: str,
        severity: str,
        mean_hu: float,
        size: int,
        body_region: str = "unknown",
        modality: str = "CT",
    ) -> str:
        """Generate a descriptive text for a finding."""
        if modality.upper() != "CT":
            return (
                f"{severity.capitalize()} local signal-deviation candidate in the "
                f"{region} of the {body_region} study. Relative mean signal: "
                f"{mean_hu:.1f}; size: {size} voxels. This is not a diagnosis and "
                "requires sequence-aware radiologist review."
            )
        if body_region == "spine":
            return (
                f"{severity.capitalize()} density-deviation candidate in the {region} "
                f"of the spine. Mean density: {mean_hu:.0f} HU; size: {size} voxels. "
                "Review alignment, cortical continuity, canal relationship, and marrow "
                "appearance on the source slices."
            )
        # Determine tissue type from HU
        if mean_hu < -500:
            tissue_type = "air-density"
            possible = "emphysematous change or pneumothorax"
        elif mean_hu < -100:
            tissue_type = "low-density"
            possible = "fatty or fluid-filled lesion"
        elif mean_hu < 200:
            tissue_type = "soft-tissue density"
            possible = "mass, consolidation, or inflammation"
        else:
            tissue_type = "high-density"
            possible = "calcification, bone lesion, or foreign body"

        return (
            f"{severity.capitalize()} {tissue_type} anomaly detected in the {region}. "
            f"Mean density: {mean_hu:.0f} HU. Size: {size} voxels. "
            f"Differential includes {possible}. Clinical correlation recommended."
        )

    def _generate_summary(self, findings: list[dict], overall_score: float, volume: np.ndarray) -> str:
        """Generate a text summary of all findings."""
        if not findings:
            return "Analysis complete. No significant abnormalities identified."

        summary_parts = [f"Analysis detected {len(findings)} region(s) of interest."]

        severe = [f for f in findings if f.get("severity") in ("severe", "moderate")]
        if severe:
            summary_parts.append(
                f"{len(severe)} finding(s) warrant clinical attention."
            )

        for i, finding in enumerate(findings[:3], 1):
            summary_parts.append(f"Finding {i}: {finding['description']}")

        summary_parts.append(
            f"Overall anomaly confidence: {overall_score:.1%}. "
            "This is an AI-assisted analysis and should be reviewed by a qualified radiologist."
        )

        return " ".join(summary_parts)
