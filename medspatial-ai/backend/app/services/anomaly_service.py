"""
MedSpatial AI — Anomaly Service
Service layer that orchestrates the AI models for anomaly detection,
segmentation, and layer dissection on reconstructed volumes.
"""

import numpy as np
import torch
from loguru import logger

from app.ai.anomaly_detector import AnomalyDetector3D, create_anomaly_detector
from app.ai.layer_dissector import LayerDissector
from app.ai.segmentation_net import SegmentationNet3D, create_segmentation_net
from app.ai.spatial_transformer import SpatialTransformer3D, create_spatial_transformer
from app.config import settings
from app.core.volume_processor import VolumeProcessor


class AnomalyService:
    """
    Orchestrates AI-powered analysis of 3D medical volumes.
    Manages model loading, inference, and result formatting.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.volume_proc = VolumeProcessor()
        self.layer_dissector = LayerDissector(device=str(self.device))

        # Lazy-load models on first use
        self._spatial_transformer: SpatialTransformer3D = None
        self._segmentation_net: SegmentationNet3D = None
        self._anomaly_detector: AnomalyDetector3D = None

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
                input_size=settings.VOLUME_SIZE,
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

    def _prepare_volume(self, volume: np.ndarray) -> torch.Tensor:
        """Normalize and resize volume for model input."""
        # Clip HU range
        vol = self.volume_proc.clip_hu(volume)
        # Normalize to [0, 1]
        vol = self.volume_proc.normalize(vol)
        # Resize to model input size
        vol = self.volume_proc.resize_volume(vol, target_size=settings.VOLUME_SIZE)
        # To tensor
        tensor = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        return tensor.to(self.device)

    def detect_anomalies(self, volume: np.ndarray) -> dict:
        """
        Run full anomaly detection pipeline.

        Args:
            volume: 3D numpy array in HU

        Returns:
            dict with heatmap, findings, overall_confidence, summary
        """
        self._load_models()

        vol_tensor = self._prepare_volume(volume)

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
        findings = self._extract_findings(heatmap, volume, anomaly_score)

        # 5. Generate summary
        summary = self._generate_summary(findings, anomaly_score, volume)

        return {
            "heatmap": heatmap,
            "findings": findings,
            "overall_confidence": anomaly_score,
            "summary": summary,
        }

    def segment_organs(self, volume: np.ndarray) -> dict:
        """
        Run organ segmentation on the volume.

        Returns:
            dict with mask (integer labels), class_names, statistics
        """
        self._load_models()

        vol_tensor = self._prepare_volume(volume)

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
        self, heatmap: np.ndarray, volume: np.ndarray, overall_score: float
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
            description = self._describe_finding(region, severity, mean_hu, component_size)

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

    def _describe_finding(self, region: str, severity: str, mean_hu: float, size: int) -> str:
        """Generate a descriptive text for a finding."""
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
