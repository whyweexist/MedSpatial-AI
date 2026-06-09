"""
MedSpatial AI — Explainability Module (XAI)
Provides Grad-CAM 3D, anomaly attribution maps, reasoning chain generation,
and LRP-style relevance for segmentation. All features degrade gracefully
when model weights are unavailable.
"""

import gc
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger


@dataclass
class ReasoningStep:
    """A single step in the XAI reasoning chain."""
    category: str  # "anomaly_evidence", "anatomical_context", etc.
    description: str
    confidence: float
    evidence_type: str  # "heatmap", "statistical", "atlas_comparison"
    data: Optional[dict[str, Any]] = None


@dataclass
class ReasoningChain:
    """Full reasoning chain for a finding."""
    finding: str
    confidence: float
    steps: list[ReasoningStep]
    anatomical_context: str
    differential: list[str]
    bbox_3d: Optional[dict[str, float]] = None
    representative_slice_idx: Optional[int] = None
    evidence_references: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


@dataclass
class XAIResult:
    """Complete XAI output for a scan."""
    grad_cam_heatmaps: dict[str, np.ndarray]  # disease_class → 3D heatmap
    anomaly_attribution: Optional[np.ndarray]  # voxel-level attribution
    reasoning_chains: list[ReasoningChain]
    segmentation_relevance: Optional[dict[str, np.ndarray]] = None  # class → relevance map
    methodology: str = "evidence_attribution"
    limitations: list[str] = field(default_factory=list)


class ExplainabilityEngine:
    """
    Generates explainable AI outputs for medical image analysis.
    All methods degrade gracefully when models are unavailable.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)

    def compute_grad_cam_3d(
        self,
        model: Optional[torch.nn.Module],
        volume_tensor: torch.Tensor,
        target_class: int,
        layer_name: str = "sage_layers.2",
    ) -> np.ndarray:
        """
        Compute Grad-CAM 3D for a specific disease class.

        Args:
            model: the classification model (AnomalyGraph or similar)
            volume_tensor: (1, 1, D, H, W) input volume
            target_class: index of the disease class to explain
            layer_name: name of the layer to hook into

        Returns:
            3D heatmap (D, H, W) normalized to [0, 1]
        """
        D, H, W = volume_tensor.shape[2], volume_tensor.shape[3], volume_tensor.shape[4]

        if model is None:
            logger.warning("No model for Grad-CAM; attribution is unavailable")
            return np.zeros((D, H, W), dtype=np.float32)

        try:
            activations: list[torch.Tensor] = []
            gradients: list[torch.Tensor] = []

            def forward_hook(module: torch.nn.Module, inp: Any, output: torch.Tensor) -> None:
                activations.append(output.detach())

            def backward_hook(module: torch.nn.Module, grad_in: Any, grad_out: Any) -> None:
                gradients.append(grad_out[0].detach())

            # Find the target layer
            target_layer = None
            for name, module in model.named_modules():
                if name == layer_name:
                    target_layer = module
                    break

            if target_layer is None:
                # Fall back to last layer with parameters
                modules_list = list(model.named_modules())
                for name, module in reversed(modules_list):
                    if list(module.parameters()):
                        target_layer = module
                        layer_name = name
                        break

            if target_layer is None:
                return np.zeros((D, H, W), dtype=np.float32)

            handle_fwd = target_layer.register_forward_hook(forward_hook)
            handle_bwd = target_layer.register_full_backward_hook(backward_hook)

            # Forward pass with gradients enabled
            model.eval()
            volume_tensor = volume_tensor.to(self.device).requires_grad_(True)

            with torch.enable_grad():
                output = model(volume_tensor)

                # Get the target logit
                if isinstance(output, dict):
                    logits = output.get("disease_logits", output.get("disease_probs"))
                else:
                    logits = output

                if logits is None:
                    handle_fwd.remove()
                    handle_bwd.remove()
                    return np.zeros((D, H, W), dtype=np.float32)

                target_logit = logits[0, target_class]
                model.zero_grad()
                target_logit.backward(retain_graph=False)

            handle_fwd.remove()
            handle_bwd.remove()

            if not activations or not gradients:
                return np.zeros((D, H, W), dtype=np.float32)

            act = activations[0]
            grad = gradients[0]

            # Compute channel-wise weights via global average pooling of gradients
            if grad.dim() == 3:
                weights = grad.mean(dim=(1, 2))  # (B, C) or similar
            elif grad.dim() == 4:
                weights = grad.mean(dim=(2, 3))
            else:
                weights = grad.mean(dim=tuple(range(2, grad.dim())))

            # Weighted combination of activation maps
            cam = torch.zeros(act.shape[2:], device=self.device)
            for i in range(min(weights.shape[-1], act.shape[1] if act.dim() > 1 else 1)):
                if act.dim() > 1:
                    cam += weights[0, i] * act[0, i]

            cam = F.relu(cam)

            # Upsample to volume dimensions
            if cam.dim() == 1:
                # Reshape from flat patches to 3D grid
                grid_size = int(round(cam.shape[0] ** (1.0 / 3.0)))
                if grid_size ** 3 == cam.shape[0]:
                    cam = cam.view(grid_size, grid_size, grid_size)
                else:
                    cam = cam.view(1, 1, -1)

            cam_np = cam.cpu().numpy().astype(np.float32)

            # Resize to full volume
            from scipy import ndimage as ndi
            if cam_np.shape != (D, H, W):
                zoom_factors = [D / cam_np.shape[0], H / cam_np.shape[1], W / cam_np.shape[2]] if cam_np.ndim == 3 else [D, H, W]
                cam_np = ndi.zoom(cam_np, zoom_factors[:cam_np.ndim], order=1)
                if cam_np.ndim == 1:
                    cam_np = cam_np.reshape(D, H, W) if cam_np.size == D * H * W else np.zeros((D, H, W), dtype=np.float32)

            # Ensure correct shape
            if cam_np.shape != (D, H, W):
                cam_np = np.zeros((D, H, W), dtype=np.float32)

            # Normalize to [0, 1]
            cam_max = cam_np.max()
            if cam_max > 1e-8:
                cam_np = cam_np / cam_max

            return cam_np

        except Exception as exc:
            logger.error(f"Grad-CAM failed: {exc}")
            return np.zeros((D, H, W), dtype=np.float32)
        finally:
            gc.collect()

    def compute_anomaly_attribution(
        self,
        patch_embeddings: Optional[np.ndarray],
        memory_bank: Optional[np.ndarray],
        volume_shape: tuple[int, int, int],
        patch_size: int = 8,
    ) -> np.ndarray:
        """
        Compute per-voxel anomaly attribution from PatchCore-style analysis.

        Args:
            patch_embeddings: (N, D) embedded patches
            memory_bank: (M, D) normal memory bank entries
            volume_shape: (D, H, W) target output shape
            patch_size: size of each patch

        Returns:
            attribution map (D, H, W) in [0, 1]
        """
        D, H, W = volume_shape

        if patch_embeddings is None or memory_bank is None:
            logger.warning("No embeddings for attribution; attribution is unavailable")
            return np.zeros((D, H, W), dtype=np.float32)

        try:
            # Compute cosine similarity between each patch and nearest bank entry
            emb = torch.from_numpy(patch_embeddings).float()
            bank = torch.from_numpy(memory_bank).float()

            emb_norm = F.normalize(emb, dim=-1)
            bank_norm = F.normalize(bank, dim=-1)

            similarities = torch.mm(emb_norm, bank_norm.T)
            max_sim, _ = similarities.max(dim=-1)
            attribution_scores = (1.0 - max_sim.clamp(0, 1)).numpy()

            # Reshape to 3D patch grid
            grid_d = D // patch_size
            grid_h = H // patch_size
            grid_w = W // patch_size
            expected_patches = grid_d * grid_h * grid_w

            if len(attribution_scores) == expected_patches:
                attr_3d = attribution_scores.reshape(grid_d, grid_h, grid_w)
            else:
                n = len(attribution_scores)
                side = int(round(n ** (1.0 / 3.0)))
                attr_3d = attribution_scores[:side ** 3].reshape(side, side, side)

            # Upsample to full resolution
            from scipy import ndimage as ndi
            zoom = [D / attr_3d.shape[0], H / attr_3d.shape[1], W / attr_3d.shape[2]]
            attr_full = ndi.zoom(attr_3d, zoom, order=1).astype(np.float32)

            # Normalize
            attr_max = attr_full.max()
            if attr_max > 1e-8:
                attr_full = attr_full / attr_max

            return attr_full

        except Exception as exc:
            logger.error(f"Anomaly attribution failed: {exc}")
            return np.zeros((D, H, W), dtype=np.float32)

    def generate_reasoning_chain(
        self,
        findings: list[dict],
        disease_probs: Optional[np.ndarray],
        anomaly_map: Optional[np.ndarray],
        volume: Optional[np.ndarray],
        tissue_results: Optional[list] = None,
        modality: str = "CT",
        body_region: str = "unknown",
    ) -> list[ReasoningChain]:
        """
        Generate structured reasoning chains for all findings.

        Args:
            findings: list of finding dicts from anomaly service
            disease_probs: (13,) disease probability vector
            anomaly_map: (D, H, W) anomaly heatmap
            volume: original HU volume
            tissue_results: tissue reconstruction results

        Returns:
            list of ReasoningChain objects
        """
        from app.ai.anomaly_graph import DISEASE_LABELS

        chains: list[ReasoningChain] = []

        for finding in findings:
            steps: list[ReasoningStep] = []
            evidence_references: list[dict[str, Any]] = []
            limitations: list[str] = []
            location = finding.get("location", {})
            confidence = finding.get("confidence", 0.0)
            severity = finding.get("severity", "unknown")
            region = finding.get("region", "unspecified")

            # Step 1: Anomaly evidence
            if anomaly_map is not None and location:
                z = int(location.get("z", 0))
                y = int(location.get("y", 0))
                x = int(location.get("x", 0))
                z = min(z, anomaly_map.shape[0] - 1)
                y = min(y, anomaly_map.shape[1] - 1)
                x = min(x, anomaly_map.shape[2] - 1)
                local_score = float(anomaly_map[z, y, x])

                steps.append(ReasoningStep(
                    category="anomaly_evidence",
                    description=f"Anomaly score at location ({x},{y},{z}): {local_score:.3f}",
                    confidence=local_score,
                    evidence_type="heatmap",
                    data={"score": local_score, "threshold": 0.65},
                ))
                evidence_references.append({
                    "type": "heatmap",
                    "slice_index": z,
                    "location": {"x": x, "y": y, "z": z},
                    "score": local_score,
                })

            # Step 2: modality-aware local signal analysis
            if volume is not None and location:
                z = min(int(location.get("z", 0)), volume.shape[0] - 1)
                y = min(int(location.get("y", 0)), volume.shape[1] - 1)
                x = min(int(location.get("x", 0)), volume.shape[2] - 1)

                # Sample neighborhood
                r = 5
                z_s, z_e = max(0, z - r), min(volume.shape[0], z + r)
                y_s, y_e = max(0, y - r), min(volume.shape[1], y + r)
                x_s, x_e = max(0, x - r), min(volume.shape[2], x + r)
                neighborhood = volume[z_s:z_e, y_s:y_e, x_s:x_e]

                mean_value = float(neighborhood.mean())
                std_value = float(neighborhood.std())
                is_ct = modality.upper() == "CT"
                unit = "HU" if is_ct else "relative intensity"
                interpretation = (
                    self._interpret_hu(mean_value)
                    if is_ct
                    else "No CT density interpretation was applied because MRI signal is sequence-dependent."
                )

                steps.append(ReasoningStep(
                    category="intensity_analysis",
                    description=(
                        f"Local signal mean={mean_value:.1f} {unit}, "
                        f"spread={std_value:.1f}. {interpretation}"
                    ),
                    confidence=confidence,
                    evidence_type="statistical",
                    data={"mean": mean_value, "standard_deviation": std_value, "unit": unit},
                ))
                evidence_references.append({
                    "type": "source_slice",
                    "slice_index": z,
                    "location": {"x": x, "y": y, "z": z},
                })

            # Step 3: Disease classification evidence
            if disease_probs is not None:
                top_indices = np.argsort(disease_probs)[::-1][:3]
                for idx in top_indices:
                    prob = float(disease_probs[idx])
                    if prob > 0.1:
                        label = DISEASE_LABELS[idx] if idx < len(DISEASE_LABELS) else f"Class {idx}"
                        steps.append(ReasoningStep(
                            category="classification_evidence",
                            description=f"Disease probability: {label} = {prob:.1%}",
                            confidence=prob,
                            evidence_type="statistical",
                            data={"disease": label, "probability": prob},
                        ))

            # Step 4: Anatomical context
            steps.append(ReasoningStep(
                category="anatomical_context",
                description=f"Candidate signal is located in the {region} of the {body_region} study.",
                confidence=confidence,
                evidence_type="statistical",
                data={"region": region, "severity": severity},
            ))

            # Build differential diagnosis
            differential = self._build_differential(finding, disease_probs, body_region)
            limitations.append(
                "This explanation summarizes model evidence and possible considerations; it does not establish a diagnosis."
            )
            if disease_probs is None:
                limitations.append("No calibrated disease classifier probabilities were available.")

            # Representative slice
            rep_slice = int(location.get("z", 0)) if location else None

            # 3D bounding box
            bbox = None
            if location:
                bbox = {
                    "x_min": float(location.get("x", 0)) - 10,
                    "x_max": float(location.get("x", 0)) + 10,
                    "y_min": float(location.get("y", 0)) - 10,
                    "y_max": float(location.get("y", 0)) + 10,
                    "z_min": float(location.get("z", 0)) - 5,
                    "z_max": float(location.get("z", 0)) + 5,
                }

            chains.append(ReasoningChain(
                finding=finding.get("description", "Finding"),
                confidence=confidence,
                steps=steps,
                anatomical_context=region,
                differential=differential,
                bbox_3d=bbox,
                representative_slice_idx=rep_slice,
                evidence_references=evidence_references,
                limitations=limitations,
            ))

        logger.info(f"Generated {len(chains)} reasoning chains")
        return chains

    def compute_full_xai(
        self,
        model: Optional[torch.nn.Module],
        volume_tensor: Optional[torch.Tensor],
        findings: list[dict],
        disease_probs: Optional[np.ndarray],
        anomaly_map: Optional[np.ndarray],
        volume: Optional[np.ndarray],
        tissue_results: Optional[list] = None,
        modality: str = "CT",
        body_region: str = "unknown",
    ) -> XAIResult:
        """
        Compute all XAI outputs in one call.

        Returns:
            XAIResult with all heatmaps, attributions, and reasoning chains
        """
        from app.ai.anomaly_graph import DISEASE_LABELS

        # Grad-CAM heatmaps for top disease classes
        grad_cam_maps: dict[str, np.ndarray] = {}
        if disease_probs is not None and volume_tensor is not None:
            top_classes = np.argsort(disease_probs)[::-1][:3]
            for cls_idx in top_classes:
                if disease_probs[cls_idx] > 0.1:
                    label = DISEASE_LABELS[cls_idx] if cls_idx < len(DISEASE_LABELS) else f"class_{cls_idx}"
                    heatmap = self.compute_grad_cam_3d(model, volume_tensor, int(cls_idx))
                    grad_cam_maps[label] = heatmap
                    gc.collect()

        limitations: list[str] = []
        if not grad_cam_maps and anomaly_map is not None:
            attribution_map = anomaly_map.astype(np.float32, copy=False)
            maximum = float(attribution_map.max())
            if maximum > 1e-8:
                attribution_map = attribution_map / maximum
            grad_cam_maps["analysis_attribution"] = attribution_map
            limitations.append(
                "A trained classifier was unavailable; this is anomaly attribution rather than Grad-CAM."
            )
        elif not grad_cam_maps:
            limitations.append("No model-derived spatial attribution map was available.")

        # Anomaly attribution
        attribution = anomaly_map  # reuse existing anomaly map as attribution

        # Reasoning chains
        chains = self.generate_reasoning_chain(
            findings,
            disease_probs,
            anomaly_map,
            volume,
            tissue_results,
            modality=modality,
            body_region=body_region,
        )

        return XAIResult(
            grad_cam_heatmaps=grad_cam_maps,
            anomaly_attribution=attribution,
            reasoning_chains=chains,
            segmentation_relevance=None,
            methodology="model_grad_cam" if model is not None else "evidence_attribution",
            limitations=limitations,
        )

    def _synthetic_grad_cam(self, d: int, h: int, w: int) -> np.ndarray:
        """Return an explicit unavailable map; never fabricate attribution."""
        return np.zeros((d, h, w), dtype=np.float32)

    def _synthetic_attribution(self, d: int, h: int, w: int) -> np.ndarray:
        """Return an explicit unavailable map; never fabricate attribution."""
        return np.zeros((d, h, w), dtype=np.float32)

    def _interpret_hu(self, hu_value: float) -> str:
        """Interpret a HU value in clinical terms."""
        if hu_value < -500:
            return "Air-density region (emphysema, pneumothorax, or normal aeration)"
        elif hu_value < -100:
            return "Low-density (fat, fluid, or hypo-attenuating tissue)"
        elif hu_value < 60:
            return "Soft-tissue density (normal parenchyma, fluid, or inflammation)"
        elif hu_value < 200:
            return "Enhanced soft tissue (vascular enhancement or solid lesion)"
        elif hu_value < 700:
            return "Dense calcification or contrast agent"
        else:
            return "Very dense material (cortical bone or metallic artifact)"

    def _build_differential(
        self,
        finding: dict,
        disease_probs: Optional[np.ndarray],
        body_region: str = "unknown",
    ) -> list[str]:
        """Build a differential diagnosis list."""
        from app.ai.anomaly_graph import DISEASE_LABELS

        differential: list[str] = []
        if body_region == "spine":
            differential.extend([
                "Degenerative change",
                "Compression deformity",
                "Alignment variation",
                "Marrow signal variation",
            ])

        # Description-based chest considerations are only valid for chest studies.
        description = finding.get("description", "").lower()
        if body_region == "chest":
            if "air" in description or "pneumo" in description:
                differential.extend(["Pneumothorax", "Emphysema", "Bulla"])
            elif "fluid" in description or "effusion" in description:
                differential.extend(["Pleural effusion", "Hemothorax", "Empyema"])
            elif "mass" in description or "nodule" in description:
                differential.extend(["Pulmonary nodule", "Lung carcinoma", "Metastasis", "Granuloma"])
            elif "consolidation" in description:
                differential.extend(["Pneumonia", "Pulmonary hemorrhage", "Atelectasis"])

        # From classification probabilities
        if disease_probs is not None:
            top_3 = np.argsort(disease_probs)[::-1][:3]
            for idx in top_3:
                if disease_probs[idx] > 0.15 and idx < len(DISEASE_LABELS):
                    label = DISEASE_LABELS[idx]
                    if label not in differential and label != "Normal/No Finding":
                        differential.append(label)

        if not differential:
            differential = ["Nonspecific finding — clinical correlation recommended"]

        return differential[:5]
