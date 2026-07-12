"""
MedSpatial AI — Body Part Labeler
Computes spatial labels for anatomical structures from segmentation masks.
Generates centroids, volumes, and anatomical context strings for the 3D viewer.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger
from scipy import ndimage


@dataclass
class AnatomyLabel:
    """A single anatomical label for 3D viewer display."""
    name: str
    position: tuple[float, float, float]  # world-space mm (x, y, z)
    volume_mm3: float
    color: tuple[float, float, float]
    layer_index: int
    description: str = ""
    sub_labels: Optional[list["AnatomyLabel"]] = None


class BodyPartLabeler:
    """
    Generates spatial labels for anatomical structures from segmentation
    and reconstruction results.
    """

    # Bone sub-structure spatial heuristics (relative z-position in volume)
    _BONE_STRUCTURES = {
        "Clavicles": {"z_range": (0.0, 0.15), "x_range": (0.15, 0.85), "y_range": (0.0, 0.4)},
        "Sternum": {"z_range": (0.05, 0.6), "x_range": (0.40, 0.60), "y_range": (0.0, 0.35)},
        "Ribs": {"z_range": (0.05, 0.85), "x_range": (0.1, 0.9), "y_range": (0.0, 0.7)},
        "Spine": {"z_range": (0.0, 1.0), "x_range": (0.35, 0.65), "y_range": (0.6, 1.0)},
        "Scapulae": {"z_range": (0.0, 0.4), "x_range": (0.0, 0.3), "y_range": (0.5, 1.0)},
    }

    def generate_labels(
        self,
        tissue_results: list,
        voxel_spacing: np.ndarray,
        volume_shape: tuple[int, int, int],
        anomaly_findings: Optional[list[dict]] = None,
    ) -> list[AnatomyLabel]:
        """
        Generate anatomical labels from tissue reconstruction results.

        Args:
            tissue_results: list of TissueResult from SurfaceProcessor
            voxel_spacing: [z, y, x] spacing in mm
            volume_shape: (D, H, W) of the original volume
            anomaly_findings: optional list of anomaly finding dicts

        Returns:
            list of AnatomyLabel for 3D viewer
        """
        labels: list[AnatomyLabel] = []

        for tissue in tissue_results:
            if tissue.mesh_path is None or tissue.voxel_count < 10:
                continue

            if tissue.centroid_mm is None:
                continue

            # Normalize position to viewer coordinates
            position = self._centroid_to_viewer_coords(
                tissue.centroid_mm, voxel_spacing, volume_shape
            )

            label = AnatomyLabel(
                name=self._format_tissue_name(tissue.name),
                position=position,
                volume_mm3=tissue.volume_mm3,
                color=tissue.color_rgb,
                layer_index=tissue.label_index,
                description=self._generate_description(tissue),
            )

            labels.append(label)

        # Generate anomaly context labels
        if anomaly_findings:
            for finding in anomaly_findings:
                anomaly_label = self._create_anomaly_label(
                    finding, voxel_spacing, volume_shape
                )
                if anomaly_label:
                    labels.append(anomaly_label)

        logger.info(f"Generated {len(labels)} anatomy labels")
        return labels

    def generate_bone_sublabels(
        self,
        bone_mask: np.ndarray,
        voxel_spacing: np.ndarray,
    ) -> list[AnatomyLabel]:
        """
        Perform connected component analysis on bone mask to identify
        individual structures (ribs, vertebrae, clavicles, etc.)
        using spatial heuristics.

        Args:
            bone_mask: binary 3D mask of bone tissue
            voxel_spacing: [z, y, x] spacing in mm

        Returns:
            list of AnatomyLabel for bone sub-structures
        """
        labels: list[AnatomyLabel] = []

        if bone_mask.sum() < 50:
            return labels

        labeled, num_features = ndimage.label(bone_mask.astype(np.int32))
        if num_features < 1:
            return labels

        D, H, W = bone_mask.shape

        # Classify each connected component by spatial position
        for i in range(1, min(num_features + 1, 100)):
            component = (labeled == i)
            size = component.sum()
            if size < 20:
                continue

            centroid = ndimage.center_of_mass(component)
            z_frac = centroid[0] / D
            y_frac = centroid[1] / H
            x_frac = centroid[2] / W

            # Classify by spatial heuristics
            structure_name = self._classify_bone_component(
                z_frac, y_frac, x_frac, size, D * H * W
            )

            centroid_mm = np.array(centroid) * voxel_spacing
            volume_mm3 = float(size * np.prod(voxel_spacing))

            labels.append(AnatomyLabel(
                name=structure_name,
                position=(float(centroid_mm[2]), float(centroid_mm[1]), float(centroid_mm[0])),
                volume_mm3=volume_mm3,
                color=(0.95, 0.92, 0.80),
                layer_index=3,
                description=f"Bone structure, {volume_mm3 / 1000:.1f} cm³",
            ))

        # Deduplicate — merge labels with same name by averaging positions
        merged: dict[str, list[AnatomyLabel]] = {}
        for label in labels:
            if label.name not in merged:
                merged[label.name] = []
            merged[label.name].append(label)

        result: list[AnatomyLabel] = []
        for name, group in merged.items():
            avg_pos = tuple(
                float(np.mean([l.position[i] for l in group]))
                for i in range(3)
            )
            total_vol = sum(l.volume_mm3 for l in group)
            result.append(AnatomyLabel(
                name=name,
                position=avg_pos,
                volume_mm3=total_vol,
                color=(0.95, 0.92, 0.80),
                layer_index=3,
                description=f"{len(group)} component(s), {total_vol / 1000:.1f} cm³",
            ))

        return result

    def generate_anomaly_context(
        self,
        finding: dict,
        tissue_results: list,
        volume_shape: tuple[int, int, int],
    ) -> str:
        """
        Generate anatomical context string for an anomaly finding.

        Examples:
        - "Pulmonary nodule in the right upper lobe"
        - "Effusion in left pleural recess"
        """
        location = finding.get("location", {})
        if not location:
            return f"Anomaly in {finding.get('region', 'unspecified region')}"

        x_frac = location.get("x", 0) / max(volume_shape[2], 1)
        y_frac = location.get("y", 0) / max(volume_shape[1], 1)
        z_frac = location.get("z", 0) / max(volume_shape[0], 1)

        # Determine laterality
        side = "right" if x_frac < 0.5 else "left"

        # Determine craniocaudal position
        if z_frac < 0.33:
            vertical = "upper"
        elif z_frac < 0.66:
            vertical = "middle"
        else:
            vertical = "lower"

        # Determine which tissue layer contains this point
        containing_tissue = "unspecified region"
        for tissue in tissue_results:
            if tissue.centroid_mm is None:
                continue
            # Simple proximity check
            dist = abs(z_frac - tissue.centroid_mm[0] / max(volume_shape[0], 1))
            if dist < 0.3 and tissue.name in ("left_lung", "right_lung"):
                containing_tissue = f"{side} lung {vertical} lobe"
                break
            elif tissue.name == "heart" and dist < 0.2:
                containing_tissue = "cardiac region"
                break

        severity = finding.get("severity", "")
        description = finding.get("description", "Anomaly")

        return f"{severity.capitalize()} finding in the {containing_tissue}"

    def _centroid_to_viewer_coords(
        self,
        centroid_mm: tuple[float, float, float],
        voxel_spacing: np.ndarray,
        volume_shape: tuple[int, int, int],
    ) -> tuple[float, float, float]:
        """Convert voxel-space centroid to viewer world coordinates."""
        # Normalize to centered coordinates matching the mesh coordinate system
        center_mm = np.array(volume_shape) * voxel_spacing / 2.0
        max_extent = float(np.max(np.array(volume_shape) * voxel_spacing))

        if max_extent < 1.0:
            max_extent = 1.0

        norm = lambda v, c: (v - c) / max_extent * 100.0

        return (
            norm(centroid_mm[2], center_mm[2]),  # x
            norm(centroid_mm[1], center_mm[1]),  # y  
            norm(centroid_mm[0], center_mm[0]),  # z
        )

    def _format_tissue_name(self, name: str) -> str:
        """Format internal tissue name to display name."""
        name_map = {
            "skin": "Skin",
            "bone": "Skeletal Structure",
            "left_lung": "Left Lung",
            "right_lung": "Right Lung",
            "heart": "Heart",
            "vessels": "Vasculature",
            "soft_tissue": "Soft Tissue",
            "pathology": "Pathology",
            "brain": "Brain",
            "liver": "Liver",
            "kidneys": "Kidneys",
        }
        return name_map.get(name, name.replace("_", " ").title())

    def _generate_description(self, tissue) -> str:
        """Generate descriptive text for a tissue label."""
        vol_cm3 = tissue.volume_mm3 / 1000.0
        return (
            f"{vol_cm3:.1f} cm³ · "
            f"Mean {tissue.mean_hu:.0f} HU · "
            f"{tissue.vertex_count} vertices"
        )

    def _classify_bone_component(
        self,
        z_frac: float,
        y_frac: float,
        x_frac: float,
        size: int,
        total_voxels: int,
    ) -> str:
        """Classify a bone connected component by spatial position."""
        size_ratio = size / max(total_voxels, 1)

        # Check against known spatial patterns
        for struct_name, ranges in self._BONE_STRUCTURES.items():
            z_ok = ranges["z_range"][0] <= z_frac <= ranges["z_range"][1]
            x_ok = ranges["x_range"][0] <= x_frac <= ranges["x_range"][1]
            y_ok = ranges["y_range"][0] <= y_frac <= ranges["y_range"][1]

            if z_ok and x_ok and y_ok:
                return struct_name

        # Generic classification
        if y_frac > 0.6 and 0.35 < x_frac < 0.65:
            return "Vertebra"
        elif z_frac < 0.15 and y_frac < 0.4:
            return "Clavicle"
        elif size_ratio > 0.001:
            return "Rib"
        else:
            return "Bone Fragment"

    def _create_anomaly_label(
        self,
        finding: dict,
        voxel_spacing: np.ndarray,
        volume_shape: tuple[int, int, int],
    ) -> Optional[AnatomyLabel]:
        """Create a viewer label for an anomaly finding."""
        location = finding.get("location")
        if not location:
            return None

        # Convert from volume coords to viewer coords
        centroid_mm = (
            location.get("z", 0) * voxel_spacing[0],
            location.get("y", 0) * voxel_spacing[1],
            location.get("x", 0) * voxel_spacing[2],
        )
        position = self._centroid_to_viewer_coords(
            centroid_mm, voxel_spacing, volume_shape
        )

        severity = finding.get("severity", "unknown")
        severity_colors = {
            "normal": (0.06, 0.72, 0.51),
            "mild": (0.96, 0.62, 0.04),
            "moderate": (0.98, 0.45, 0.09),
            "severe": (0.94, 0.27, 0.27),
            "critical": (0.86, 0.15, 0.15),
        }

        return AnatomyLabel(
            name=f"⚠ {finding.get('region', 'Finding')}",
            position=position,
            volume_mm3=0.0,
            color=severity_colors.get(severity, (1.0, 0.5, 0.0)),
            layer_index=11,
            description=finding.get("description", "Anomaly detected"),
        )
