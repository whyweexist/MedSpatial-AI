"""
MedSpatial AI — Body Region Classifier
Automatically identifies the body region from DICOM metadata or image-based
heuristics, and configures the downstream pipeline accordingly.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
from loguru import logger


class BodyRegion(str, Enum):
    """Supported body regions."""
    HEAD = "head"
    NECK = "neck"
    CHEST = "chest"
    ABDOMEN = "abdomen"
    PELVIS = "pelvis"
    SPINE = "spine"
    EXTREMITY = "extremity"
    WHOLE_BODY = "whole_body"
    UNKNOWN = "unknown"


@dataclass
class RegionDetectionResult:
    """Result of body region detection."""
    region: BodyRegion
    confidence: float  # 0.0 – 1.0
    method: str  # "dicom_metadata", "hu_histogram", "fallback"
    modality: str  # CT, XR, MR, etc.
    details: str = ""


# Mapping from DICOM BodyPartExamined values to BodyRegion
_BODY_PART_MAP: dict[str, BodyRegion] = {
    "HEAD": BodyRegion.HEAD,
    "BRAIN": BodyRegion.HEAD,
    "SKULL": BodyRegion.HEAD,
    "NECK": BodyRegion.NECK,
    "CSPINE": BodyRegion.NECK,
    "CHEST": BodyRegion.CHEST,
    "THORAX": BodyRegion.CHEST,
    "LUNG": BodyRegion.CHEST,
    "ABDOMEN": BodyRegion.ABDOMEN,
    "LIVER": BodyRegion.ABDOMEN,
    "KIDNEY": BodyRegion.ABDOMEN,
    "PELVIS": BodyRegion.PELVIS,
    "HIP": BodyRegion.PELVIS,
    "SPINE": BodyRegion.SPINE,
    "LSPINE": BodyRegion.SPINE,
    "TSPINE": BodyRegion.SPINE,
    "EXTREMITY": BodyRegion.EXTREMITY,
    "HAND": BodyRegion.EXTREMITY,
    "FOOT": BodyRegion.EXTREMITY,
    "KNEE": BodyRegion.EXTREMITY,
    "ELBOW": BodyRegion.EXTREMITY,
    "SHOULDER": BodyRegion.EXTREMITY,
    "ANKLE": BodyRegion.EXTREMITY,
    "WRIST": BodyRegion.EXTREMITY,
    "LEG": BodyRegion.EXTREMITY,
    "ARM": BodyRegion.EXTREMITY,
    "WHOLEBODY": BodyRegion.WHOLE_BODY,
    "WHOLE BODY": BodyRegion.WHOLE_BODY,
}

# Keywords in study/series descriptions that hint at body region
_DESCRIPTION_KEYWORDS: dict[str, BodyRegion] = {
    "head": BodyRegion.HEAD,
    "brain": BodyRegion.HEAD,
    "cranial": BodyRegion.HEAD,
    "skull": BodyRegion.HEAD,
    "sinus": BodyRegion.HEAD,
    "orbit": BodyRegion.HEAD,
    "neck": BodyRegion.NECK,
    "cervical": BodyRegion.NECK,
    "carotid": BodyRegion.NECK,
    "chest": BodyRegion.CHEST,
    "thorax": BodyRegion.CHEST,
    "lung": BodyRegion.CHEST,
    "pulmonary": BodyRegion.CHEST,
    "cardiac": BodyRegion.CHEST,
    "heart": BodyRegion.CHEST,
    "mediastin": BodyRegion.CHEST,
    "abdomen": BodyRegion.ABDOMEN,
    "abdominal": BodyRegion.ABDOMEN,
    "liver": BodyRegion.ABDOMEN,
    "kidney": BodyRegion.ABDOMEN,
    "renal": BodyRegion.ABDOMEN,
    "pancrea": BodyRegion.ABDOMEN,
    "spleen": BodyRegion.ABDOMEN,
    "bowel": BodyRegion.ABDOMEN,
    "pelvis": BodyRegion.PELVIS,
    "pelvic": BodyRegion.PELVIS,
    "hip": BodyRegion.PELVIS,
    "bladder": BodyRegion.PELVIS,
    "prostate": BodyRegion.PELVIS,
    "uterus": BodyRegion.PELVIS,
    "spine": BodyRegion.SPINE,
    "lumbar": BodyRegion.SPINE,
    "thoracic": BodyRegion.SPINE,
    "vertebr": BodyRegion.SPINE,
    "extremity": BodyRegion.EXTREMITY,
    "hand": BodyRegion.EXTREMITY,
    "foot": BodyRegion.EXTREMITY,
    "knee": BodyRegion.EXTREMITY,
    "ankle": BodyRegion.EXTREMITY,
    "elbow": BodyRegion.EXTREMITY,
    "wrist": BodyRegion.EXTREMITY,
    "shoulder": BodyRegion.EXTREMITY,
    "whole body": BodyRegion.WHOLE_BODY,
    "wholebody": BodyRegion.WHOLE_BODY,
}


class BodyRegionClassifier:
    """
    Detects body region from DICOM metadata and/or image-based heuristics.

    Detection priority:
      1. DICOM BodyPartExamined tag
      2. StudyDescription / SeriesDescription keyword matching
      3. HU histogram analysis (fallback for CT volumes)
      4. Default to CHEST
    """

    def classify_from_metadata(
        self, metadata: dict[str, Any]
    ) -> Optional[RegionDetectionResult]:
        """
        Attempt region detection from DICOM metadata fields.

        Args:
            metadata: dict with keys like body_part, study_description, etc.

        Returns:
            RegionDetectionResult if detected, None if no match found
        """
        modality = str(metadata.get("modality", "")).upper()

        # 1. Check BodyPartExamined
        body_part = str(metadata.get("body_part", "")).upper().strip()
        if body_part and body_part in _BODY_PART_MAP:
            region = _BODY_PART_MAP[body_part]
            logger.info(f"Body region from BodyPartExamined: {region.value}")
            return RegionDetectionResult(
                region=region,
                confidence=0.95,
                method="dicom_metadata",
                modality=modality,
                details=f"BodyPartExamined={body_part}",
            )

        # 2. Check study/series descriptions
        descriptions = " ".join([
            str(metadata.get("study_description", "")),
            str(metadata.get("series_description", "")),
        ]).lower()

        if descriptions.strip():
            for keyword, region in _DESCRIPTION_KEYWORDS.items():
                if keyword in descriptions:
                    logger.info(
                        f"Body region from description keyword '{keyword}': {region.value}"
                    )
                    return RegionDetectionResult(
                        region=region,
                        confidence=0.80,
                        method="dicom_metadata",
                        modality=modality,
                        details=f"Keyword '{keyword}' in description",
                    )

        return None

    def classify_from_volume(
        self, volume: np.ndarray
    ) -> RegionDetectionResult:
        """
        Classify body region from HU histogram analysis.
        Works for CT volumes only.

        Heuristics:
        - Head: very high bone fraction, compact round cross-section
        - Chest: significant air fraction (lungs), moderate bone (ribs)
        - Abdomen: low air, moderate soft tissue, some bone (spine)
        - Extremity: very high bone/air ratio, elongated cross-section

        Args:
            volume: 3D numpy array in Hounsfield Units

        Returns:
            RegionDetectionResult with heuristic-based detection
        """
        flat = volume.ravel()
        total = len(flat)

        if total == 0:
            return RegionDetectionResult(
                region=BodyRegion.CHEST,
                confidence=0.3,
                method="fallback",
                modality="CT",
                details="Empty volume, defaulting to CHEST",
            )

        # Compute tissue fractions
        air_fraction = float(np.sum(flat < -500)) / total
        lung_fraction = float(np.sum((flat >= -950) & (flat < -200))) / total
        soft_tissue_fraction = float(np.sum((flat >= -100) & (flat < 200))) / total
        bone_fraction = float(np.sum(flat >= 200)) / total

        # Cross-sectional shape analysis on middle slice
        mid_z = volume.shape[0] // 2
        mid_slice = volume[mid_z]
        body_pixels = mid_slice > -400
        body_count = body_pixels.sum()

        aspect_ratio = 1.0
        if body_count > 100:
            body_coords = np.argwhere(body_pixels)
            y_range = body_coords[:, 0].max() - body_coords[:, 0].min() + 1
            x_range = body_coords[:, 1].max() - body_coords[:, 1].min() + 1
            aspect_ratio = max(y_range, x_range) / max(min(y_range, x_range), 1)

        details = (
            f"air={air_fraction:.2f} lung={lung_fraction:.2f} "
            f"soft={soft_tissue_fraction:.2f} bone={bone_fraction:.2f} "
            f"aspect={aspect_ratio:.2f}"
        )

        # Decision tree
        if lung_fraction > 0.15:
            region = BodyRegion.CHEST
            confidence = min(0.5 + lung_fraction, 0.85)
        elif bone_fraction > 0.15 and aspect_ratio < 1.3 and air_fraction < 0.3:
            region = BodyRegion.HEAD
            confidence = 0.65
        elif soft_tissue_fraction > 0.5 and bone_fraction < 0.08:
            region = BodyRegion.ABDOMEN
            confidence = 0.55
        elif bone_fraction > 0.2 and aspect_ratio > 2.0:
            region = BodyRegion.EXTREMITY
            confidence = 0.55
        elif bone_fraction > 0.1 and aspect_ratio > 1.8:
            region = BodyRegion.SPINE
            confidence = 0.50
        else:
            region = BodyRegion.CHEST
            confidence = 0.40

        logger.info(f"Body region from HU histogram: {region.value} ({confidence:.0%})")
        return RegionDetectionResult(
            region=region,
            confidence=confidence,
            method="hu_histogram",
            modality="CT",
            details=details,
        )

    def classify(
        self,
        metadata: dict[str, Any],
        volume: Optional[np.ndarray] = None,
    ) -> RegionDetectionResult:
        """
        Full classification pipeline: metadata first, then volume-based fallback.

        Args:
            metadata: DICOM metadata dict
            volume: optional HU volume for histogram-based fallback

        Returns:
            RegionDetectionResult
        """
        # Try metadata first
        result = self.classify_from_metadata(metadata)
        if result is not None:
            return result

        # Try volume-based heuristics
        if volume is not None:
            return self.classify_from_volume(volume)

        # Ultimate fallback
        modality = str(metadata.get("modality", "")).upper()
        logger.warning("Could not determine body region, defaulting to CHEST")
        return RegionDetectionResult(
            region=BodyRegion.CHEST,
            confidence=0.3,
            method="fallback",
            modality=modality or "UNKNOWN",
            details="No metadata or volume available for classification",
        )
