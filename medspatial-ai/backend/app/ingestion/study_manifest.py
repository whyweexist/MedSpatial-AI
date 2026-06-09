"""Build an AWM study manifest from ingestion results."""

from pathlib import Path
from typing import Any

from app.awm.schema import BodyRegion, Modality, StudyManifest
from app.ingestion.modality_router import ModalityRouter
from app.ingestion.quality_gate import QualityGate


def create_study_manifest(
    study_id: str,
    paths: list[Path],
    metadata: dict[str, Any],
    deidentified_study_id: str | None = None,
) -> StudyManifest:
    modality = ModalityRouter().detect(paths, metadata)
    quality = QualityGate().evaluate(metadata, len(paths))
    body_value = str(metadata.get("body_part", "unknown")).lower()
    region = next((item for item in BodyRegion if item.value in body_value), BodyRegion.UNKNOWN)
    suffixes = {path.suffix.lower() for path in paths}
    source_format = (
        "nifti" if suffixes & {".nii", ".gz"}
        else "image" if suffixes & {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        else "dicom"
    )
    return StudyManifest(
        study_id=study_id,
        deidentified_study_id=deidentified_study_id or study_id,
        modality=modality,
        body_regions=[region],
        source_format=source_format,
        file_count=len(paths),
        quality_score=quality.score,
        warnings=quality.warnings,
    )
