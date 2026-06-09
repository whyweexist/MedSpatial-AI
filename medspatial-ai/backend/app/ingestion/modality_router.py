"""File and metadata based modality routing."""

from pathlib import Path
from typing import Any

from app.awm.schema import Modality


class ModalityRouter:
    _DICOM_MAP = {
        "CT": Modality.CT, "MR": Modality.MR, "MRI": Modality.MR,
        "XR": Modality.XR, "DX": Modality.DX, "CR": Modality.CR,
        "US": Modality.US, "PT": Modality.PT, "NM": Modality.NM,
    }

    def detect(self, paths: list[Path], metadata: dict[str, Any]) -> Modality:
        declared = str(metadata.get("modality", "")).upper()
        if declared in self._DICOM_MAP:
            return self._DICOM_MAP[declared]
        suffixes = {path.suffix.lower() for path in paths}
        if suffixes & {".nii", ".gz"}:
            return Modality.NIFTI
        if suffixes & {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            return Modality.XR
        return Modality.UNKNOWN
