"""DICOM metadata de-identification without silent source modification."""

from __future__ import annotations

import hashlib
from typing import Any


class DeidentificationService:
    PHI_FIELDS = {
        "patient_name", "patient_birth_date", "patient_address", "institution",
        "referring_physician", "accession_number",
    }

    def deidentify_metadata(self, metadata: dict[str, Any], salt: str) -> dict[str, Any]:
        clean = {key: value for key, value in metadata.items() if key not in self.PHI_FIELDS}
        patient_id = str(metadata.get("patient_id", "unknown"))
        clean["deidentified_patient_id"] = hashlib.sha256(
            f"{salt}:{patient_id}".encode("utf-8")
        ).hexdigest()[:20]
        clean.pop("patient_id", None)
        clean["deidentified"] = True
        return clean
