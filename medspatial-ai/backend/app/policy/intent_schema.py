"""Armor-IQ-style typed action proposals."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class Action(str, Enum):
    VIEW_STUDY = "ViewStudy"
    SEGMENT_ANATOMY = "SegmentAnatomy"
    GENERATE_MESH = "GenerateMesh"
    MEASURE_LESION = "MeasureLesion"
    ASK_CLINICAL_QUESTION = "AskClinicalQuestion"
    NAVIGATE_TO_FINDING = "NavigateToFinding"
    TOGGLE_LAYER = "ToggleLayer"
    CREATE_ANNOTATION = "CreateAnnotation"
    EXPORT_ANONYMIZED_MODEL = "ExportAnonymizedModel"
    EXPORT_RAW_STUDY = "ExportRawStudy"
    SHARE_SESSION = "ShareSession"
    DELETE_STUDY = "DeleteStudy"
    EXTERNAL_DATA_TRANSFER = "ExternalDataTransfer"


class Intent(BaseModel):
    intent_id: str = Field(default_factory=lambda: str(uuid4()))
    actor_id: str
    actor_role: str
    patient_id: str | None = None
    deidentified_study_id: str | None = None
    study_id: str
    modality: str
    requested_action: Action
    target_structure: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    sensitivity_level: str = "deidentified"
    output_scope: str = "session"
    network_destination: str | None = None
    provenance_requirement: bool = True
    risk_level: str = "low"
    current_view_scope: dict[str, Any] = Field(default_factory=dict)
    selected_finding_id: str | None = None
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_confirmed: bool = False

    @model_validator(mode="after")
    def require_study_identity(self) -> "Intent":
        if not self.patient_id and not self.deidentified_study_id:
            raise ValueError("patient_id or deidentified_study_id is required")
        return self
