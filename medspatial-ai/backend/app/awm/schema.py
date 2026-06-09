"""Typed source-of-truth schema for the AWM-JGEM architecture."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Modality(str, Enum):
    CT = "CT"
    MR = "MR"
    XR = "XR"
    DX = "DX"
    CR = "CR"
    NIFTI = "NIFTI"
    US = "US"
    PT = "PT"
    NM = "NM"
    SYNTHETIC = "SYNTHETIC"
    UNKNOWN = "UNKNOWN"


class GroundingType(str, Enum):
    SCAN_DERIVED = "scan_derived"
    ESTIMATED = "estimated"
    SYNTHETIC = "synthetic"
    USER_AUTHORED = "user_authored"


class BodyRegion(str, Enum):
    HEAD = "head"
    BRAIN = "brain"
    FACE = "face"
    NECK = "neck"
    CHEST = "chest"
    HEART = "heart"
    ABDOMEN = "abdomen"
    PELVIS = "pelvis"
    SPINE = "spine"
    SHOULDER = "shoulder"
    ARM = "arm"
    ELBOW = "elbow"
    FOREARM = "forearm"
    WRIST = "wrist"
    HAND = "hand"
    HIP = "hip"
    THIGH = "thigh"
    KNEE = "knee"
    LEG = "leg"
    ANKLE = "ankle"
    FOOT = "foot"
    WHOLE_BODY = "whole_body"
    UNKNOWN = "unknown"


class AnatomicalSystem(str, Enum):
    NERVOUS = "nervous"
    CARDIOVASCULAR = "cardiovascular"
    RESPIRATORY = "respiratory"
    DIGESTIVE = "digestive"
    URINARY = "urinary"
    REPRODUCTIVE = "reproductive"
    MUSCULOSKELETAL = "musculoskeletal"
    ENDOCRINE = "endocrine"
    LYMPHATIC = "lymphatic"
    INTEGUMENTARY = "integumentary"
    SENSORY = "sensory"
    OTHER = "other"


class SpatialPoint(BaseModel):
    x: float
    y: float
    z: float
    coordinate_system: str = "voxel"


class BoundingBox3D(BaseModel):
    minimum: SpatialPoint
    maximum: SpatialPoint


class ModelIdentity(BaseModel):
    name: str
    version: str
    provider: str = "local"
    weights_digest: str | None = None
    calibrated: bool = False


class EvidenceAnchor(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    evidence_type: Literal[
        "source_slice",
        "segmentation_mask",
        "heatmap",
        "measurement",
        "anatomical_node",
        "uncertainty_map",
        "provenance_record",
        "model_output",
        "user_annotation",
    ]
    reference_id: str
    label: str
    slice_index: int | None = None
    frame_of_reference_uid: str | None = None
    location: SpatialPoint | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    grounding: GroundingType = GroundingType.SCAN_DERIVED


class ProvenanceRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    operation: str
    source_ids: list[str] = Field(default_factory=list)
    output_ids: list[str] = Field(default_factory=list)
    actor_id: str = "system"
    model: ModelIdentity | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)
    grounding: GroundingType = GroundingType.SCAN_DERIVED


class VolumeReference(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    uri: str
    shape_zyx: tuple[int, int, int]
    voxel_spacing_zyx_mm: tuple[float, float, float]
    dtype: str = "float32"
    memory_mapped: bool = True
    grounding: GroundingType = GroundingType.SCAN_DERIVED


class ImageReference(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    uri: str
    width: int
    height: int
    frame_index: int | None = None
    grounding: GroundingType = GroundingType.SCAN_DERIVED


class SegmentationReference(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    structure_id: str
    mask_uri: str
    provider: str
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty_uri: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    grounding: GroundingType = GroundingType.SCAN_DERIVED


class MeshReference(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    structure_id: str
    uri: str
    lod: int = Field(default=0, ge=0)
    vertex_count: int = Field(default=0, ge=0)
    face_count: int = Field(default=0, ge=0)
    source_segmentation_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    grounding: GroundingType = GroundingType.SCAN_DERIVED


class Measurement(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    value: float
    unit: str
    structure_id: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)


class AnatomicalStructure(BaseModel):
    id: str
    name: str
    canonical_name: str
    body_region: BodyRegion
    system: AnatomicalSystem
    laterality: Literal["left", "right", "midline", "bilateral", "not_applicable"] = "not_applicable"
    parent_id: str | None = None
    bounds: BoundingBox3D | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: float = Field(default=1.0, ge=0.0, le=1.0)
    grounding: GroundingType = GroundingType.SCAN_DERIVED
    evidence_ids: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class FindingCandidate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    label: str
    description: str
    structure_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    energy_score: float | None = None
    status: Literal["candidate", "reviewed", "rejected"] = "candidate"
    evidence_ids: list[str] = Field(min_length=1)
    model: ModelIdentity
    grounding: GroundingType = GroundingType.SCAN_DERIVED


class SceneNode(BaseModel):
    id: str
    node_type: Literal[
        "structure", "lesion", "finding", "measurement", "uncertainty",
        "atlas_structure", "annotation"
    ]
    label: str
    feature_vector: list[float] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class SceneEdge(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    target_id: str
    relation: Literal[
        "contains", "adjacent_to", "connected_to", "overlaps", "derived_from",
        "suspicious_for", "measured_by", "visible_in_slice",
        "supported_by_heatmap", "estimated_from", "synthetic_from",
        "conflicts_with"
    ]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)


class SceneGraph(BaseModel):
    nodes: list[SceneNode] = Field(default_factory=list)
    edges: list[SceneEdge] = Field(default_factory=list)
    model: ModelIdentity | None = None


class LatentState(BaseModel):
    token_uri: str | None = None
    embedding_dim: int = 0
    slot_embeddings: dict[str, list[float]] = Field(default_factory=dict)
    continuity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    residual_score: float = Field(default=0.0, ge=0.0)
    uncertainty: float = Field(default=1.0, ge=0.0, le=1.0)
    model: ModelIdentity | None = None


class ViewState(BaseModel):
    plane: Literal["axial", "coronal", "sagittal", "3d"] = "3d"
    slice_index: int | None = None
    selected_structure_id: str | None = None
    selected_finding_id: str | None = None
    visible_layers: list[str] = Field(default_factory=list)
    camera: dict[str, float] = Field(default_factory=dict)


class ProcessingEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    status: Literal["pending", "running", "completed", "failed", "blocked"]
    message: str
    trace_id: str
    timestamp: datetime = Field(default_factory=utc_now)
    details: dict[str, Any] = Field(default_factory=dict)


class StudyManifest(BaseModel):
    study_id: str
    deidentified_study_id: str
    modality: Modality
    body_regions: list[BodyRegion] = Field(default_factory=lambda: [BodyRegion.UNKNOWN])
    source_format: Literal["dicom", "nifti", "image", "synthetic"]
    file_count: int = Field(default=0, ge=0)
    data_classification: Literal["phi", "deidentified", "synthetic"] = "deidentified"
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class AnatomicalWorldModel(BaseModel):
    schema_version: str = "1.0.0"
    architecture: Literal["AWM-JGEM"] = "AWM-JGEM"
    study: StudyManifest
    volumes: list[VolumeReference] = Field(default_factory=list)
    images: list[ImageReference] = Field(default_factory=list)
    segmentations: list[SegmentationReference] = Field(default_factory=list)
    meshes: list[MeshReference] = Field(default_factory=list)
    structures: list[AnatomicalStructure] = Field(default_factory=list)
    findings: list[FindingCandidate] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    uncertainty_maps: dict[str, str] = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    evidence: list[EvidenceAnchor] = Field(default_factory=list)
    scene_graph: SceneGraph = Field(default_factory=SceneGraph)
    latent_state: LatentState = Field(default_factory=LatentState)
    retrieval_index_refs: list[str] = Field(default_factory=list)
    model_versions: list[ModelIdentity] = Field(default_factory=list)
    processing_events: list[ProcessingEvent] = Field(default_factory=list)
    audit_events: list[dict[str, Any]] = Field(default_factory=list)
    current_view: ViewState = Field(default_factory=ViewState)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def enforce_grounding(self) -> "AnatomicalWorldModel":
        if self.study.modality in {Modality.XR, Modality.DX, Modality.CR}:
            invalid = [
                item.id for item in [*self.volumes, *self.meshes, *self.structures]
                if item.grounding == GroundingType.SCAN_DERIVED and self.volumes
            ]
            if invalid and any(volume.shape_zyx[0] > 1 for volume in self.volumes):
                raise ValueError(
                    "Single-view radiography 3D assets must be marked estimated, not scan-derived"
                )
        if self.study.source_format == "synthetic":
            for item in [*self.volumes, *self.meshes, *self.structures]:
                if item.grounding != GroundingType.SYNTHETIC:
                    raise ValueError("No-image atlas assets must be marked synthetic")
        return self
