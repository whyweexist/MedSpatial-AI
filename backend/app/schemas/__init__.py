"""
MedSpatial AI — Pydantic Schemas
Request / response models for the API layer.
"""

import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
#  Scan Schemas
# ──────────────────────────────────────────────────────────────

class ScanUploadResponse(BaseModel):
    scan_id: str
    status: str
    message: str
    num_slices: int = 0


class ScanMetadata(BaseModel):
    id: str
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    study_description: Optional[str] = None
    series_description: Optional[str] = None
    modality: Optional[str] = None
    body_part: Optional[str] = None
    num_slices: int = 0
    slice_thickness: Optional[float] = None
    pixel_spacing_x: Optional[float] = None
    pixel_spacing_y: Optional[float] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    status: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class ScanListResponse(BaseModel):
    scans: list[ScanMetadata]
    total: int


# ──────────────────────────────────────────────────────────────
#  Volume / Reconstruction Schemas
# ──────────────────────────────────────────────────────────────

class ReconstructionRequest(BaseModel):
    scan_id: str
    iso_level: Optional[float] = None
    step_size: Optional[int] = None
    generate_layers: bool = True


class ReconstructionResponse(BaseModel):
    scan_id: str
    volume_id: str
    status: str
    mesh_url: Optional[str] = None
    layer_urls: Optional[dict[str, str]] = None
    dimensions: Optional[dict[str, int]] = None


class SliceRequest(BaseModel):
    scan_id: str
    axis: str = Field("axial", pattern="^(axial|coronal|sagittal)$")
    index: int = 0


class SliceResponse(BaseModel):
    image_data: str  # base64 encoded PNG
    axis: str
    index: int
    total_slices: int
    window_center: float = 0.0
    window_width: float = 1.0


# ──────────────────────────────────────────────────────────────
#  Analysis Schemas
# ──────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    scan_id: str
    analysis_type: str = Field("full", pattern="^(anomaly|segmentation|layer|full)$")


class AnomalyFinding(BaseModel):
    region: str
    description: str
    confidence: float
    location: Optional[dict[str, float]] = None  # {x, y, z} in volume coords
    severity: str = "unknown"  # mild, moderate, severe, critical


class AnalysisResponse(BaseModel):
    analysis_id: str
    scan_id: str
    status: str
    analysis_type: str
    findings: Optional[list[AnomalyFinding]] = None
    heatmap_url: Optional[str] = None
    segmentation_url: Optional[str] = None
    summary: Optional[str] = None
    confidence: Optional[float] = None


# ──────────────────────────────────────────────────────────────
#  Chat / Q&A Schemas
# ──────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    timestamp: Optional[datetime.datetime] = None


class ChatRequest(BaseModel):
    scan_id: str
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    referenced_slices: Optional[list[int]] = None
    referenced_regions: Optional[list[dict[str, Any]]] = None
    findings_mentioned: Optional[list[AnomalyFinding]] = None


# ──────────────────────────────────────────────────────────────
#  WebSocket Schemas
# ──────────────────────────────────────────────────────────────

class ProcessingUpdate(BaseModel):
    scan_id: str
    stage: str  # upload, reconstruction, analysis, complete
    progress: float  # 0.0 - 1.0
    message: str
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────
#  Body Region Schemas
# ──────────────────────────────────────────────────────────────

class BodyRegionInfo(BaseModel):
    region: str
    confidence: float
    method: str
    modality: str
    details: str = ""
    display_name: str = ""
    icon: str = "📦"


# ──────────────────────────────────────────────────────────────
#  Tissue Layer / Segment Schemas
# ──────────────────────────────────────────────────────────────

class TissueLayerInfo(BaseModel):
    name: str
    label_index: int
    mesh_url: Optional[str] = None
    vertex_count: int = 0
    face_count: int = 0
    volume_mm3: float = 0.0
    volume_cm3: float = 0.0
    color_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)
    opacity: float = 0.8
    centroid: Optional[dict[str, float]] = None
    mean_hu: float = 0.0
    voxel_count: int = 0
    description: str = ""


class ReconstructionSummary(BaseModel):
    scan_id: str
    body_region: Optional[BodyRegionInfo] = None
    tissues: list[TissueLayerInfo] = []
    total_mesh_vertices: int = 0
    total_mesh_faces: int = 0
    processing_time_s: float = 0.0


class SegmentInfo(BaseModel):
    """Full segment data for the dissection module."""
    name: str
    label_index: int
    mesh_url: Optional[str] = None
    visible: bool = True
    opacity: float = 0.8
    color: str = "#808080"
    color_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)
    volume_cm3: float = 0.0
    mean_hu: float = 0.0
    voxel_count: int = 0
    centroid: Optional[dict[str, float]] = None
    description: str = ""
    dissection_order: int = 0


class SegmentsResponse(BaseModel):
    scan_id: str
    segments: list[SegmentInfo] = []
    body_region: Optional[BodyRegionInfo] = None


# ──────────────────────────────────────────────────────────────
#  Anatomy Label Schemas
# ──────────────────────────────────────────────────────────────

class AnatomyLabelSchema(BaseModel):
    name: str
    position: dict[str, float]  # {x, y, z} in viewer coords
    volume_mm3: float = 0.0
    color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    layer_index: int = 0
    description: str = ""


class AnatomyLabelsResponse(BaseModel):
    scan_id: str
    labels: list[AnatomyLabelSchema] = []


# ──────────────────────────────────────────────────────────────
#  XAI / Explainability Schemas
# ──────────────────────────────────────────────────────────────

class ReasoningStepSchema(BaseModel):
    category: str
    description: str
    confidence: float
    evidence_type: str


class ReasoningChainSchema(BaseModel):
    finding: str
    confidence: float
    steps: list[ReasoningStepSchema] = []
    anatomical_context: str = ""
    differential: list[str] = []
    bbox_3d: Optional[dict[str, float]] = None
    representative_slice_idx: Optional[int] = None


class ExplainResponse(BaseModel):
    scan_id: str
    status: str
    heatmaps: dict[str, str] = {}  # disease_class → URL
    reasoning_chains: list[ReasoningChainSchema] = []
    reasoning_url: Optional[str] = None


# ──────────────────────────────────────────────────────────────
#  Report Schemas
# ──────────────────────────────────────────────────────────────

class ReportRequest(BaseModel):
    scan_id: str
    format: str = Field("pdf", pattern="^(pdf|docx)$")
