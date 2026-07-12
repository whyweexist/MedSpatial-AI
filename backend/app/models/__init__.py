"""
MedSpatial AI — Database Models
SQLAlchemy ORM models for persisting scan metadata, volumes, analyses, and chat sessions.
"""

import datetime
import enum
import uuid
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


class ScanStatus(str, enum.Enum):
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    RECONSTRUCTED = "reconstructed"
    ANALYZED = "analyzed"
    FAILED = "failed"


class Scan(Base):
    """Represents an uploaded DICOM scan (one or more DICOM files forming a series)."""

    __tablename__ = "scans"

    id: str = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Optional[str] = Column(String(128), nullable=True, index=True)
    patient_name: Optional[str] = Column(String(256), nullable=True)
    study_description: Optional[str] = Column(String(512), nullable=True)
    series_description: Optional[str] = Column(String(512), nullable=True)
    modality: Optional[str] = Column(String(16), nullable=True)  # CT, XR, MR
    body_part: Optional[str] = Column(String(128), nullable=True)
    body_region: Optional[str] = Column(String(64), nullable=True)  # head, chest, abdomen, etc.
    region_confidence: Optional[float] = Column(Float, nullable=True)

    num_slices: int = Column(Integer, default=0)
    slice_thickness: Optional[float] = Column(Float, nullable=True)
    pixel_spacing_x: Optional[float] = Column(Float, nullable=True)
    pixel_spacing_y: Optional[float] = Column(Float, nullable=True)
    rows: Optional[int] = Column(Integer, nullable=True)
    columns: Optional[int] = Column(Integer, nullable=True)

    upload_path: str = Column(String(1024), nullable=False)
    status: ScanStatus = Column(Enum(ScanStatus), default=ScanStatus.UPLOADING)
    metadata_json: Optional[dict] = Column(JSON, nullable=True)

    created_at: datetime.datetime = Column(
        DateTime, default=datetime.datetime.utcnow
    )
    updated_at: datetime.datetime = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    # Relationships
    volume = relationship("Volume", back_populates="scan", uselist=False, cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="scan", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="scan", cascade="all, delete-orphan")


class Volume(Base):
    """Reconstructed 3D volume and generated meshes from a scan."""

    __tablename__ = "volumes"

    id: str = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    scan_id: str = Column(String(36), ForeignKey("scans.id"), nullable=False, unique=True)

    volume_path: str = Column(String(1024), nullable=False)  # .npy file
    mesh_path: Optional[str] = Column(String(1024), nullable=True)  # .glb file
    dimensions: Optional[dict] = Column(JSON, nullable=True)  # {"x":N, "y":N, "z":N}
    voxel_spacing: Optional[dict] = Column(JSON, nullable=True)

    # Layer meshes — stored as JSON: {tissue_name: {mesh_path, vertex_count, ...}}
    layer_mesh_paths: Optional[dict] = Column(JSON, nullable=True)

    # Legacy columns for backwards compat
    bone_mesh_path: Optional[str] = Column(String(1024), nullable=True)
    soft_tissue_mesh_path: Optional[str] = Column(String(1024), nullable=True)
    air_mesh_path: Optional[str] = Column(String(1024), nullable=True)
    vessel_mesh_path: Optional[str] = Column(String(1024), nullable=True)

    hu_min: Optional[float] = Column(Float, nullable=True)
    hu_max: Optional[float] = Column(Float, nullable=True)

    # Reconstruction summary
    reconstruction_summary: Optional[dict] = Column(JSON, nullable=True)

    created_at: datetime.datetime = Column(DateTime, default=datetime.datetime.utcnow)

    scan = relationship("Scan", back_populates="volume")


class Analysis(Base):
    """AI analysis results for a scan — anomalies, segmentation masks, etc."""

    __tablename__ = "analyses"

    id: str = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    scan_id: str = Column(String(36), ForeignKey("scans.id"), nullable=False)

    analysis_type: str = Column(String(64), nullable=False)  # anomaly, segmentation, layer
    status: str = Column(String(32), default="pending")  # pending, running, completed, failed

    # Results
    heatmap_path: Optional[str] = Column(String(1024), nullable=True)
    segmentation_mask_path: Optional[str] = Column(String(1024), nullable=True)
    findings: Optional[dict] = Column(JSON, nullable=True)
    confidence: Optional[float] = Column(Float, nullable=True)
    summary: Optional[str] = Column(Text, nullable=True)

    # XAI outputs
    xai_heatmap_path: Optional[str] = Column(String(1024), nullable=True)
    reasoning_json: Optional[dict] = Column(JSON, nullable=True)

    created_at: datetime.datetime = Column(DateTime, default=datetime.datetime.utcnow)

    scan = relationship("Scan", back_populates="analyses")


class ChatSession(Base):
    """Conversational Q&A session tied to a specific scan."""

    __tablename__ = "chat_sessions"

    id: str = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    scan_id: str = Column(String(36), ForeignKey("scans.id"), nullable=False)

    messages: list = Column(JSON, default=list)  # [{role, content, timestamp}, ...]
    context_summary: Optional[str] = Column(Text, nullable=True)

    created_at: datetime.datetime = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    scan = relationship("Scan", back_populates="chat_sessions")
