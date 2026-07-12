"""
MedSpatial AI — Application Configuration
Centralized settings management via pydantic-settings.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables / .env file."""

    # ── App Meta ──────────────────────────────────────────────
    APP_NAME: str = "MedSpatial AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # ── Server ────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # ── Database ──────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./medspatial.db"

    # ── File Storage ──────────────────────────────────────────
    UPLOAD_DIR: str = "./data/uploads"
    VOLUME_DIR: str = "./data/volumes"
    MESH_DIR: str = "./data/meshes"
    ANALYSIS_DIR: str = "./data/analysis"
    REPORTS_DIR: str = "./data/reports"
    XAI_DIR: str = "./data/analysis/xai"
    ATLAS_DIR: str = "./data/atlas"

    # ── AI Model Paths ────────────────────────────────────────
    MODEL_DIR: str = "./models"
    SPATIAL_TRANSFORMER_WEIGHTS: Optional[str] = None
    SEGMENTATION_WEIGHTS: Optional[str] = None
    ANOMALY_DETECTOR_WEIGHTS: Optional[str] = None

    # ── AI Model Hyperparams ──────────────────────────────────
    VOLUME_SIZE: int = 128  # resize volumes to NxNxN for model input
    PATCH_SIZE: int = 16
    EMBED_DIM: int = 512
    NUM_HEADS: int = 8
    NUM_LAYERS: int = 6
    ANOMALY_THRESHOLD: float = 0.65

    # ── Reconstruction ────────────────────────────────────────
    MARCHING_CUBES_STEP_SIZE: int = 2
    MESH_SIMPLIFY_RATIO: float = 0.3

    # ── CORS ──────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def ensure_directories(self) -> None:
        """Create all required data directories on startup."""
        for dir_path in [
            self.UPLOAD_DIR,
            self.VOLUME_DIR,
            self.MESH_DIR,
            self.ANALYSIS_DIR,
            self.REPORTS_DIR,
            self.XAI_DIR,
            self.ATLAS_DIR,
            self.MODEL_DIR,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


settings = Settings()
