"""
MedSpatial AI — Production Configuration
=========================================
Replaces backend/app/config.py at runtime when running as a packaged
desktop application.

Key differences from dev config:
  - All paths are relative to APPDATA (portable, per-user, no admin needed)
  - HOST is always 127.0.0.1 — never exposed to the network
  - PORT is read from MEDSPATIAL_PORT env var (set by Tauri at launch)
  - DEBUG is always False
  - CORS allows only tauri://localhost and 127.0.0.1
  - Model paths resolve to the bundled onnx_models/ directory next to the exe

Path strategy:
  %APPDATA%\MedSpatialAI\
    ├── data\
    │   ├── uploads\
    │   ├── volumes\
    │   ├── meshes\
    │   ├── analysis\
    │   ├── reports\
    │   └── atlas\
    ├── models\
    │   └── onnx\           ← ONNX model files (bundled by installer)
    ├── db\
    │   └── medspatial.db
    └── logs\
        └── backend.log

The installer copies onnx_models/*.onnx into the `models\onnx` dir so
the paths here are always valid on a clean install.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Resolve portable base directory
# ---------------------------------------------------------------------------

def _appdata_dir() -> Path:
    """
    Returns %APPDATA%\MedSpatialAI on Windows.
    Falls back to ~/.medspatialai on Linux (Codespaces build runner).
    """
    win_appdata = os.environ.get("APPDATA")
    if win_appdata:
        return Path(win_appdata) / "MedSpatialAI"
    return Path.home() / ".medspatialai"


def _exe_dir() -> Path:
    """
    Directory of the running executable (or script during dev).
    ONNX models are bundled relative to this directory by the installer.
    """
    # MEDSPATIAL_EXE_DIR is injected by Tauri's sidecar launcher
    env_dir = os.environ.get("MEDSPATIAL_EXE_DIR")
    if env_dir:
        return Path(env_dir)
    # Fallback: parent of this file → desktop/backend_prod/ → desktop/ → medspatial-ai/
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Production Settings  (plain dataclass — no pydantic-settings dependency)
# ---------------------------------------------------------------------------

class ProductionSettings:
    # ── App meta ──────────────────────────────────────────────────────
    APP_NAME:    str = "MedSpatial AI"
    APP_VERSION: str = "1.0.0"
    DEBUG:       bool = False

    # ── Server ────────────────────────────────────────────────────────
    HOST:    str = "127.0.0.1"                          # localhost ONLY
    PORT:    int = int(os.environ.get("MEDSPATIAL_PORT", "8765"))
    WORKERS: int = 1                                     # single worker, async

    # ── Security ──────────────────────────────────────────────────────
    # 32-byte hex token generated fresh at each process start.
    # Tauri reads this via the MEDSPATIAL_TOKEN env var it sets itself,
    # and injects it into every API request as X-MedSpatial-Token header.
    API_TOKEN: str = os.environ.get(
        "MEDSPATIAL_TOKEN",
        secrets.token_hex(32),    # safe default for dev runs without Tauri
    )

    # ── CORS ──────────────────────────────────────────────────────────
    # Tauri WebView2 uses tauri://localhost as the origin in production builds.
    CORS_ORIGINS: list[str] = [
        "tauri://localhost",
        f"http://127.0.0.1:{int(os.environ.get('MEDSPATIAL_PORT', '8765'))}",
        "http://localhost:5173",   # kept for Codespaces / dev convenience
    ]

    def __init__(self) -> None:
        base = _appdata_dir()
        exe  = _exe_dir()

        # ── File storage (per-user APPDATA) ───────────────────────────
        self.DATA_DIR:     Path = base / "data"
        self.UPLOAD_DIR:   Path = base / "data" / "uploads"
        self.VOLUME_DIR:   Path = base / "data" / "volumes"
        self.MESH_DIR:     Path = base / "data" / "meshes"
        self.ANALYSIS_DIR: Path = base / "data" / "analysis"
        self.REPORTS_DIR:  Path = base / "data" / "reports"
        self.XAI_DIR:      Path = base / "data" / "analysis" / "xai"
        self.ATLAS_DIR:    Path = base / "data" / "atlas"
        self.LOG_DIR:      Path = base / "logs"

        # ── Database ──────────────────────────────────────────────────
        db_path = base / "db" / "medspatial.db"
        self.DATABASE_URL: str = f"sqlite+aiosqlite:///{db_path}"

        # ── ONNX models (bundled next to the exe by installer) ────────
        # Installer copies onnx_models/ into  <exe_dir>\resources\onnx_models\
        bundled_onnx = exe / "resources" / "onnx_models"
        if not bundled_onnx.exists():
            # Dev fallback: look in desktop/onnx_models/
            bundled_onnx = exe / "onnx_models"
        self.ONNX_DIR: Path = bundled_onnx

        # Atlas data (bundled)
        bundled_atlas = exe / "resources" / "data"
        self.MODEL_DIR: str = str(bundled_onnx)
        self.ATLAS_BUNDLED_DIR: Path = bundled_atlas

        # ── AI model hyperparams (must match export settings) ─────────
        self.VOLUME_SIZE:   int   = 128
        self.PATCH_SIZE:    int   = 16
        self.EMBED_DIM:     int   = 512
        self.NUM_HEADS:     int   = 8
        self.NUM_LAYERS:    int   = 6
        self.ANOMALY_THRESHOLD: float = 0.65

        # ── Reconstruction ────────────────────────────────────────────
        self.MARCHING_CUBES_STEP_SIZE: int   = 2
        self.MESH_SIMPLIFY_RATIO:      float = 0.3

        # Keep optional PyTorch weight paths as None — they are unused
        # in production (ONNX engine is used instead)
        self.SPATIAL_TRANSFORMER_WEIGHTS: Optional[str] = None
        self.SEGMENTATION_WEIGHTS:        Optional[str] = None
        self.ANOMALY_DETECTOR_WEIGHTS:    Optional[str] = None

    def ensure_directories(self) -> None:
        """Create all runtime data directories. Called from FastAPI lifespan."""
        dirs = [
            self.UPLOAD_DIR,
            self.VOLUME_DIR,
            self.MESH_DIR,
            self.ANALYSIS_DIR,
            self.REPORTS_DIR,
            self.XAI_DIR,
            self.ATLAS_DIR,
            self.LOG_DIR,
            Path(self.DATABASE_URL.replace("sqlite+aiosqlite:///", "")).parent,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # str() / repr() — never print the token in logs
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"ProductionSettings("
            f"host={self.HOST}:{self.PORT}, "
            f"debug={self.DEBUG}, "
            f"onnx_dir={self.ONNX_DIR})"
        )


# Module-level singleton — imported as `from config_prod import prod_settings`
prod_settings = ProductionSettings()
