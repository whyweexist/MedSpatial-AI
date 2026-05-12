"""
MedSpatial AI — FastAPI Application Entry Point
Mounts all routers, configures CORS, WebSocket, and startup/shutdown hooks.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import settings
from app.models.database import close_db, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    settings.ensure_directories()
    await init_db()
    logger.info("Database initialized")
    
    # Pre-load AI models to avoid first-request delay
    try:
        from app.services.anomaly_service import anomaly_svc
        logger.info("Pre-loading AI models...")
        # Force model loading by calling _load_models
        anomaly_svc._load_models()
        logger.info("AI models loaded successfully")
    except Exception as e:
        logger.warning(f"Model pre-loading failed: {e}")
    
    yield
    await close_db()
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Medical imaging platform: DICOM → interactive 3D volumetric models with AI analysis.",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (serve meshes, etc.) ────────────────────────
# Will be created on first run via ensure_directories
import os
for dir_name, dir_path in [("meshes", settings.MESH_DIR), ("analysis", settings.ANALYSIS_DIR)]:
    os.makedirs(dir_path, exist_ok=True)
    app.mount(f"/static/{dir_name}", StaticFiles(directory=dir_path), name=dir_name)

# ── Register API Routers ─────────────────────────────────────
from app.api.upload import router as upload_router
from app.api.reconstruction import router as recon_router
from app.api.analysis import router as analysis_router
from app.api.chat import router as chat_router
from app.api.explain import router as explain_router
from app.api.reports import router as reports_router

app.include_router(upload_router)
app.include_router(recon_router)
app.include_router(analysis_router)
app.include_router(chat_router)
app.include_router(explain_router)
app.include_router(reports_router)


# ── WebSocket for real-time processing updates ───────────────
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, scan_id: str):
        await websocket.accept()
        if scan_id not in self.active_connections:
            self.active_connections[scan_id] = []
        self.active_connections[scan_id].append(websocket)
        logger.info(f"WebSocket connected for scan {scan_id}")

    def disconnect(self, websocket: WebSocket, scan_id: str):
        if scan_id in self.active_connections:
            self.active_connections[scan_id].remove(websocket)
            if not self.active_connections[scan_id]:
                del self.active_connections[scan_id]
        logger.info(f"WebSocket disconnected for scan {scan_id}")

    async def broadcast(self, scan_id: str, message: dict):
        if scan_id in self.active_connections:
            for connection in self.active_connections[scan_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


ws_manager = ConnectionManager()


@app.websocket("/ws/{scan_id}")
async def websocket_endpoint(websocket: WebSocket, scan_id: str):
    """WebSocket endpoint for real-time processing status updates."""
    await ws_manager.connect(websocket, scan_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back or handle ping
            await websocket.send_json({"type": "pong", "scan_id": scan_id})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, scan_id)


# ── Health Check ──────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


# Make ws_manager importable by services
app.state.ws_manager = ws_manager
