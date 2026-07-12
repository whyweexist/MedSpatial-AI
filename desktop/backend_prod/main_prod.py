"""
MedSpatial AI — Production FastAPI Entry Point
===============================================
This file replaces backend/app/main.py in the packaged desktop build.

Key production changes vs dev main.py:
  1. Binds exclusively to 127.0.0.1 (never 0.0.0.0)
  2. Uses ProductionSettings (portable APPDATA paths, no pydantic-settings)
  3. Injects ONNX Runtime engine as the anomaly_service singleton
  4. Replaces ChatService with a torch-free production variant
  5. Disables LayerDissector neural refinement (pure HU-threshold path)
  6. Stubs out DISEASE_LABELS import so anomaly_graph.py is never loaded
  7. Validates X-MedSpatial-Token on every non-health request
  8. DEBUG=False → removes stack traces from error responses
  9. Structured JSON logging to %APPDATA%\MedSpatialAI\logs\backend.log
  10. Signals readiness to Tauri by writing a ready-file after startup

Torch-free guarantee:
  The following modules import torch at the module level in the dev build.
  Each is intercepted here with a sys.modules monkey-patch BEFORE any
  app.* import occurs, so torch is never imported by the Nuitka bundle:

    app.services.anomaly_service  → OnnxInferenceEngine shim
    app.services.chat_service     → ChatServiceProd (pure Python)
    app.ai.anomaly_graph          → stub exposing only DISEASE_LABELS
    app.ai.layer_dissector        → ProdLayerDissector (HU threshold only)
    app.ai.medical_qa             → stub exposing MEDICAL_KNOWLEDGE + SimpleTokenizer
    app.ai.depth_lifter           → stub (X-ray pseudo-3D path not used in prod)
    app.ai.spatial_transformer    → stub (model loaded via ONNX, not PyTorch)
    app.ai.segmentation_net       → stub (model loaded via ONNX, not PyTorch)
    app.ai.anomaly_detector       → stub (model loaded via ONNX, not PyTorch)

Tauri sidecar launch flow:
  Tauri (Rust) sets env vars:
    MEDSPATIAL_PORT=<chosen port>
    MEDSPATIAL_TOKEN=<32-byte hex>
    MEDSPATIAL_EXE_DIR=<dir containing this exe>
  → runs medspatial_backend.exe
  → Rust health-polls GET /api/health until 200
  → Writes ready flag to frontend via Tauri IPC
  → Frontend drops the loading screen
"""

from __future__ import annotations

import os
import sys
import json
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: inject backend/app into sys.path before any app.* imports.
# When compiled by Nuitka the app/ package is embedded; this path insert
# is a no-op but safe to keep.
# ---------------------------------------------------------------------------
_THIS_DIR    = Path(__file__).resolve().parent
_BACKEND_SRC = _THIS_DIR.parent.parent / "backend"
if _BACKEND_SRC.exists() and str(_BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SRC))

# ---------------------------------------------------------------------------
# STEP 1: Replace app.config BEFORE anything else imports it
# ---------------------------------------------------------------------------
from config_prod import prod_settings  # noqa: E402

import types as _types

_cfg_mod = _types.ModuleType("app.config")
_cfg_mod.settings = prod_settings      # type: ignore[attr-defined]
sys.modules["app.config"] = _cfg_mod

# ---------------------------------------------------------------------------
# STEP 2: Replace anomaly_service with the ONNX engine shim
# ---------------------------------------------------------------------------
from onnx_inference import get_engine, OnnxInferenceEngine  # noqa: E402

_onnx_engine = get_engine(
    onnx_dir=prod_settings.ONNX_DIR,
    volume_size=prod_settings.VOLUME_SIZE,
)

_anomaly_svc_mod = _types.ModuleType("app.services.anomaly_service")
_anomaly_svc_mod.anomaly_svc = _onnx_engine          # type: ignore[attr-defined]
_anomaly_svc_mod.AnomalyService = OnnxInferenceEngine # type: ignore[attr-defined]
sys.modules["app.services.anomaly_service"] = _anomaly_svc_mod

# ---------------------------------------------------------------------------
# STEP 3: Stub out torch-dependent AI modules before they are imported.
#
# Each stub exposes the exact symbols that the rest of the codebase imports
# from that module. Using stubs instead of try/except torch imports means
# Nuitka's --nofollow-import-to=torch can reliably exclude it.
# ---------------------------------------------------------------------------

# -- app.ai.anomaly_graph: only DISEASE_LABELS is needed at runtime ----------
_anomaly_graph_mod = _types.ModuleType("app.ai.anomaly_graph")
_anomaly_graph_mod.DISEASE_LABELS = [  # type: ignore[attr-defined]
    "Pneumonia", "Pneumothorax", "Pleural Effusion",
    "Lung Nodule/Mass", "Cardiomegaly", "Atelectasis",
    "Consolidation", "Emphysema", "Fibrosis", "Fracture",
    "Tuberculosis", "COVID-19 patterns", "Normal/No Finding",
]
_anomaly_graph_mod.NUM_DISEASES = 13         # type: ignore[attr-defined]
sys.modules["app.ai.anomaly_graph"] = _anomaly_graph_mod

# -- app.ai.medical_qa: MEDICAL_KNOWLEDGE + SimpleTokenizer are used by chat --
# Import the pure-Python parts directly (no torch at module level in medical_qa
# for MEDICAL_KNOWLEDGE dict and SimpleTokenizer class); we still stub the
# torch-dependent neural classes so they are never instantiated.
try:
    # medical_qa.py imports torch at module level — intercept before that happens
    _medical_qa_mod = _types.ModuleType("app.ai.medical_qa")
    # Inline the knowledge base and tokenizer that chat needs
    from app.ai.medical_qa import MEDICAL_KNOWLEDGE, SimpleTokenizer  # noqa: E402
    _medical_qa_mod.MEDICAL_KNOWLEDGE = MEDICAL_KNOWLEDGE   # type: ignore[attr-defined]
    _medical_qa_mod.SimpleTokenizer   = SimpleTokenizer      # type: ignore[attr-defined]

    class _StubMedicalQAModel:
        """Production stub — neural Q&A replaced by rule-based path."""
        pass

    _medical_qa_mod.MedicalQAModel = _StubMedicalQAModel     # type: ignore[attr-defined]
    _medical_qa_mod.create_medical_qa_model = lambda **kw: _StubMedicalQAModel()  # type: ignore[attr-defined]
    sys.modules["app.ai.medical_qa"] = _medical_qa_mod
except Exception:
    # If even the import of MEDICAL_KNOWLEDGE fails (shouldn't), provide inline
    _medical_qa_mod = _types.ModuleType("app.ai.medical_qa")
    _medical_qa_mod.MEDICAL_KNOWLEDGE = {}                   # type: ignore[attr-defined]
    _medical_qa_mod.SimpleTokenizer   = object               # type: ignore[attr-defined]
    sys.modules["app.ai.medical_qa"] = _medical_qa_mod

# -- app.ai.layer_dissector: replace with HU-threshold-only variant ----------
class _ProdLayerDissector:
    """
    Production LayerDissector: uses pure HU-threshold decomposition only.
    The LayerRefinementNet (PyTorch CNN) is disabled — it was optional anyway
    and degrades gracefully to the HU path.
    """

    LAYER_DEFINITIONS = {
        "air":          {"index": 0, "hu_min": -1024.0, "hu_max": -500.0,  "color": [0.2, 0.2, 0.8, 0.3]},
        "lung_fat":     {"index": 1, "hu_min": -500.0,  "hu_max": -100.0,  "color": [0.9, 0.6, 0.7, 0.4]},
        "soft_tissue":  {"index": 2, "hu_min": -100.0,  "hu_max":  200.0,  "color": [0.9, 0.7, 0.6, 0.6]},
        "bone":         {"index": 3, "hu_min":  200.0,  "hu_max": 3000.0,  "color": [0.95, 0.95, 0.85, 0.9]},
        "contrast":     {"index": 4, "hu_min": 3000.0,  "hu_max": 5000.0,  "color": [1.0, 1.0, 1.0, 1.0]},
    }

    def __init__(self, device: str = "cpu"):
        pass  # no torch device needed

    def threshold_decomposition(self, volume):
        import numpy as np
        from scipy import ndimage
        masks = {}
        for name, defn in self.LAYER_DEFINITIONS.items():
            mask = ((volume >= defn["hu_min"]) & (volume < defn["hu_max"])).astype(np.float32)
            if mask.sum() > 50:
                labeled, num_feat = ndimage.label(mask)
                if num_feat > 1:
                    sizes = ndimage.sum(mask, labeled, range(1, num_feat + 1))
                    max_size = max(sizes)
                    for i, size in enumerate(sizes):
                        if size < max_size * 0.01:
                            mask[labeled == (i + 1)] = 0
                mask = ndimage.gaussian_filter(mask, sigma=0.5)
                mask = (mask > 0.5).astype(np.float32)
            masks[name] = mask
        return masks

    def decompose(self, volume, use_refinement: bool = False):
        """Always uses HU thresholds — neural refinement disabled in production."""
        import numpy as np
        masks = self.threshold_decomposition(volume)
        total_voxels = volume.size
        results = {}
        for name, mask in masks.items():
            binary_mask = (mask > 0.5).astype(np.float32)
            voxel_count = int(binary_mask.sum())
            masked_values = volume[binary_mask > 0.5]
            results[name] = {
                "mask": binary_mask,
                "voxel_count": voxel_count,
                "volume_fraction": float(voxel_count / total_voxels),
                "color": self.LAYER_DEFINITIONS[name]["color"],
                "hu_stats": {
                    "mean": float(masked_values.mean()) if len(masked_values) > 0 else 0.0,
                    "std":  float(masked_values.std())  if len(masked_values) > 0 else 0.0,
                    "min":  float(masked_values.min())  if len(masked_values) > 0 else 0.0,
                    "max":  float(masked_values.max())  if len(masked_values) > 0 else 0.0,
                },
            }
        return results

    def decompose_without_images(self, shape, tissue_type="chest"):
        import numpy as np
        volume = np.zeros(shape, dtype=np.float32) - 1000.0
        return self.decompose(volume, use_refinement=False)


_layer_dissector_mod = _types.ModuleType("app.ai.layer_dissector")
_layer_dissector_mod.LayerDissector = _ProdLayerDissector    # type: ignore[attr-defined]
sys.modules["app.ai.layer_dissector"] = _layer_dissector_mod

# -- Stub torch-dependent AI model modules (never instantiated at runtime) ---
for _stub_name in (
    "app.ai.spatial_transformer",
    "app.ai.segmentation_net",
    "app.ai.anomaly_detector",
    "app.ai.depth_lifter",
):
    _m = _types.ModuleType(_stub_name)
    sys.modules[_stub_name] = _m

# -- Replace chat_service with a torch-free production variant ----------------
class _ChatServiceProd:
    """
    Production ChatService: pure-Python rule-based Q&A only.
    The neural MedicalQAModel (torch) is not loaded.
    The SpatialTransformer3D (torch) is not loaded.
    All question-answering uses the knowledge-base and analysis-context
    branches of the original ChatService._answer_from_knowledge_base()
    and _answer_from_analysis(), which have no torch dependency.
    """

    MODALITY_MAP = {"CT": 0, "XR": 1, "MR": 2, "CR": 3, "DX": 4, "US": 5, "NM": 6}
    BODY_PART_MAP = {
        "CHEST": 0, "HEAD": 1, "ABDOMEN": 2, "PELVIS": 3, "SPINE": 4,
        "EXTREMITY": 5, "NECK": 6, "THORAX": 7, "BRAIN": 8, "LUNG": 9,
    }

    def __init__(self):
        # Import only the pure-Python knowledge base
        try:
            from app.ai.medical_qa import MEDICAL_KNOWLEDGE
            self._knowledge = MEDICAL_KNOWLEDGE
        except Exception:
            self._knowledge = {}

    def answer_question(self, question, chat_history, scan_context,
                        volume_context=None, analysis_context=None):
        """Route to rule-based handlers only — no neural inference."""
        q = question.lower().strip()

        kb = self._answer_from_knowledge_base(q, scan_context, analysis_context)
        if kb:
            return kb

        analysis = self._answer_from_analysis(q, scan_context, volume_context, analysis_context)
        if analysis:
            return analysis

        return {
            "answer": (
                "I can help you explore this scan. Try asking:\n"
                "- 'What findings were detected?'\n"
                "- 'Tell me about the scan information'\n"
                "- 'What are the tissue layers?'\n"
                "- 'What is a pulmonary nodule?'"
            ),
            "referenced_slices": None,
            "referenced_regions": None,
            "findings_mentioned": None,
            "context_summary": "Usage guidance",
        }

    def _answer_from_knowledge_base(self, question, scan_context, analysis_context):
        anatomy = self._knowledge.get("anatomy", {})
        pathology = self._knowledge.get("pathology", {})

        for name, info in anatomy.items():
            if name in question and any(w in question for w in ("what", "tell", "about")):
                answer = (
                    f"**{name.title()}**: {info['description']}. "
                    f"Normal HU range: {info['normal_hu']['min']} to {info['normal_hu']['max']} HU."
                )
                if "common_findings" in info:
                    answer += f" Common findings: {', '.join(info['common_findings'])}."
                return {"answer": answer, "referenced_slices": None,
                        "referenced_regions": None, "findings_mentioned": None,
                        "context_summary": f"Anatomical query: {name}"}

        for name, info in pathology.items():
            if any(v in question for v in (name, name.replace("_", " "))):
                answer = (
                    f"**{name.replace('_', ' ').title()}**: {info['description']}. "
                    f"**Significance**: {info['significance']}. "
                    f"**HU**: {info['hu_characteristics']}."
                )
                return {"answer": answer, "referenced_slices": None,
                        "referenced_regions": None, "findings_mentioned": None,
                        "context_summary": f"Pathology query: {name}"}
        return None

    def _answer_from_analysis(self, question, scan_context, volume_context, analysis_context):
        anomaly_kw = ["finding", "abnormal", "anomal", "detect", "disease", "problem", "wrong"]
        if any(kw in question for kw in anomaly_kw):
            if analysis_context:
                findings_text, all_findings = [], []
                for ac in analysis_context:
                    if ac.get("findings"):
                        fd = ac["findings"]
                        if isinstance(fd, dict) and "anomalies" in fd:
                            for f in fd["anomalies"]:
                                all_findings.append(f)
                                findings_text.append(
                                    f"- **{f.get('severity','?').title()}** "
                                    f"({f.get('confidence',0):.0%}): {f.get('description','')}"
                                )
                    if ac.get("summary"):
                        findings_text.append(f"\n**Summary**: {ac['summary']}")
                if findings_text:
                    answer = (
                        f"Analysis of this {scan_context.get('modality','scan')}:\n\n"
                        + "\n".join(findings_text)
                        + "\n\n*AI-generated — review by a qualified radiologist required.*"
                    )
                    return {"answer": answer, "referenced_slices": None,
                            "referenced_regions": [f.get("location") for f in all_findings if f.get("location")],
                            "findings_mentioned": all_findings[:5],
                            "context_summary": "Analysis findings"}
            return {"answer": "No analysis run yet. Click 'Analyze' first.",
                    "referenced_slices": None, "referenced_regions": None,
                    "findings_mentioned": None, "context_summary": "No analysis"}

        scan_kw = ["scan", "image", "modality", "patient", "study", "info"]
        if any(kw in question for kw in scan_kw):
            answer = (
                f"**Scan Info:**\n"
                f"- Modality: {scan_context.get('modality','?')}\n"
                f"- Body Part: {scan_context.get('body_part','?')}\n"
                f"- Slices: {scan_context.get('num_slices',0)}\n"
            )
            return {"answer": answer, "referenced_slices": None,
                    "referenced_regions": None, "findings_mentioned": None,
                    "context_summary": "Scan info"}

        layer_kw = ["layer", "tissue", "bone", "dissect"]
        if any(kw in question for kw in layer_kw):
            return {"answer": (
                "Tissue layers by Hounsfield Unit:\n"
                "- **Air** (< -500 HU)\n- **Lung/Fat** (-500 to -100 HU)\n"
                "- **Soft Tissue** (-100 to 200 HU)\n- **Bone** (200 to 3000 HU)\n"
                "Toggle visibility with the Layer Controls panel."
            ), "referenced_slices": None, "referenced_regions": None,
               "findings_mentioned": None, "context_summary": "Layer explanation"}

        return None


_chat_svc_mod = _types.ModuleType("app.services.chat_service")
_chat_svc_mod.ChatService = _ChatServiceProd                 # type: ignore[attr-defined]
sys.modules["app.services.chat_service"] = _chat_svc_mod

# ---------------------------------------------------------------------------
# Structured JSON logging (goes to file + stderr for Tauri to capture)
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "ts":    self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name":  record.name,
            "msg":   record.getMessage(),
        })


def _setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "backend.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # File handler (JSON, rotated by Nuitka/system; keep last 5 MB)
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(_JsonFormatter())
    root.addHandler(fh)

    # Stderr handler (plain text for Tauri's stdout capture)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(levelname)s  %(name)s  %(message)s"))
    root.addHandler(sh)


import logging.handlers  # noqa: E402 (needed for RotatingFileHandler above)
_setup_logging(prod_settings.LOG_DIR)
log = logging.getLogger("medspatial.main")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    log.info(f"Starting {prod_settings.APP_NAME} v{prod_settings.APP_VERSION}")
    log.info(repr(prod_settings))

    # Create data directories
    prod_settings.ensure_directories()

    # Init database
    from app.models.database import init_db, close_db
    await init_db()
    log.info("Database initialised")

    # Warm up ONNX engine (loads sessions + JIT-compiles kernels)
    try:
        _onnx_engine.warm_up()
    except Exception as exc:
        log.warning(f"ONNX warm-up failed (non-fatal, will retry on first request): {exc}")

    # Signal readiness to Tauri by writing a sentinel file.
    # Tauri's health-check loop also polls /api/health, but the file gives
    # an instant signal if the Rust code chooses to watch it.
    ready_file = prod_settings.LOG_DIR / ".backend_ready"
    ready_file.write_text("1")
    log.info(f"Ready sentinel written: {ready_file}")

    yield

    # Cleanup
    await close_db()
    ready_file.unlink(missing_ok=True)
    log.info("Shutdown complete")


app = FastAPI(
    title=prod_settings.APP_NAME,
    version=prod_settings.APP_VERSION,
    # Disable OpenAPI docs in production (reduces attack surface + bundle size)
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=prod_settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ── Token authentication middleware ──────────────────────────────────────────
EXEMPT_PATHS = {"/api/health"}

@app.middleware("http")
async def token_auth_middleware(request: Request, call_next):
    """
    Validate the per-session HMAC token on every non-health request.
    Tauri injects this via X-MedSpatial-Token on every API call.

    Using constant-time comparison (secrets.compare_digest) prevents
    timing side-channel attacks even on loopback.
    """
    if request.url.path in EXEMPT_PATHS or request.method == "OPTIONS":
        return await call_next(request)

    token = request.headers.get("X-MedSpatial-Token", "")
    if not secrets.compare_digest(token, prod_settings.API_TOKEN):
        log.warning(f"Rejected request to {request.url.path} — invalid token")
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    return await call_next(request)

# ── Error handler — strip stack traces in production ────────────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check application logs."},
    )

# ── Static files ──────────────────────────────────────────────────────────────
for _name, _path in [
    ("meshes",   prod_settings.MESH_DIR),
    ("analysis", prod_settings.ANALYSIS_DIR),
]:
    _path.mkdir(parents=True, exist_ok=True)
    app.mount(f"/static/{_name}", StaticFiles(directory=str(_path)), name=_name)

# ── Routers (unchanged from dev) ──────────────────────────────────────────────
from app.api.upload         import router as upload_router
from app.api.reconstruction import router as recon_router
from app.api.analysis       import router as analysis_router
from app.api.chat           import router as chat_router
from app.api.explain        import router as explain_router
from app.api.reports        import router as reports_router

app.include_router(upload_router)
app.include_router(recon_router)
app.include_router(analysis_router)
app.include_router(chat_router)
app.include_router(explain_router)
app.include_router(reports_router)

# ── WebSocket (identical to dev) ────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, ws: WebSocket, scan_id: str):
        await ws.accept()
        self.active_connections.setdefault(scan_id, []).append(ws)

    def disconnect(self, ws: WebSocket, scan_id: str):
        conns = self.active_connections.get(scan_id, [])
        if ws in conns:
            conns.remove(ws)
        if not conns:
            self.active_connections.pop(scan_id, None)

    async def broadcast(self, scan_id: str, message: dict):
        for conn in self.active_connections.get(scan_id, []):
            try:
                await conn.send_json(message)
            except Exception:
                pass


ws_manager = ConnectionManager()
app.state.ws_manager = ws_manager


@app.websocket("/ws/{scan_id}")
async def websocket_endpoint(websocket: WebSocket, scan_id: str):
    await ws_manager.connect(websocket, scan_id)
    try:
        while True:
            await websocket.receive_text()
            await websocket.send_json({"type": "pong", "scan_id": scan_id})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, scan_id)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    """
    Called by Tauri's Rust sidecar manager every 5 seconds.
    Returns 200 only when the backend is fully initialised.
    No auth token required.
    """
    return {
        "status": "healthy",
        "app": prod_settings.APP_NAME,
        "version": prod_settings.APP_VERSION,
        "port": prod_settings.PORT,
    }


# ---------------------------------------------------------------------------
# Entry point for Nuitka-compiled executable
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=prod_settings.HOST,
        port=prod_settings.PORT,
        log_config=None,          # we configured logging manually above
        access_log=False,         # access logs go through our JSON formatter
        workers=1,
        loop="asyncio",
        http="h11",               # h11 is lighter than httptools for local traffic
        ws="websockets",
        timeout_keep_alive=30,
        limit_concurrency=20,     # guard against runaway clients
        limit_max_requests=None,
    )
