# MedSpatial AI

## AWM-JGEM Refactor

The application now includes an Anatomical World Model control plane with:

- whole-body typed anatomical structures and object-slot memory
- lightweight JEPA latent prediction
- anatomical scene graph reasoning with a pure-PyTorch fallback
- interpretable energy-based plausibility scoring
- budget-aware local expert routing
- promptable segmentation provider adapters
- evidence-grounded clinical Q&A
- deterministic policy evaluation and PHI-aware audit logs
- explicit scan-derived, estimated, synthetic, and user-authored provenance

The original upload, reconstruction, analysis, mesh, and viewer APIs remain
available. New contracts are exposed under `/api/studies`, `/api/intents`,
`/api/audit`, and `/api/models/registry`.

Single-view radiography-derived 3D is labeled estimated, probabilistic,
atlas-aligned, and non-ground-truth. No-image atlas content is labeled
synthetic, educational, non-diagnostic, and non-patient-specific.

See `docs/architecture/awm-jgem.md` and `docs/safety/clinical-safety.md`.

Common commands:

```bash
make install       # local dependencies, no implicit model download
make dev           # backend + frontend development servers
make run           # complete Docker stack
make doctor        # runtime and dependency checks
```

**3D Medical Imaging Platform** — Convert X-ray and CT DICOM images into interactive 3D volumetric models with AI-powered layer dissection, anomaly detection, and conversational Q&A.

## Architecture

| Component | Stack |
|-----------|-------|
| **Backend** | Python 3.11 · FastAPI · PyTorch · pydicom · scikit-image |
| **Frontend** | React 18 · Three.js · React Three Fiber · Vite |
| **AI Models** | SpatialTransformer3D · 3D U-Net · VAE Anomaly Detector · Medical Q&A |
| **Database** | SQLite (async) via SQLAlchemy |
| **3D Output** | GLB (glTF Binary) meshes via custom generator |

## Features

- 📤 **DICOM Upload** — Drag-and-drop multi-file DICOM ingestion with metadata extraction
- 🏗️ **3D Reconstruction** — Marching cubes surface extraction with voxel-accurate spacing
- 🔬 **Layer Dissection** — Separate bone, soft tissue, air, vessels with opacity controls
- 🌡️ **Anomaly Detection** — VAE + density estimation with 3D heatmap overlays
- 🧠 **Custom AI** — Novel SpatialTransformer3D with cross-plane attention (not standard ViT)
- 💬 **Medical Q&A** — Ask questions about anatomy, findings, and scan details
- 🖼️ **Slice Viewer** — Navigate axial/coronal/sagittal planes interactively
- ✂️ **Clipping Plane** — Cut through the 3D model in real-time

## Quick Start

### Option 1: Docker (Recommended)

```bash
docker-compose up --build
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Option 2: Local Development

**Backend:**

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Project Structure

```
medspatial-ai/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI entry point
│   │   ├── config.py               # Configuration
│   │   ├── models/                 # Database models
│   │   ├── schemas/                # API schemas
│   │   ├── api/                    # Route handlers
│   │   │   ├── upload.py           # DICOM upload
│   │   │   ├── reconstruction.py   # 3D reconstruction
│   │   │   ├── analysis.py         # AI analysis
│   │   │   └── chat.py             # Medical Q&A
│   │   ├── services/               # Business logic
│   │   │   ├── dicom_service.py    # DICOM parsing
│   │   │   ├── reconstruction_service.py
│   │   │   ├── anomaly_service.py  # AI orchestration
│   │   │   └── chat_service.py     # Q&A logic
│   │   ├── ai/                     # Neural networks
│   │   │   ├── spatial_transformer.py  # Custom 3D ViT
│   │   │   ├── segmentation_net.py     # 3D U-Net
│   │   │   ├── anomaly_detector.py     # VAE detector
│   │   │   ├── layer_dissector.py      # Tissue separation
│   │   │   └── medical_qa.py          # Q&A model
│   │   └── core/                   # Utilities
│   │       ├── volume_processor.py # HU windowing
│   │       └── mesh_generator.py   # GLB exporter
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Main layout
│   │   ├── index.css               # Design system
│   │   ├── components/
│   │   │   ├── Viewer3D.jsx        # Three.js 3D viewer
│   │   │   ├── LayerControls.jsx   # Dissection UI
│   │   │   ├── SliceViewer.jsx     # 2D slices
│   │   │   ├── UploadPanel.jsx     # DICOM upload
│   │   │   ├── ChatPanel.jsx       # AI chat
│   │   │   ├── AnomalyOverlay.jsx  # Findings panel
│   │   │   └── Header.jsx
│   │   └── services/api.js         # API client
│   └── package.json
├── docker-compose.yml
└── README.md
```

## AI Architecture: SpatialTransformer3D

The core innovation is the **SpatialTransformer3D** — a custom 3D vision transformer designed specifically for volumetric medical data:

1. **3D Patch Embedding** — Overlapping 3D convolution-based patching
2. **Cross-Plane Attention** — Learns inter-slice relationships across axial/coronal/sagittal planes simultaneously
3. **Volumetric Position Encoding** — Combined sinusoidal + learnable 3D positional encoding
4. **Spatial Attention Gates** — Dynamic focusing on anatomically relevant regions
5. **Multi-Scale Feature Pyramid** — Extracts features at multiple resolutions

This is **not** a standard Vision Transformer — it's specifically designed for the unique challenges of 3D medical imaging.

## Training

The models ship with random weights for structural completeness. To train with your own data:

1. Place DICOM datasets in `data/training/`
2. Use the bundled training utilities in `backend/app/ai/`
3. Save weights to `models/` directory
4. Update paths in `.env` or `config.py`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/scans/upload` | Upload DICOM files |
| GET | `/api/scans/` | List all scans |
| POST | `/api/reconstruction/build` | Start 3D reconstruction |
| GET | `/api/reconstruction/mesh/{id}/{layer}` | Get GLB mesh |
| POST | `/api/reconstruction/slice` | Get 2D slice |
| POST | `/api/analysis/run` | Run AI analysis |
| POST | `/api/chat/ask` | Ask medical question |
| WS | `/ws/{scan_id}` | Real-time updates |

## License

For research and educational purposes. Not intended for clinical diagnostic use without proper validation and regulatory approval.
