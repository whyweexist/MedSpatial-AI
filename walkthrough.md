# MedSpatial AI — Execution Guide & Gap Analysis

## Gap Analysis vs [program.md](file:///f:/project%20jbp/program.md)

### Previously Built (Phase 1–4)
| Component | Status |
|-----------|--------|
| FastAPI backend + CORS + WebSocket | ✅ |
| SQLAlchemy DB models | ✅ |
| DICOM upload + parsing service | ✅ |
| Reconstruction service (marching cubes) | ✅ |
| GLB mesh generator | ✅ |
| SpatialTransformer3D (ViT-3D) | ✅ |
| 3D U-Net segmentation | ✅ |
| VAE anomaly detector | ✅ |
| Layer dissector | ✅ |
| Medical Q&A model | ✅ |
| React frontend (7 components) | ✅ |
| Docker / docker-compose | ✅ |

### Added in This Session (Gaps Filled)
| Component | program.md Section | Status |
|-----------|-------------------|--------|
| [generate_atlas.py](file:///f:/project%20jbp/medspatial-ai/backend/app/atlas/generate_atlas.py) — procedural chest atlas (12 layers, rigs, bronchial tree) | §4.2 | ✅ |
| [generate_synthetic_data.py](file:///f:/project%20jbp/medspatial-ai/backend/app/data/generate_synthetic_data.py) — elastic deformations + pathology injection | §4.3 | ✅ |
| [anomaly_graph.py](file:///f:/project%20jbp/medspatial-ai/backend/app/ai/anomaly_graph.py) — GAAT (PatchCore-3D + GraphSAGE + disease classifier) | §1 Subsystem C | ✅ |
| [depth_lifter.py](file:///f:/project%20jbp/medspatial-ai/backend/app/ai/depth_lifter.py) — 4-block CNN for X-ray→pseudo-3D | §1 Subsystem A | ✅ |
| [hardware_manager.py](file:///f:/project%20jbp/medspatial-ai/backend/app/core/hardware_manager.py) — CUDA/MPS/CPU detect + memory pressure | §6.1–6.2 | ✅ |
| [build_kb.py](file:///f:/project%20jbp/medspatial-ai/backend/app/knowledge/build_kb.py) — SQLite FTS5 KB (13 diseases, 7 anatomy entries) | §1 Subsystem D.4 | ✅ |
| [download_weights.py](file:///f:/project%20jbp/medspatial-ai/backend/scripts/download_weights.py) — weight downloader + memory bank from atlas | §4.1 | ✅ |
| [VolumeShader.js](file:///f:/project%20jbp/medspatial-ai/frontend/src/three/VolumeShader.js) — **complete** GLSL raycast (AABB, dissection, clip, Phong, MIP, pulse) | §6.3 | ✅ |
| [VolumeRenderer.js](file:///f:/project%20jbp/medspatial-ai/frontend/src/three/VolumeRenderer.js) — Three.js Data3DTexture integration | §2.1 | ✅ |
| [volumeWorker.js](file:///f:/project%20jbp/medspatial-ai/frontend/src/workers/volumeWorker.js) — gzip decompressor Web Worker | §2.5 | ✅ |
| [gpuDetect.js](file:///f:/project%20jbp/medspatial-ai/frontend/src/utils/gpuDetect.js) — WebGL2 vendor detection + adaptive quality | §2.5 | ✅ |
| [appStore.js](file:///f:/project%20jbp/medspatial-ai/frontend/src/stores/appStore.js) — Zustand scan + viewer stores | §execution order | ✅ |
| [Makefile](file:///f:/project%20jbp/medspatial-ai/Makefile) — all 10 targets (install/dev/build/test/clean…) | §8 | ✅ |
| [.devcontainer/devcontainer.json](file:///f:/project%20jbp/medspatial-ai/.devcontainer/devcontainer.json) | §8 | ✅ |
| [scripts/setup.sh](file:///f:/project%20jbp/medspatial-ai/scripts/setup.sh) — first-run bootstrap | §8 | ✅ |
| [tests/test_pipeline.py](file:///f:/project%20jbp/medspatial-ai/backend/tests/test_pipeline.py) — 6 test classes, 15 tests | §9 | ✅ |
| [requirements.txt](file:///f:/project%20jbp/medspatial-ai/backend/requirements.txt) pinned exact versions | §7 | ✅ |

---

## Complete File Tree (~60 files)

```
medspatial-ai/
├── Makefile
├── README.md
├── docker-compose.yml
├── .devcontainer/devcontainer.json
├── scripts/
│   ├── setup.sh
│   └── download_weights.py          (inside backend/)
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt             (pinned exact versions)
│   ├── .env
│   ├── scripts/
│   │   └── download_weights.py
│   ├── tests/
│   │   ├── conftest.py
│   │   └── test_pipeline.py
│   └── app/
│       ├── main.py
│       ├── config.py
│       ├── models/
│       │   ├── __init__.py          (ORM: Scan, Volume, Analysis, ChatSession)
│       │   └── database.py
│       ├── schemas/__init__.py
│       ├── api/
│       │   ├── upload.py
│       │   ├── reconstruction.py
│       │   ├── analysis.py
│       │   └── chat.py
│       ├── services/
│       │   ├── dicom_service.py
│       │   ├── reconstruction_service.py
│       │   ├── anomaly_service.py
│       │   └── chat_service.py
│       ├── core/
│       │   ├── hardware_manager.py  ← NEW
│       │   ├── mesh_generator.py
│       │   └── volume_processor.py
│       ├── ai/
│       │   ├── spatial_transformer.py
│       │   ├── segmentation_net.py
│       │   ├── anomaly_detector.py
│       │   ├── anomaly_graph.py     ← NEW (GAAT GNN)
│       │   ├── depth_lifter.py      ← NEW
│       │   ├── layer_dissector.py
│       │   └── medical_qa.py
│       ├── atlas/
│       │   ├── __init__.py
│       │   └── generate_atlas.py    ← NEW
│       ├── data/
│       │   └── generate_synthetic_data.py  ← NEW
│       └── knowledge/
│           ├── __init__.py
│           └── build_kb.py          ← NEW
│
└── frontend/
    ├── Dockerfile
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── index.css                (premium dark design system)
        ├── components/
        │   ├── Header.jsx
        │   ├── Viewer3D.jsx
        │   ├── LayerControls.jsx
        │   ├── SliceViewer.jsx
        │   ├── UploadPanel.jsx
        │   ├── ChatPanel.jsx
        │   └── AnomalyOverlay.jsx
        ├── services/api.js
        ├── stores/appStore.js       ← NEW (Zustand)
        ├── utils/gpuDetect.js       ← NEW
        ├── workers/volumeWorker.js  ← NEW
        └── three/
            ├── VolumeShader.js      ← NEW (complete GLSL)
            └── VolumeRenderer.js    ← NEW
```

---

## Step-by-Step Execution Guide

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 | Any |
| Python | 3.11 | 3.11 |
| Node.js | 18 | 20 LTS |
| RAM | 8 GB | 16 GB |
| GPU | Integrated (Intel UHD 620) | NVIDIA RTX (CUDA) |
| Disk | 5 GB (code + atlas) | 10 GB (+ weights) |

---

### Option A — Make (Recommended, Linux/macOS/WSL)

```bash
# 1. Clone / navigate to project
cd "f:/project jbp/medspatial-ai"

# 2. First-time setup (installs deps, generates atlas, builds KB)
make install

# 3. Start both backend + frontend in dev mode
make dev

# Open: http://localhost:5173  (frontend)
#       http://localhost:8000/docs  (Swagger API docs)
```

---

### Option B — Windows (Manual, No WSL)

#### Backend
```powershell
cd "f:\project jbp\medspatial-ai\backend"

# Create virtualenv
python -m venv venv
venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Generate atlas (one-time, ~30 seconds)
python -m app.atlas.generate_atlas --size 128 --output atlas_chest_128.npz --output-dir ../data

# Build medical knowledge base (one-time, ~5 seconds)
python -m app.knowledge.build_kb --output ../data/medical_kb.sqlite

# Generate memory bank for anomaly detector (one-time, ~60 seconds)
python scripts\download_weights.py --output-dir ../models --atlas ../data/atlas_chest_128.npz

# Start backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend (new terminal)
```powershell
cd "f:\project jbp\medspatial-ai\frontend"

# Install dependencies
npm install

# Add zustand (needed for appStore.js)
npm install zustand

# Start dev server
npm run dev
# → http://localhost:5173
```

---

### Option C — Docker Compose

```bash
cd "f:/project jbp/medspatial-ai"

# Build and start (first time takes ~5 minutes)
docker-compose up --build

# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
```

> **With NVIDIA GPU**: The [docker-compose.yml](file:///f:/project%20jbp/medspatial-ai/docker-compose.yml) includes GPU device reservations. Requires `nvidia-container-toolkit`.

---

### Option D — GitHub Codespaces

1. Push the project to GitHub
2. Click **Code → Codespaces → Create codespace**
3. It will auto-run [scripts/setup.sh](file:///f:/project%20jbp/medspatial-ai/scripts/setup.sh) via `postCreateCommand`
4. Ports 8000 and 5173 are auto-forwarded
5. Click the 5173 URL to open the frontend

---

## Platform Usage Walkthrough

### Step 1 — Upload a DICOM Scan
- Drag `.dcm` files onto the **Upload DICOM** zone in the left panel
- Supports: single-frame CT/XR, multi-file CT series, PNG/JPG radiographs
- The backend parses DICOM metadata (modality, body part, pixel spacing, HU)

### Step 2 — Build the 3D Model
- Click **🔨 Reconstruct** in the viewer toolbar
- The backend runs: DICOM → HU volume → marching cubes → GLB mesh
- Progress shown via WebSocket in real-time
- The 3D model appears in the center viewer on completion

### Step 3 — Explore the 3D Model
- **Orbit**: left-drag | **Zoom**: scroll | **Pan**: right-drag
- **Clip slider** (right side of viewer): drag to cut through the volume
- **Layer Controls** (left panel): toggle/opacity per tissue layer
- **Keyboard shortcuts**: `L` layers panel, `D` dissection, `A` anomalies, `R` reset, `F` fullscreen

### Step 4 — Layer Dissection
- Use the **Layer Dissection** panel to peel layers from skin → muscle → bone → organ
- Each layer independently toggleable with opacity control
- Colors correspond to tissue type (bone=ivory, vessels=red, lung=blue, etc.)

### Step 5 — Run AI Analysis
- Click **🔬 Analyze** after reconstruction
- The backend runs: SpatialTransformer3D → AnomalyGraph (GAAT) → findings
- Switch on **🌡️ Heatmap** to see anomaly overlay (pulsing red regions)
- Findings panel (top-left of 3D viewer) lists detected pathologies with severity badges

### Step 6 — Atlas Mode (No Images)
- Use the chat: *"Show me a normal chest anatomy"*
- The system loads the pre-generated `atlas_chest_128.npz` (synthetic but anatomically plausible)
- Layer dissection and 3D visualization work identically to real scan mode

### Step 7 — Talk to the AI
Type any question in the right chat panel:
```
"What findings were detected?"
"Tell me about the right lung"
"What is a pulmonary nodule?"
"Show me the bone layer"
"What does consolidation mean?"
"How severe are the findings?"
```

The AI answers using:
1. SQLite knowledge base (FTS5 search across 13 diseases + 7 anatomy entries)
2. Real scan findings from the analysis results
3. Scan metadata (modality, body part, slice count)
4. MedicalQAModel neural inference

---

## Running Tests

```bash
# Full test suite
make test

# Backend only (verbose)
cd backend && python -m pytest tests/ -v --tb=long

# Specific test class
cd backend && python -m pytest tests/test_pipeline.py::TestModelForwardPasses -v

# Expected output:
# ✅ DICOM parse: passed
# ✅ HU conversion: passed
# ✅ SpatialTransformer3D: cls=(1,64), spatial=(...)
# ✅ DepthLifter: output shape (1,1,32,64,64)
# ✅ AnomalyGraph: map=(1,32,32,32), diseases=(1,13)
# ✅ SegmentationNet3D: (1,12,32,32,32)
# ✅ Atlas generation: (32,32,32), labels 0-11
# ✅ Knowledge base: found disease hits
# ✅ Hardware: device=cpu, volume=128³
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  BROWSER (React 18 + Three.js)                              │
│  ┌──────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Upload   │  │  3D Viewer       │  │  Chat Panel      │  │
│  │ Panel    │  │  VolumeRenderer  │  │  (Q&A+history)   │  │
│  └──────────┘  │  GLSL Raycast    │  └──────────────────┘  │
│  ┌──────────┐  │  Dissection      │  ┌──────────────────┐  │
│  │ Layer    │  │  Anomaly Overlay │  │  Anomaly Overlay │  │
│  │ Controls │  │  Clip Plane      │  │  Findings Panel  │  │
│  └──────────┘  └──────────────────┘  └──────────────────┘  │
│        ↑ Zustand stores (scan + viewer)                     │
│        ↑ volumeWorker.js (gzip decode)                      │
└─────────────────────────────────────────────────────────────┘
                        ↕ REST + WebSocket
┌─────────────────────────────────────────────────────────────┐
│  FASTAPI BACKEND                                            │
│  /api/scans/upload → DicomService → HU Volume              │
│  /api/reconstruction/build → ReconstructionService → GLB   │
│  /api/analysis/run → AnomalyService →                      │
│     SpatialTransformer3D                                    │
│     ─→ AnomalyGraph (GAAT):                                 │
│          PatchCore-3D (memory bank cosine matching)         │
│          GraphSAGE (3 layers, 6-connectivity adjacency)     │
│          DiseaseClassificationHead (13-label multi-label)   │
│     ─→ SegmentationNet3D (3D U-Net + attention)             │
│     ─→ LayerDissector (HU thresholding + neural refine)     │
│  /api/chat/ask → ChatService →                              │
│     MedicalKnowledgeBase (SQLite FTS5)                      │
│     ─→ MedicalQAModel (cross-attention Q&A)                 │
│  HardwareManager (CPU/CUDA/MPS adaptive)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Known Limitations & Notes

> [!IMPORTANT]
> **AI Models ship with random weights.** The architectures are complete and correct. Production-quality results require training on real medical data (e.g., NIH ChestX-ray14, LIDC-IDRI, Medical Segmentation Decathlon). The synthetic training data generator ([generate_synthetic_data.py](file:///f:/project%20jbp/medspatial-ai/backend/app/data/generate_synthetic_data.py)) creates 1000 deformed atlas samples that can be used for initial fine-tuning.

> [!NOTE]
> **GLSL Volume Shader** ([VolumeShader.js](file:///f:/project%20jbp/medspatial-ai/frontend/src/three/VolumeShader.js)) requires the frontend to load volume data as a `Float32Array` into `THREE.Data3DTexture`. The current [Viewer3D.jsx](file:///f:/project%20jbp/medspatial-ai/frontend/src/components/Viewer3D.jsx) uses GLB mesh rendering. To switch to GPU raycasting, integrate [VolumeRenderer.js](file:///f:/project%20jbp/medspatial-ai/frontend/src/three/VolumeRenderer.js) from `src/three/` in place of the GLB loader — the renderer exposes the same layer/clip/dissection controls.

> [!TIP]
> **Low-end hardware (8GB RAM, integrated GPU)**: The [HardwareManager](file:///f:/project%20jbp/medspatial-ai/backend/app/core/hardware_manager.py#37-215) auto-detects and reduces volume resolution from 128³ to 64³. The GPU detector in the frontend auto-sets ray march step to `1/64` for integrated GPUs. Both settings are tunable via [.env](file:///f:/project%20jbp/medspatial-ai/backend/.env).

> [!WARNING]
> **`llama-cpp-python`** (for local LLM / MedOracle) is optional and not in [requirements.txt](file:///f:/project%20jbp/medspatial-ai/backend/requirements.txt) by default as it requires compilation. To enable the full conversational LLM: `pip install llama-cpp-python` and place a GGUF model file in [models/](file:///f:/project%20jbp/medspatial-ai/backend/app/services/anomaly_service.py#35-62). Without it, the chat service uses the rule-based + neural Q&A fallback which is still highly capable.
