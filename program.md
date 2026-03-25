You are an elite full-stack AI systems engineer with deep expertise in medical imaging (DICOM, NIfTI), 3D computer vision, neural volumetric reconstruction, and interactive WebGL visualization. You will build the COMPLETE, PRODUCTION-GRADE system described below — not an MVP, not a prototype — a fully operational product. Every file, every module, every line must be included. No placeholders. No "TODO". No stubs. Ship-quality code only.

=====================================================================
PROJECT: MedSpatial AI — Volumetric Medical Intelligence Platform
=====================================================================

PRODUCT DEFINITION:
MedSpatial AI is a locally-runnable, browser-accessed platform that:
1. Ingests 2D medical images (X-ray DICOM, CT scan DICOM series, PNG/JPG radiographs)
2. Reconstructs them into interactive, semantically-segmented 3D volumetric models
3. Allows real-time layer-by-layer dissection of the 3D model (skin → muscle → bone → organ → pathology)
4. Automatically detects, highlights, and classifies abnormalities/diseases
5. Creates an interactive "medical world" around each scan where the user can orbit, slice, peel, annotate, and query
6. Provides a conversational AI interface where the user asks ANY question about the loaded scan and gets grounded, image-referenced answers
7. Can generate anatomical 3D layer models even WITHOUT input images (atlas mode) for educational comparison
8. Runs on a machine with 8GB RAM, integrated GPU, inside GitHub Codespaces or a low-end laptop

=====================================================================
SECTION 1: SYSTEM ARCHITECTURE (Novel — NOT standard GenAI)
=====================================================================

Implement a novel architecture called "Hierarchical Volumetric Intelligence Network" (HVIN) composed of FOUR interconnected subsystems:

SUBSYSTEM A — "VolEncoder" (Perception Engine)
───────────────────────────────────────────────
Purpose: Convert 2D medical image(s) into dense volumetric feature representations.

Architecture:
- Input Stage: Adaptive DICOM/image parser that normalizes any input (single X-ray, CT series, or plain image) into a standardized tensor format [B, C, H, W] or [B, C, D, H, W] for volumes.
- Encoder Backbone: Use a LIGHTWEIGHT EfficientViT-based encoder (NOT a full ViT — use the MIT Han Lab EfficientViT architecture for mobile/edge deployment). Quantize to INT8 using PyTorch's dynamic quantization for low-end hardware.
- For single 2D X-rays: Use a learned "Depth Lifting Module" — a small CNN (4 residual blocks) that predicts a pseudo-depth map from monocular radiographs using self-supervised monocular depth estimation principles (inspired by Monodepth2 but trained on chest X-ray depth proxies).
- For CT series: Stack slices along the Z-axis, apply 3D convolutions (lightweight 3D MobileNet blocks) to create the volumetric tensor.
- Output: A multi-scale feature volume F ∈ R^(C×D×H×W) where D=128, H=128, W=128 (downsampled for performance), with C=64 feature channels.

Specific implementation details:
- Use PyTorch as the framework
- All models must have ONNX export capability for CPU inference optimization
- Include automatic mixed precision (AMP) for systems with GPUs
- Include pure CPU fallback path with ONNX Runtime for systems without GPU
- Maximum model size for any single component: 50MB (use knowledge distillation if needed)

SUBSYSTEM B — "VolSegmentor" (Semantic Layer Decomposition Engine)
──────────────────────────────────────────────────────────────────
Purpose: Decompose the feature volume into semantically meaningful anatomical layers that can be individually toggled, peeled, and inspected.

Architecture:
- Use a lightweight 3D U-Net variant ("MicroUNet3D") with:
  - Encoder: 4 downsampling blocks, each = [3D Conv 3×3×3 → GroupNorm → GELU → 3D Conv 3×3×3 → GroupNorm → GELU → MaxPool 2×2×2]
  - Bottleneck: 1 block with channel attention (SE-block adapted to 3D)
  - Decoder: 4 upsampling blocks with skip connections from encoder, each = [Trilinear Upsample → Concat skip → 3D Conv → GroupNorm → GELU → 3D Conv → GroupNorm → GELU]
  - Final layer: 1×1×1 Conv → N_classes channels (N_classes = number of anatomical layers)
- Channel counts: [32, 64, 128, 256, 512] (encoder), mirrored in decoder
- Depthwise separable 3D convolutions wherever possible to reduce parameters

Anatomical Layer Classes (for chest/lung domain — the primary domain):
  Layer 0: Background/Air
  Layer 1: Skin/Subcutaneous tissue
  Layer 2: Musculature (intercostals, pectorals, diaphragm)
  Layer 3: Skeletal structure (ribs, sternum, clavicles, scapulae, vertebrae)
  Layer 4: Pleural membrane
  Layer 5: Lung parenchyma (left lung)
  Layer 6: Lung parenchyma (right lung)
  Layer 7: Bronchial tree / Airways
  Layer 8: Pulmonary vasculature
  Layer 9: Heart / Cardiac silhouette
  Layer 10: Mediastinum (great vessels, esophagus, trachea)
  Layer 11: Abnormality/Pathology regions (highlighted distinctly)

- For WITHOUT-IMAGE mode (atlas mode): Include a pre-generated atlas volume stored as a compressed NumPy array (.npz) of shape [12, 128, 128, 128] representing the 12 layer masks of a normal healthy chest anatomy. This atlas is bundled with the application.

- Output: A label volume L ∈ Z^(D×H×W) with values 0–11, PLUS a soft probability volume P ∈ R^(12×D×H×W) for smooth layer boundary rendering.

SUBSYSTEM C — "AnomalyGraph" (Pathology Detection & Reasoning Engine)
─────────────────────────────────────────────────────────────────────
Purpose: Detect, localize, classify, and explain abnormalities. This is the NOVEL ARCHITECTURE component.

Architecture — "Graph-Augmented Anomaly Transformer" (GAAT):
This is a NEW architecture that combines three paradigms:

1. Patch-level Anomaly Scoring:
   - Divide the feature volume F into non-overlapping 8×8×8 patches (= 4096 patches for 128³ volume)
   - Each patch gets a learned "normality embedding" via a small MLP (256→128→64)
   - Compare each patch embedding against a "Normal Anatomy Memory Bank" (a stored codebook of 1024 normal patch embeddings, computed offline from healthy scans using k-means on patch features)
   - Anomaly score per patch = 1 - max_cosine_similarity(patch_embedding, memory_bank)
   - This is inspired by PatchCore (Roth et al., CVPR 2022) but extended to 3D

2. Graph Neural Network for Anatomical Context:
   - Construct a graph G = (V, E) where:
     - V = the 4096 patches (nodes), each with features = [patch_embedding ∥ anomaly_score ∥ layer_class_distribution]
     - E = edges connecting spatially adjacent patches (6-connectivity in 3D) AND edges connecting patches of the same anatomical layer (layer-semantic edges)
   - Apply 3 layers of GraphSAGE (Hamilton et al.) with mean aggregation, hidden dim = 128
   - After message passing, each node has context-aware features that understand whether an anomaly is isolated or part of a pattern

3. Classification Head:
   - Global: Average-pool all node features → MLP → multi-label classification of diseases:
     [Pneumonia, Pneumothorax, Pleural Effusion, Lung Nodule/Mass, Cardiomegaly,
      Atelectasis, Consolidation, Emphysema, Fibrosis, Fracture, Tuberculosis,
      COVID-19 patterns, Normal/No Finding]
   - Local: Per-node binary classification (abnormal yes/no) + regression of severity score [0–1]
   - Output confidence calibration using temperature scaling

- Total model parameters target: < 15M (must fit in memory on 8GB system alongside other components)

SUBSYSTEM D — "MedOracle" (Conversational Intelligence Interface)
─────────────────────────────────────────────────────────────────
Purpose: Allow natural language Q&A about the loaded scan, 3D model, detected findings, and general medical knowledge.

Architecture — "Retrieval-Grounded Medical Dialogue" (RGMD):
- DO NOT use a large LLM. Instead, build a LIGHTWEIGHT system:

1. Question Understanding:
   - Use a small sentence transformer (all-MiniLM-L6-v2, 80MB) to encode the user's question into a 384-dim vector
   - Classify question intent into categories: [finding_query, anatomy_query, comparison_query, severity_query, recommendation_query, general_medical, visualization_command]

2. Context Assembly:
   - Structured context document auto-generated from current scan analysis:
     - Patient scan metadata (modality, body part, date if available)
     - List of detected findings with locations, severities, confidence scores
     - Layer-by-layer statistics (volume, density, symmetry metrics)
     - Comparison against atlas normals (deviation percentages)
   - This document is regenerated every time a new scan is loaded

3. Answer Generation:
   - Use a SMALL local language model: Microsoft Phi-3-mini (3.8B params) quantized to 4-bit using llama.cpp / GGUF format (~2GB on disk, ~2.5GB RAM)
   - If Phi-3-mini is too large for the target system, fall back to TinyLlama-1.1B-Chat quantized to 4-bit (~700MB RAM)
   - System prompt includes the structured context document
   - For visualization commands (e.g., "show me the right lung", "peel back the ribs", "highlight the nodule"), parse the intent and emit a JSON command to the frontend:
     {"action": "isolate_layer", "layer": 6, "highlight": true}
     {"action": "set_dissection_depth", "depth": 0.6}
     {"action": "focus_anomaly", "anomaly_id": 2, "zoom": true}

4. Medical Knowledge Base:
   - Bundle a compressed SQLite database (~50MB) containing:
     - 500 disease descriptions (ICD-10 mapped)
     - 200 anatomy descriptions
     - Standard radiological finding descriptors
     - Differential diagnosis decision trees for the 13 supported pathologies
   - Use FTS5 full-text search to retrieve relevant entries based on query
   - Retrieved passages are injected into the LLM context as grounding

=====================================================================
SECTION 2: 3D VISUALIZATION ENGINE (Frontend)
=====================================================================

Technology: Three.js + WebGL2 (browser-based, no install required for the viewer)
Framework: React 18 + TypeScript + Vite (fast builds, works in Codespaces)

RENDERING PIPELINE:
1. Volume Rendering:
   - Receive the volumetric data from the backend as a compressed binary ArrayBuffer (gzip'd Float32Array)
   - Implement GPU-accelerated raycasting volume rendering using a custom Three.js ShaderMaterial:
     - Vertex shader: Full-screen quad
     - Fragment shader: Ray marching through a 3D texture (THREE.Data3DTexture)
     - Support for transfer functions mapping density → color + opacity per layer
     - Maximum Intensity Projection (MIP) mode toggle
     - Physically-based lighting with ambient occlusion approximation

2. Layer Dissection System:
   - Each anatomical layer is a separate 3D texture channel
   - UI provides:
     a) Layer toggle panel (checkbox per layer with color swatch)
     b) "Dissection slider" — a depth slider that progressively peels layers from outside in (skin → muscle → bone → organ)
     c) "Clip plane" tool — user can place an arbitrary cutting plane and drag it through the volume
     d) "Exploded view" — layers separate spatially with animated transitions so each floats independently
   - All transitions animated at 60fps using requestAnimationFrame with lerped uniforms

3. Anomaly Visualization:
   - Detected abnormalities rendered as pulsing, semi-transparent colored overlays (red/orange) on the relevant voxels
   - Each anomaly has a 3D bounding box with a floating label (HTML overlay via CSS2DRenderer)
   - Clicking an anomaly opens a detail panel showing: classification, confidence, severity, textual description, and a "Ask AI about this" button

4. Interaction:
   - OrbitControls for rotation/zoom/pan
   - Scroll-wheel = zoom, right-drag = pan, left-drag = orbit
   - Touch support for tablet use
   - Keyboard shortcuts: L = toggle layers panel, D = dissection mode, A = anomaly highlight, R = reset view, F = fullscreen
   - Measurement tool: click two points → display distance in mm (if DICOM pixel spacing is available)

5. Performance Optimizations for Low-End Hardware:
   - Adaptive resolution: detect GPU capabilities via WebGL2 renderer info; if integrated GPU detected, render volume at half resolution (64³) and upscale with bilinear filtering
   - Level-of-detail: when rotating, render at ¼ resolution; on mouse-up, re-render at full resolution (progressive refinement)
   - Web Workers for data decompression (keep main thread free)
   - Texture atlas packing: combine all layer masks into a single RGBA 3D texture (4 layers per texture, 3 textures total for 12 layers)
   - Frame budget: target 30fps on Intel UHD 620; 60fps on discrete GPU



=====================================================================
SECTION 4: DATA & MODEL WEIGHTS
=====================================================================

Since this must work WITHOUT requiring users to train models, implement the following strategy:

1. PRE-TRAINED WEIGHT INITIALIZATION:
   - VolEncoder: Initialize EfficientViT backbone from ImageNet pre-trained weights (download automatically on first run from a URL, cache locally in ~/.medspatial/weights/)
   - VolSegmentor: Initialize from the Medical Segmentation Decathlon pre-trained weights for lung CT segmentation. If unavailable, use Xavier initialization and rely on the atlas-guided pseudo-labeling.
   - AnomalyGraph: Initialize the memory bank with synthetic normal patch embeddings generated from the atlas volume.

2. ATLAS DATA:
   - Include a Python script `generate_atlas.py` that creates a synthetic but anatomically plausible chest atlas volume using procedural generation:
     - Ellipsoidal lung fields with realistic curvature
     - Rib cage modeled as curved cylinders at correct anatomical positions
     - Heart as an offset ellipsoid
     - Airways as a branching tree (L-system generation)
     - Mediastinum as a central column
   - Store as `atlas_chest_128.npz` (~2MB compressed)
   - This atlas serves dual purpose: (a) atlas mode visualization (b) normal reference for anomaly detection

3. SYNTHETIC TRAINING DATA GENERATION:
   - Include a script `generate_synthetic_data.py` that creates synthetic training pairs:
     - Take atlas, apply random elastic deformations, add synthetic pathologies (spherical nodules, ground-glass regions, fluid collections)
     - Generate 1000 synthetic volumes with labels
     - This allows users to fine-tune models on their own hardware without needing real medical data

4. FALLBACK MODE:
   - If no model weights are available, the system MUST still function:
     - 3D visualization works with raw voxel data (Otsu thresholding for basic segmentation)
     - Layer decomposition uses intensity-based heuristics (Hounsfield unit ranges for CT: air < -500, lung -500 to -200, soft tissue -200 to 200, bone > 200)
     - Anomaly detection uses statistical outlier detection (z-score > 2 from atlas mean per voxel)
     - Chat still works using the knowledge base + LLM without scan-specific grounding


=====================================================================
SECTION 6: CRITICAL IMPLEMENTATION DETAILS
=====================================================================

1. MEMORY MANAGEMENT (CRITICAL for 8GB systems):
   - Never hold more than ONE volume in memory at a time
   - Use memory-mapped files (numpy.memmap) for volumes > 64MB
   - After inference, delete intermediate tensors immediately (del tensor; torch.cuda.empty_cache())
   - Set PyTorch to use no more than 2GB RAM for model weights (quantize everything)
   - Frontend: Use SharedArrayBuffer + Atomics for zero-copy volume data transfer between worker and main thread
   - Implement a "memory pressure monitor" that reduces volume resolution if available memory < 1GB

2. STARTUP SEQUENCE:
   - On first run: download model weights (~200MB total), generate atlas, build knowledge base
   - On subsequent runs: load weights lazily (only load a model when its pipeline stage is needed)
   - VolEncoder loads only when upload happens
   - AnomalyGraph loads only after segmentation completes
   - MedOracle LLM loads only when chat is first used
   - Display memory usage in the UI footer

3. GLSL VOLUME SHADER (write the COMPLETE shader):
   - Ray entry/exit calculation via AABB intersection
   - Adaptive step size: 1/128 of volume diagonal for quality, 1/64 for performance mode
   - Per-layer transfer functions defined as uniform arrays
   - Dissection: uniform float u_dissectionDepth; discard fragments where layer_order[label] > dissectionDepth * max_layer
   - Clip plane: uniform vec4 u_clipPlane; discard if dot(position, clipPlane.xyz) + clipPlane.w < 0
   - Anomaly overlay: blend anomaly_score * vec4(1,0,0, 0.5 * sin(time*3)*0.5+0.5) for pulsing effect
   - Phong-style shading using gradient-estimated normals (central differences on density)

4. CHAT INTEGRATION WITH VISUALIZATION:
   When the user types something like "show me the nodule in the right lung", the MedOracle must:
   a) Detect this is a visualization_command
   b) Map "nodule" to the specific anomaly_id from findings
   c) Map "right lung" to layer 6
   d) Emit JSON: {"action": "focus_anomaly", "anomaly_id": <id>, "isolate_layers": [6], "camera_position": <computed_position_facing_anomaly>}
   e) The frontend receives this via the chat WebSocket and executes it on the 3D viewport
   f) Simultaneously, return a text explanation of the nodule

5. OFFLINE / AIR-GAPPED OPERATION:
   - After initial setup (weight download), the system must work 100% offline
   - No API calls to external services ever
   - All AI models run locally
   - The knowledge base is local SQLite

6. DICOM HANDLING:
   - Support both single-frame and multi-frame DICOM
   - Extract: pixel data, pixel spacing, slice thickness, window center/width, modality, body part
   - Apply rescale slope/intercept to get Hounsfield Units (for CT)
   - Handle compressed transfer syntaxes (JPEG2000, JPEG-LS, RLE) via pydicom with pylibjpeg
   - For non-DICOM images: assume chest X-ray, standard projection, no pixel spacing

7. ERROR HANDLING:
   - Every API endpoint has try/except with structured error responses
   - Model inference wrapped in timeout decorator (60s for CPU, 15s for GPU)
   - Corrupt DICOM files: return meaningful error "Invalid DICOM file at byte offset X"
   - Out of memory: catch RuntimeError, reduce volume resolution by 50%, retry once
   - Frontend: all API calls have retry logic (3 attempts, exponential backoff)

=====================================================================
SECTION 7: DEPENDENCIES (pinned for reproducibility)
=====================================================================

Python (requirements.txt):
  fastapi==0.109.0
  uvicorn[standard]==0.27.0
  python-multipart==0.0.6
  pydicom==2.4.4
  pylibjpeg==2.0.0
  pylibjpeg-libjpeg==2.0.0
  numpy==1.26.3
  scipy==1.12.0
  scikit-image==0.22.0
  torch==2.2.0+cpu  (CPU-only build to save space; detect GPU at runtime and advise user to install CUDA version)
  onnxruntime==1.17.0
  transformers==4.37.2
  sentence-transformers==2.3.1
  llama-cpp-python==0.2.44
  Pillow==10.2.0
  websockets==12.0
  aiofiles==23.2.1
  python-jose==3.3.0
  pydantic==2.5.3
  httpx==0.26.0
  tqdm==4.66.1
  pytest==7.4.4
  pytest-asyncio==0.23.3

Frontend (package.json):
  react: ^18.2.0
  react-dom: ^18.2.0
  three: ^0.161.0
  @react-three/fiber: ^8.15.0  (OPTIONAL — only if it helps; raw Three.js is also fine)
  @react-three/drei: ^9.96.0   (OPTIONAL)
  zustand: ^4.5.0
  typescript: ^5.3.3
  vite: ^5.0.12
  @vitejs/plugin-react: ^4.2.1

=====================================================================
SECTION 8: DEVCONTAINER & DEPLOYMENT
=====================================================================

.devcontainer/devcontainer.json:
{
  "name": "MedSpatial AI",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "features": {
    "ghcr.io/devcontainers/features/node:1": {"version": "20"}
  },
  "postCreateCommand": "bash scripts/setup.sh",
  "forwardPorts": [8000, 5173],
  "hostRequirements": {
    "cpus": 4,
    "memory": "8gb",
    "storage": "32gb"
  }
}

docker-compose.yml — single service running both backend (port 8000) and frontend (port 5173), or a combined Nginx setup serving frontend static files and proxying /api to backend.

Makefile:
  make install — pip install + npm install + download weights + generate atlas + build KB
  make dev — run backend + frontend in dev mode (hot reload)
  make build — production build (frontend build + collect static)
  make run — production mode
  make test — pytest + vitest
  make clean — remove generated files, caches, weights

=====================================================================
SECTION 9: TESTING STRATEGY
=====================================================================

Include working tests:
1. Unit tests for DICOM parser (test with a synthetic DICOM file created in the test)
2. Unit tests for each model (test forward pass with random tensor input, verify output shapes)
3. Integration test for full pipeline (synthetic image → upload → get findings)
4. Frontend: at minimum, verify components render without crashing (React Testing Library)
5. Volume shader: include a test that renders a known volume and compares screenshot hash (optional)

=====================================================================
SECTION 10: EXECUTION ORDER
=====================================================================

Generate the code in this EXACT order (this ensures each file can reference its dependencies):

1. README.md
2. requirements.txt, package.json, tsconfig.json, vite.config.ts
3. .devcontainer/devcontainer.json, docker-compose.yml, Dockerfile, Makefile
4. scripts/setup.sh, scripts/run.sh, scripts/download_weights.py
5. backend/config.py
6. backend/utils/ (all files)
7. backend/models/ (efficient_vit.py, micro_unet3d.py, graph_sage.py, depth_lifter.py, memory_bank.py)
8. backend/core/ (hardware_manager.py, dicom_parser.py, vol_encoder.py, vol_segmentor.py, anomaly_graph.py, med_oracle.py, pipeline.py)
9. backend/atlas/ (generate_atlas.py)
10. backend/knowledge/ (build_kb.py)
11. backend/data/ (generate_synthetic.py)
12. backend/api/ (all route files)
13. backend/main.py
14. backend/tests/ (all test files)
15. frontend/src/api/ (client.ts, types.ts)
16. frontend/src/stores/ (scanStore.ts, uiStore.ts)
17. frontend/src/utils/ (volumeDecoder.ts, gpuDetect.ts)
18. frontend/src/workers/ (volumeWorker.ts)
19. frontend/src/three/ (VolumeShader.ts, VolumeRenderer.ts, LayerManager.ts, AnomalyOverlay.ts, ClipPlane.ts, ExplodedView.ts)
20. frontend/src/components/ (all components)
21. frontend/src/App.tsx, frontend/src/main.tsx
22. frontend/src/styles/ (all CSS)
23. frontend/index.html

=====================================================================
SECTION 11: ABSOLUTE REQUIREMENTS (NON-NEGOTIABLE)
=====================================================================

1. Every single file listed in the project structure MUST be generated with COMPLETE, WORKING code. No shortcuts.
2. The GLSL shaders must be COMPLETE and FUNCTIONAL — not pseudo-code.
3. The 3D rendering must actually work — volume raycasting, layer toggling, dissection slider, clip planes.
4. The AI models must have correct architectures with correct tensor dimensions throughout.
5. The chat must actually stream responses to the frontend.
6. The upload endpoint must actually process a DICOM file end-to-end.
7. The system must start with `make dev` and be usable immediately.
8. Total project size (code only, no weights) must be < 5MB.
9. All Python code must have type hints.
10. All TypeScript must be strictly typed (no `any` except where interfacing with Three.js internals).

=====================================================================
BEGIN GENERATING ALL CODE NOW. START WITH FILE 1 OF 50+.
=====================================================================
