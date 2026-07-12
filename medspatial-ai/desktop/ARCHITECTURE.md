# MedSpatial AI — Enterprise Desktop Architecture

## Overview

This document describes the production-ready Windows desktop packaging architecture
for MedSpatial AI. The final product is a single `.exe` NSIS installer that bundles
everything the end-user needs — no Python, Node.js, PyTorch, or CUDA required.

---

## Stack

| Layer | Technology | Role |
|-------|-----------|------|
| Desktop Shell | Tauri 2 (Rust) | Window management, OS integration, sidecar process lifecycle |
| Frontend | React 18 + Vite 5 | UI — compiled to static HTML/JS/CSS, served from Tauri |
| Backend | FastAPI + Uvicorn | Local HTTP server on 127.0.0.1, compiled by Nuitka |
| AI Inference | ONNX Runtime CPU | Replaces PyTorch — ~10x smaller, no CUDA required |
| Installer | NSIS via Tauri bundler | One-click Windows installer with uninstaller |

---

## Deployment Topology (Desktop)

```
┌─────────────────────────────────────────────────────────────────┐
│  Windows Process: MedSpatial AI (Tauri/WebView2)                │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Tauri Rust Core                                         │   │
│  │  - Spawns backend sidecar on startup                    │   │
│  │  - Monitors sidecar health every 5s                     │   │
│  │  - Kills sidecar on app close                           │   │
│  │  - Injects API base URL into frontend via tauri://      │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │  IPC                              │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │  WebView2 (Frontend)                                     │   │
│  │  React 18 + Three.js + Vite build (static assets)      │   │
│  │  Talks to http://127.0.0.1:{PORT}/api/*                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Sidecar: medspatial_backend.exe (Nuitka compiled)      │   │
│  │  FastAPI + Uvicorn on 127.0.0.1:8765                    │   │
│  │  ONNX Runtime CPU inference (no PyTorch)                │   │
│  │  SQLite database in %APPDATA%\MedSpatialAI\             │   │
│  │  Logs in %APPDATA%\MedSpatialAI\logs\                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Production Folder Structure

```
medspatial-ai/
├── backend/                          # Original FastAPI backend (dev)
│   └── app/
├── frontend/                         # Original React frontend (dev)
│   └── src/
│
└── desktop/                          # ← ALL PACKAGING CODE LIVES HERE
    ├── ARCHITECTURE.md               # This file
    │
    ├── scripts/                      # Build and tooling scripts
    │   ├── export_onnx.py            # PyTorch → ONNX export (all 3 models)
    │   ├── validate_onnx.py          # ONNX validation + PyTorch comparison
    │   ├── quantize_onnx.py          # Optional INT8 quantization
    │   └── build_nuitka.bat          # Nuitka compile command
    │
    ├── backend_prod/                 # Production backend (ONNX-powered)
    │   ├── main_prod.py              # Production FastAPI entry point
    │   ├── config_prod.py            # Production config (portable paths)
    │   ├── onnx_inference.py         # ONNX Runtime inference engine
    │   ├── requirements_prod.txt     # Minimal deps (no torch)
    │   └── app/                     # Symlink / copy of backend/app/
    │       └── ...                  # (services, api, core — unchanged)
    │
    ├── tauri/                        # Tauri desktop shell
    │   ├── package.json              # Tauri JS deps
    │   ├── vite.config.ts            # Vite config for Tauri
    │   ├── src/                      # Frontend source (copy from frontend/src)
    │   │   ├── main.jsx
    │   │   ├── App.jsx               # + loading screen injected
    │   │   └── tauri_api.js          # Tauri ↔ backend bridge
    │   └── src-tauri/               # Tauri Rust crate
    │       ├── Cargo.toml
    │       ├── tauri.conf.json       # Installer config, window, sidecar
    │       ├── build.rs
    │       └── src/
    │           ├── main.rs           # App entry
    │           └── backend.rs        # Sidecar launcher + health check
    │
    ├── build_all.bat                 # Master Windows build script
    └── .github/
        └── workflows/
            └── build_desktop.yml     # GitHub Codespaces CI build
```

---

## Security Model

- Backend binds exclusively to `127.0.0.1` — never `0.0.0.0`
- All API calls carry a per-session HMAC token generated at startup
- CORS allows only `tauri://localhost` and `http://127.0.0.1`
- Debug endpoints disabled in production (`DEBUG=false`)
- No telemetry, no outbound network calls in production build

---

## Size Budget

| Component | Estimated Size |
|-----------|---------------|
| Nuitka compiled backend | ~80–120 MB |
| ONNX model files (3 models, FP16) | ~30–60 MB |
| Frontend static assets | ~5–8 MB |
| Tauri shell + WebView2 bootstrap | ~3–5 MB |
| ONNX Runtime DLL | ~12 MB |
| **Total installer** | **~150–200 MB** |

Compare to PyTorch baseline: ~2.5 GB. Savings: **~92%**.

---

## Build Environments

| Step | Where to run |
|------|-------------|
| ONNX export + validation | GitHub Codespaces (Python 3.11) |
| Nuitka compilation | GitHub Codespaces (Linux → cross-compile) OR Windows runner |
| Tauri NSIS build | GitHub Codespaces with `windows-latest` runner |
| Full end-to-end | `build_desktop.yml` GitHub Actions workflow |
