# MedSpatial AI — Production Checklists

---

## 1. Testing Checklist

Run every item before shipping. Mark ✓ when verified.

### 1.1 ONNX Export & Validation
- [ ] `export_onnx.py` completes without errors for all 3 models
- [ ] `validate_onnx.py` reports ALL PASSED (max-abs-diff < 1e-3)
- [ ] AnomalyDetector VAE output is deterministic across 5 consecutive runs (no random sampling)
- [ ] SpatialTransformer output shape `(1, N+1, 512)` matches expected N for volume size 128³
- [ ] SegmentationNet output shape `(1, 6, 128, 128, 128)` verified
- [ ] Latency on target CPU (4-core) measured and documented
- [ ] If quantized: INT8 accuracy check passes (mean-max-diff < 0.1 vs FP32)

### 1.2 Backend Process (Nuitka .exe)
- [ ] `medspatial_backend.exe` starts in < 5 seconds on a cold run
- [ ] GET `http://127.0.0.1:8765/api/health` returns `{"status":"healthy"}`
- [ ] Backend binds **only** to `127.0.0.1` — confirmed with `netstat -an`
- [ ] Token auth: requests without `X-MedSpatial-Token` return `401`
- [ ] Token auth: requests with correct token return expected responses
- [ ] ONNX warm-up log line appears in `%APPDATA%\MedSpatialAI\logs\backend.log`
- [ ] Ready sentinel file `%APPDATA%\MedSpatialAI\logs\.backend_ready` created on startup
- [ ] Upload DICOM: end-to-end pipeline works (upload → reconstruct → mesh returned)
- [ ] Anomaly detection: heatmap and findings returned correctly
- [ ] Segmentation: integer label mask returned with correct class count
- [ ] Report generation: PDF and DOCX download without errors
- [ ] WebSocket: scan status updates arrive during reconstruction
- [ ] Backend exits cleanly when taskkilled (no zombie processes in Task Manager)
- [ ] Log file rotates at 5 MB (check after large volume processing)

### 1.3 Frontend + Tauri Shell
- [ ] Loading screen appears immediately on app launch
- [ ] Loading screen disappears once backend is healthy (< 30 s)
- [ ] Retry button works if backend fails to start
- [ ] App window opens at correct size (1440×900)
- [ ] All 3D viewer interactions work: rotate, zoom, pan
- [ ] Layer visibility toggles work
- [ ] Dissection controls work (peel, explode, isolate)
- [ ] Chat panel sends and receives messages
- [ ] Anomaly overlay renders on 3D model
- [ ] Slice viewer floats and is draggable
- [ ] Report download triggers file save dialog
- [ ] Closing the window terminates the backend process (verify in Task Manager)
- [ ] App reopens cleanly after previous session (DB and data files preserved)

### 1.4 Installer
- [ ] Installer runs on a **clean Windows 10/11 machine** (no Python, Node, etc.)
- [ ] UAC prompt appears during install (expected for perMachine mode)
- [ ] Installation completes without errors
- [ ] Desktop shortcut created and launches app
- [ ] Start Menu shortcut created and launches app
- [ ] App appears in Add/Remove Programs with correct name and version
- [ ] Uninstaller removes all installed files
- [ ] `%APPDATA%\MedSpatialAI\` data directory persists after uninstall (user data preserved)
- [ ] WebView2 bootstrapper installs automatically if not present
- [ ] Re-install over existing version works without errors

### 1.5 Performance Targets (low-end PC: 4-core CPU, 8 GB RAM)
- [ ] App launch to interactive UI: < 30 seconds
- [ ] DICOM upload (50 slices): < 10 seconds
- [ ] 3D reconstruction: < 60 seconds
- [ ] Anomaly detection (128³): < 120 seconds
- [ ] Segmentation (128³): < 90 seconds
- [ ] RAM usage during inference: < 2 GB
- [ ] RAM usage idle: < 500 MB

### 1.6 Security Verification
- [ ] `netstat -an` shows backend only on `127.0.0.1:876x` — not `0.0.0.0`
- [ ] API token is different on each app launch (confirmed in log: "Token generated:")
- [ ] API docs disabled: GET `/docs`, `/redoc`, `/openapi.json` return 404
- [ ] Stack traces not exposed in error responses (only "Internal server error" message)
- [ ] CORS blocks requests from a browser on a different origin (test with curl --origin)
- [ ] No hardcoded secrets or tokens in compiled binary (strings | grep -i secret)

---

## 2. Size Optimization Checklist

Apply these to reduce installer size. Current target: < 200 MB.

### 2.1 ONNX Models
- [ ] Use FP16 ONNX (`--fp16` flag): saves ~50% over FP32 per model
- [ ] Use INT8 quantization: saves ~75% per model — verify accuracy first
- [ ] Trim unused ONNX nodes with `onnxsim` (onnx-simplifier):
      `python -m onnxsim spatial_transformer.onnx spatial_transformer_opt.onnx`
- [ ] Remove deep supervision outputs from SegmentationNet export if unused

### 2.2 Nuitka Binary
- [ ] Confirm `--nofollow-import-to=torch` is effective: check exe size vs torch present
- [ ] Add `--nofollow-import-to=matplotlib` (imported transitively by scipy in some versions)
- [ ] Add `--nofollow-import-to=IPython` if sentence-transformers pulls it in
- [ ] Check for unexpected large inclusions: Nuitka reports all included packages with `--verbose`
- [ ] Use `--onefile` with zstandard compression (already set in build_nuitka.bat)
- [ ] If transformers/sentence-transformers pulls in too much:
      replace with a lightweight ONNX-exported sentence model via `optimum`

### 2.3 Frontend Bundle
- [ ] Verify Three.js chunk is split (check `dist/assets/vendor-three-*.js`)
- [ ] Check for accidental full lodash import: `grep -r "import _ from 'lodash'"` → use `lodash-es`
- [ ] Remove unused Three.js examples: review `@react-three/drei` usage — only import what's used
- [ ] Enable Vite `assetsInlineLimit: 10240` (already set) — verify small PNGs are inlined
- [ ] Run `npx vite-bundle-visualizer` on the dist to find unexpected large modules

### 2.4 Rust Binary
- [ ] Confirm `opt-level = "z"`, `lto = true`, `strip = true` in Cargo.toml profile.release
- [ ] Run `cargo bloat --release` to find largest contributors to binary size
- [ ] Remove unused Tauri features from `tauri = { features = [...] }`
- [ ] Consider `upx --best --lzma` on the Tauri shell binary (saves 40-60%)

### 2.5 NSIS Installer
- [ ] Use NSIS LZMA solid compression (Tauri sets this by default)
- [ ] Exclude `.pdb` debug files from the bundle
- [ ] Exclude any `.whl` or `.dist-info` directories that Nuitka may include
- [ ] Verify `resources/onnx_models/` only contains the model files needed at runtime

---

## 3. Production Risk Register and Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|-----------|
| 1 | **Port conflict** — port 8765 occupied by another app | Medium | High | `find_free_port()` in backend.rs scans 8765–8800; Tauri reads the chosen port dynamically |
| 2 | **WebView2 not installed** on target PC (Windows 7/very old Win10) | Low | Critical | NSIS installer bootstraps WebView2 Evergreen; `minimumWebview2Version` config ensures correct version |
| 3 | **ONNX model version mismatch** — model exported with opset 17, ORT 1.17 supports up to opset 18 | Low | High | Pin `onnxruntime==1.17.0` in requirements; pin `--opset 17` in export; validate in CI |
| 4 | **Nuitka includes unexpected packages** (e.g., torch pulled in transitively) | Medium | High | `--nofollow-import-to=torch` explicitly excluded; verify installer size in CI; bloat check step |
| 5 | **Cold start too slow** on 4-core CPU | Medium | Medium | ONNX warm-up in lifespan; LoadingScreen with 45 s timeout; background thread so UI isn't blocked |
| 6 | **ONNX INT8 accuracy regression** | Medium | High | quantize_onnx.py runs accuracy comparison and warns if mean-max-diff > 0.1; CI falls back to FP32 |
| 7 | **Antivirus flags Nuitka exe** (false positive — common for self-extracting stubs) | High | Medium | Submit to AV vendors for whitelisting; code-sign with a trusted cert (DigiCert/Sectigo); use `--onefile-no-splash` to avoid stub detection |
| 8 | **Backend process orphaned after crash** | Low | Medium | `start_health_monitor()` in Rust detects unresponsiveness in 15 s and restarts; `taskkill /F /T` on shutdown kills entire process tree |
| 9 | **SQLite DB corruption** on unclean shutdown | Low | Medium | WAL journal mode (set in database.py `PRAGMA journal_mode=WAL`); DB in APPDATA survives reinstall |
| 10 | **Missing model weights in release build** | High | Critical | CI step downloads from LFS/secrets URL; `export_onnx.py` logs a warning if weights missing and exports random-weight model; validate step catches large numerical errors |
| 11 | **CORS bypass via local network** | Low | High | `HOST=127.0.0.1` blocks all non-loopback traffic at the OS level; CORS middleware as defence-in-depth |
| 12 | **Token exposed in process environment** | Low | Medium | Token is in env var, not a file; Windows env vars are process-scoped; parent Tauri process owns env; no token logging in production |
| 13 | **Large DICOM series OOM on 4 GB RAM machine** | Medium | High | VolumeProcessor resizes to 128³ before inference; add explicit memory guard in dicom_service with a 500 MB limit check |
| 14 | **einops not found at runtime** (Nuitka may miss it) | Medium | Medium | Explicitly listed in `--follow-import-to=einops` and `--include-package=einops` in build_nuitka.bat |
| 15 | **tauri.conf.json5 comments break strict JSON parser** | Low | High | CI step (Strip comments from tauri.conf.json5) converts to valid JSON before `tauri build` runs |

---

## 4. Step-by-Step: Build in GitHub Codespaces and Run on Windows

### Prerequisites
- A GitHub account with this repo
- A Windows PC (Windows 10 64-bit or later) to run the final installer

### Steps

**Step 1 — Open Codespaces**
```
1. Go to your GitHub repo → Code → Codespaces → New codespace
2. Wait for the environment to initialise (2–3 minutes)
```

**Step 2 — (One-time) Add model weights**
```bash
# If you have trained .pth files, copy them to models/
cp /path/to/spatial_transformer.pth medspatial-ai/models/
cp /path/to/segmentation_net.pth     medspatial-ai/models/
cp /path/to/anomaly_detector.pth     medspatial-ai/models/
# Or skip this — random weights will be exported (for testing the pipeline only)
```

**Step 3 — Trigger the workflow**
```
Option A (manual dispatch):
  1. Go to Actions tab → "Build Desktop App" → Run workflow
  2. Choose options (quantize, fp16, volume_size) → Run

Option B (tag-based release):
  git tag v1.0.0
  git push origin v1.0.0
  → Workflow triggers automatically
```

**Step 4 — Download the installer**
```
1. Go to Actions → latest "Build Desktop App" run
2. Under Artifacts → click "MedSpatialAI-Windows-Installer"
3. Download the .zip → extract → find "MedSpatial AI_1.0.0_x64-setup.exe"
```

**Step 5 — Install on Windows**
```
1. Copy the .exe to your Windows PC
2. Right-click → Run as Administrator
3. Follow the installer wizard
4. Launch "MedSpatial AI" from the desktop shortcut
5. The loading screen appears; AI models initialise in ~10–20 seconds
6. Upload a DICOM scan and start working
```

**Step 6 — Verify the install**
```powershell
# In Windows PowerShell — verify backend is running correctly
# (only works while the app is open)
Invoke-WebRequest http://127.0.0.1:8765/api/health
# Expected: {"status":"healthy","app":"MedSpatial AI","version":"1.0.0"}

# Check logs
Get-Content "$env:APPDATA\MedSpatialAI\logs\backend.log" -Tail 50
```
