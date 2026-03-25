#!/usr/bin/env bash
# MedSpatial AI — First-Run Setup Script
# Run this once to install all dependencies and initialize the system.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "   MedSpatial AI — Setup"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC}   $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERR]${NC}  $1"; exit 1; }

# ── Python environment ───────────────────────────────────────────
info "Checking Python version..."
python --version 2>/dev/null || python3 --version 2>/dev/null || error "Python not found"

info "Installing backend Python dependencies..."
cd "$PROJECT_ROOT/backend"
pip install --upgrade pip -q
pip install -r requirements.txt -q
success "Python dependencies installed"

# ── Node.js / frontend ───────────────────────────────────────────
info "Checking Node.js..."
node --version 2>/dev/null || error "Node.js not found. Install Node 20+."

info "Installing frontend dependencies..."
cd "$PROJECT_ROOT/frontend"
npm install --silent
success "Frontend dependencies installed"

# ── Data directories ─────────────────────────────────────────────
info "Creating data directories..."
cd "$PROJECT_ROOT"
mkdir -p data/uploads data/volumes data/meshes data/analysis data/synthetic_training
mkdir -p models
success "Directories created"

# ── Atlas generation ─────────────────────────────────────────────
if [ ! -f "$PROJECT_ROOT/data/atlas_chest_128.npz" ]; then
    info "Generating chest atlas (first time, ~30s)..."
    cd "$PROJECT_ROOT/backend"
    python -m app.atlas.generate_atlas \
        --size 128 \
        --output atlas_chest_128.npz \
        --output-dir ../data
    success "Atlas generated: data/atlas_chest_128.npz"
else
    success "Atlas already exists, skipping"
fi

# ── Model weights ────────────────────────────────────────────────
info "Checking model weights..."
if [ -f "$PROJECT_ROOT/backend/scripts/download_weights.py" ]; then
    cd "$PROJECT_ROOT/backend"
    python scripts/download_weights.py --output-dir ../models
    success "Weights ready"
else
    warn "download_weights.py not found, skipping. Models will use random initialization."
fi

# ── Knowledge base ───────────────────────────────────────────────
if [ ! -f "$PROJECT_ROOT/data/medical_kb.sqlite" ]; then
    info "Building medical knowledge base (~5s)..."
    cd "$PROJECT_ROOT/backend"
    python -m app.knowledge.build_kb --output ../data/medical_kb.sqlite
    success "Knowledge base built: data/medical_kb.sqlite"
else
    success "Knowledge base already exists, skipping"
fi

# ── System check ─────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "   System Check"
echo "=================================================="
python - <<'PYEOF'
import sys, platform
print(f"Python:    {sys.version.split()[0]}")
print(f"Platform:  {platform.system()} {platform.machine()}")

try:
    import torch
    cuda = torch.cuda.is_available()
    print(f"PyTorch:   {torch.__version__} ({'CUDA ' + torch.version.cuda if cuda else 'CPU-only'})")
    if cuda:
        print(f"GPU:       {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch:   NOT INSTALLED")

try:
    import pydicom
    print(f"pydicom:   {pydicom.__version__}")
except ImportError:
    print("pydicom:   NOT INSTALLED")

try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"RAM:       {mem.total/1e9:.1f} GB total, {mem.available/1e9:.1f} GB available")
except ImportError:
    pass
PYEOF

echo ""
success "Setup complete!"
echo ""
echo -e "${GREEN}Run the platform:${NC}"
echo "  make dev          — start both backend and frontend"
echo "  make dev-backend  — backend only (port 8000)"
echo "  make dev-frontend — frontend only (port 5173)"
echo ""
echo -e "${BLUE}URLs:${NC}"
echo "  Frontend:  http://localhost:5173"
echo "  Backend:   http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo ""
