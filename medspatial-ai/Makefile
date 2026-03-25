.PHONY: install dev build run test clean atlas weights

# ─── Variables ────────────────────────────────────────────────────
BACKEND_DIR := backend
FRONTEND_DIR := frontend
DATA_DIR     := data
MODELS_DIR   := models
PYTHON       := python
PIP          := pip
NPM          := npm

# ─── Setup ────────────────────────────────────────────────────────
install:
	@echo "🔧 Installing backend dependencies..."
	cd $(BACKEND_DIR) && $(PIP) install -r requirements.txt
	@echo "📦 Installing frontend dependencies..."
	cd $(FRONTEND_DIR) && $(NPM) install
	@echo "⚙️  Generating atlas volume..."
	$(MAKE) atlas
	@echo "🧠 Downloading model weights..."
	$(MAKE) weights
	@echo "🗄️  Building knowledge base..."
	$(MAKE) kb
	@echo "✅ Setup complete! Run 'make dev' to start."

# ─── Atlas & Data ─────────────────────────────────────────────────
atlas:
	@echo "🫁 Generating chest atlas..."
	@mkdir -p $(DATA_DIR)
	cd $(BACKEND_DIR) && $(PYTHON) -m app.atlas.generate_atlas \
		--size 128 --output atlas_chest_128.npz --output-dir ../$(DATA_DIR)

synthetic:
	@echo "🔬 Generating synthetic training data..."
	cd $(BACKEND_DIR) && $(PYTHON) -m app.data.generate_synthetic_data \
		--atlas ../$(DATA_DIR)/atlas_chest_128.npz \
		--output ../$(DATA_DIR)/synthetic_training \
		--num-samples 1000

# ─── Model Weights ────────────────────────────────────────────────
weights:
	@echo "⬇️  Downloading pre-trained weights..."
	@mkdir -p $(MODELS_DIR)
	cd $(BACKEND_DIR) && $(PYTHON) scripts/download_weights.py --output-dir ../$(MODELS_DIR)

# ─── Knowledge Base ───────────────────────────────────────────────
kb:
	@echo "🗄️  Building medical knowledge base..."
	@mkdir -p $(DATA_DIR)
	cd $(BACKEND_DIR) && $(PYTHON) -m app.knowledge.build_kb \
		--output ../$(DATA_DIR)/medical_kb.sqlite

# ─── Development ──────────────────────────────────────────────────
dev:
	@echo "🚀 Starting MedSpatial AI in development mode..."
	@echo "   Backend:  http://localhost:8000"
	@echo "   Frontend: http://localhost:5173"
	@echo "   API Docs: http://localhost:8000/docs"
	$(MAKE) -j2 dev-backend dev-frontend

dev-backend:
	cd $(BACKEND_DIR) && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd $(FRONTEND_DIR) && $(NPM) run dev

# ─── Production Build ─────────────────────────────────────────────
build:
	@echo "🏗️  Building production frontend..."
	cd $(FRONTEND_DIR) && $(NPM) run build
	@echo "📦 Frontend built to frontend/dist/"

run:
	@echo "🚀 Starting MedSpatial AI in production mode..."
	cd $(BACKEND_DIR) && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2

# ─── Docker ───────────────────────────────────────────────────────
docker-build:
	docker-compose build

docker-up:
	docker-compose up

docker-down:
	docker-compose down

# ─── Testing ──────────────────────────────────────────────────────
test:
	@echo "🧪 Running backend tests..."
	cd $(BACKEND_DIR) && $(PYTHON) -m pytest tests/ -v --tb=short
	@echo "🧪 Running frontend tests..."
	cd $(FRONTEND_DIR) && $(NPM) run test 2>/dev/null || echo "No frontend tests configured"

test-backend:
	cd $(BACKEND_DIR) && $(PYTHON) -m pytest tests/ -v --tb=long

# ─── Cleanup ──────────────────────────────────────────────────────
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf $(DATA_DIR)/uploads/* $(DATA_DIR)/volumes/* $(DATA_DIR)/meshes/* $(DATA_DIR)/analysis/*
	rm -rf $(BACKEND_DIR)/__pycache__ $(BACKEND_DIR)/app/**/__pycache__
	rm -rf $(FRONTEND_DIR)/dist $(FRONTEND_DIR)/node_modules/.vite
	@echo "✅ Cleaned."

clean-all: clean
	@echo "⚠️  Removing models and generated data..."
	rm -rf $(MODELS_DIR)/* $(DATA_DIR)/atlas_chest_128.npz $(DATA_DIR)/synthetic_training

# ─── Utilities ────────────────────────────────────────────────────
check-gpu:
	$(PYTHON) -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

check-memory:
	$(PYTHON) -c "import psutil; mem = psutil.virtual_memory(); print(f'RAM: {mem.total/1e9:.1f}GB total, {mem.available/1e9:.1f}GB available')"

info:
	@echo "MedSpatial AI — Project Info"
	@echo "  Backend:   http://localhost:8000"
	@echo "  Frontend:  http://localhost:5173"
	@echo "  API Docs:  http://localhost:8000/docs"
	@echo "  ReDoc:     http://localhost:8000/redoc"
