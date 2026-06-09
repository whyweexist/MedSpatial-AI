.DEFAULT_GOAL := help

.PHONY: help install install-backend install-frontend dev dev-backend dev-frontend \
	build run run-backend docker-build docker-up docker-down test test-backend \
	test-frontend atlas kb weights doctor check-gpu check-memory

BACKEND_DIR ?= backend
FRONTEND_DIR ?= frontend
DATA_DIR ?= data
MODELS_DIR ?= models
PYTHON ?= python
NPM ?= npm
COMPOSE ?= docker compose

help:
	@echo "MedSpatial AI / AWM-JGEM"
	@echo "  make install       Install local dependencies (no model downloads)"
	@echo "  make dev           Start backend and frontend development servers"
	@echo "  make run           Build and run the complete Docker stack"
	@echo "  make run-backend   Start the local production-style backend"
	@echo "  make test          Run backend tests and a frontend production build"
	@echo "  make doctor        Check required tools and runtime configuration"
	@echo "  make weights       Explicitly download optional model weights"

install: install-backend install-frontend
	@echo "Local dependencies installed. Optional assets: make atlas kb weights"

install-backend:
	cd $(BACKEND_DIR) && $(PYTHON) -m pip install -r requirements.txt

install-frontend:
	cd $(FRONTEND_DIR) && $(NPM) ci

dev:
	@echo "Backend:  http://localhost:8000"
	@echo "Frontend: http://localhost:5173"
	@echo "API docs: http://localhost:8000/docs"
	$(MAKE) -j2 dev-backend dev-frontend

dev-backend:
	cd $(BACKEND_DIR) && $(PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd $(FRONTEND_DIR) && $(NPM) run dev -- --host 0.0.0.0

build:
	cd $(FRONTEND_DIR) && $(NPM) run build

run:
	$(COMPOSE) up --build

run-backend:
	cd $(BACKEND_DIR) && $(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

docker-build:
	$(COMPOSE) build

docker-up:
	$(COMPOSE) up

docker-down:
	$(COMPOSE) down

test: test-backend test-frontend

test-backend:
	cd $(BACKEND_DIR) && $(PYTHON) -m pytest tests -v --tb=short

test-frontend:
	cd $(FRONTEND_DIR) && $(NPM) run build

atlas:
	cd $(BACKEND_DIR) && $(PYTHON) -m app.atlas.generate_atlas --size 128 \
		--output atlas_chest_128.npz --output-dir ../$(DATA_DIR)

kb:
	cd $(BACKEND_DIR) && $(PYTHON) -m app.knowledge.build_kb \
		--output ../$(DATA_DIR)/medical_kb.sqlite

weights:
	cd $(BACKEND_DIR) && $(PYTHON) scripts/download_weights.py --output-dir ../$(MODELS_DIR)

doctor:
	@$(PYTHON) --version
	@$(PYTHON) -c "import fastapi, numpy, torch; print('Backend imports: OK'); print('Torch device:', 'cuda' if torch.cuda.is_available() else 'cpu')"
	@$(NPM) --version
	@$(COMPOSE) version

check-gpu:
	$(PYTHON) -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

check-memory:
	$(PYTHON) -c "import psutil; m=psutil.virtual_memory(); print(f'RAM: {m.total/1e9:.1f} GB total, {m.available/1e9:.1f} GB available')"
