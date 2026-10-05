.PHONY: help install setup train field import-data dev-api dev-ui test lint build up down samples clean
PY ?= python3

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:         ## Python (incl. CPU PyTorch for training) + Node dependencies
	cd backend && $(PY) -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt
	cd frontend && npm ci

setup: train field  ## train models + build the field-dataset store (one-off, ~2-3 min)

train:           ## train the card CNN + live-twin thermal model (~20 s, CPU)
	cd backend && $(PY) -m scripts.train_models

field:           ## build the field-dataset store and models from data/field (~1.5-3 min)
	cd backend && $(PY) -m scripts.build_field_store

import-data:     ## import new dataset CSVs: make import-data SRC=/path/to/csv/folder
	cd backend && $(PY) -m scripts.import_dataset --src $(SRC) && $(PY) -m scripts.build_field_store

samples:         ## regenerate the dirty demo CSVs in data/samples
	cd backend && $(PY) -m scripts.generate_samples

dev-api:         ## API on http://localhost:8000 (docs at /docs)
	cd backend && $(PY) -m uvicorn app.main:app --reload --port 8000

dev-ui:          ## dashboard on http://localhost:5173
	cd frontend && npm run dev

test:            ## backend + frontend tests
	cd backend && $(PY) -m pytest -q
	cd frontend && npm run typecheck && npm test

lint:
	cd backend && $(PY) -m ruff check .

build:           ## production build of the dashboard (frontend/dist)
	cd frontend && npm run build

up:              ## both containers: http://localhost:8080
	docker compose up --build

down:
	docker compose down

clean:
	rm -rf frontend/dist backend/.pytest_cache backend/.ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
