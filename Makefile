# MukhaSpectra developer commands. Research prototype — not for clinical use.
PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup setup-python setup-frontend test test-pipeline test-backend test-frontend coverage \
        lint format schema schema-check sample-data demo-report dev-backend dev-frontend e2e

setup: setup-python setup-frontend

setup-python:
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e "pipeline[raw,onnx,dev]" -e "backend[dev]"

setup-frontend:
	cd frontend && npm ci

test: test-pipeline test-backend test-frontend

test-pipeline:
	$(PY) -m pytest pipeline/tests -q

test-backend:
	$(PY) -m pytest backend/tests -q

test-frontend:
	cd frontend && npm test

coverage:
	$(PY) -m pytest pipeline/tests -q --cov=mukha_pipeline --cov-report=term-missing --cov-fail-under=80

lint:
	.venv/bin/ruff check pipeline backend
	.venv/bin/black --check pipeline backend
	cd frontend && npm run lint && npm run typecheck

format:
	.venv/bin/ruff check --fix pipeline backend
	.venv/bin/black pipeline backend
	cd frontend && npm run format

schema:
	$(PY) -m mukha_pipeline.schema.export
	cd frontend && npm run gen:types

schema-check:
	$(PY) -m mukha_pipeline.schema.export --check

sample-data:
	$(PY) -m mukha_pipeline.synthetic sample_data/synthetic

dev-backend:
	cd backend && ../.venv/bin/uvicorn app.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

e2e:
	cd frontend && npm run e2e
