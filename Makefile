.PHONY: setup install serve check test lint format docker-up docker-down clean help

help:
	@echo "Fact-Checking Agent — Available commands:"
	@echo ""
	@echo "  make setup       Full local setup (venv + deps + model download)"
	@echo "  make install     Install Python dependencies only"
	@echo "  make serve       Start the API server (port 8000)"
	@echo "  make check       Verify a claim: make check CLAIM='...'"
	@echo "  make ingest      Force feed ingestion into vector index"
	@echo "  make demo        Run demo against running server"
	@echo "  make test        Run test suite with coverage"
	@echo "  make lint        Run ruff + mypy"
	@echo "  make format      Auto-format with black"
	@echo "  make docker-up   Start full stack with Docker Compose"
	@echo "  make docker-down Stop Docker Compose stack"
	@echo "  make clean       Remove caches and build artifacts"

setup:
	bash scripts/setup.sh

install:
	pip install -r requirements.txt

serve:
	python -m src.main serve

check:
	python -m src.main check "$(CLAIM)"

ingest:
	python -m src.main ingest

demo:
	python scripts/demo.py

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -x --no-cov

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ scripts/

docker-up:
	docker-compose up --build -d
	@echo "Services starting... API will be ready in ~90s at http://localhost:8000"

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ .pytest_cache/ .ruff_cache/ .mypy_cache/
	rm -rf data/diskcache/
