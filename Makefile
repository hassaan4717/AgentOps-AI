# AgentOps AI Platform — local demo commands
# Run: make help

.PHONY: help dev backend frontend install install-backend install-frontend \
	venv check check-env lint format test clean kill security-check

PYTHON := python3
PIP := pip3
VENV := venv
VENV_BIN := $(VENV)/bin

help:
	@echo "AgentOps AI Platform (demo)"
	@echo ""
	@echo "  make venv              Create Python virtualenv"
	@echo "  make install           Install Python + Node deps"
	@echo "  make dev               Start backend + frontend"
	@echo "  make backend           Start API on :8000"
	@echo "  make frontend          Start UI on :3000"
	@echo "  make test              Run pytest"
	@echo "  make lint              Run Ruff + ESLint"
	@echo "  make check             Verify env + imports"
	@echo "  make security-check    Scan for secrets"
	@echo "  make kill              Free ports 8000 / 3000"
	@echo "  make clean             Remove caches / build artifacts"
	@echo ""

dev: check-env
	@echo "Starting AgentOps AI Platform..."
	@echo "  Backend:  http://localhost:8000"
	@echo "  Frontend: http://localhost:3000"
	@echo "  Docs:     http://localhost:8000/docs"
	@trap 'kill 0' INT; \
	$(MAKE) backend & \
	$(MAKE) frontend & \
	wait

backend: check-env
	@echo "Starting FastAPI on http://localhost:8000..."
	@cd "$(CURDIR)" && \
	export $$(grep -v '^#' .env.local | grep -v '^$$' | xargs) 2>/dev/null; \
	PYTHONPATH="$(CURDIR)/src:$(CURDIR):$$PYTHONPATH" \
	if [ -d "$(VENV)" ]; then \
		$(VENV_BIN)/uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000; \
	else \
		uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000; \
	fi

frontend:
	@echo "Starting Next.js on http://localhost:3000..."
	@cd frontend && npm run dev

install: install-backend install-frontend
	@echo "All dependencies installed. Run: make dev"

install-backend:
	@echo "Installing Python dependencies..."
	@if [ -d "$(VENV)" ]; then \
		$(VENV_BIN)/pip install -r requirements.txt; \
	else \
		$(PIP) install -r requirements.txt; \
	fi

install-frontend:
	@echo "Installing Node dependencies..."
	@cd frontend && npm install

venv:
	@echo "Creating virtualenv at ./$(VENV)..."
	@$(PYTHON) -m venv $(VENV)
	@echo "Activate with: source $(VENV_BIN)/activate"

check-env:
	@if [ ! -f ".env.local" ]; then \
		echo "Missing .env.local — copy .env.example and add GOOGLE_API_KEY"; \
		echo "  cp .env.example .env.local"; \
		exit 1; \
	fi

check: check-env
	@export $$(grep -v '^#' .env.local | grep -v '^$$' | xargs) 2>/dev/null; \
	python3 -c "import os; k=os.environ.get('GOOGLE_API_KEY',''); print('GOOGLE_API_KEY set' if k else 'GOOGLE_API_KEY missing')"
	@python3 -c "import fastapi; print('fastapi ok')" 2>/dev/null || echo "fastapi missing (make install)"
	@python3 -c "import langgraph; print('langgraph ok')" 2>/dev/null || echo "langgraph missing"

kill:
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@lsof -ti:3000 | xargs kill -9 2>/dev/null || true
	@echo "Ports 8000 and 3000 cleared"

lint:
	@echo "Python (ruff)..."
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check . --exclude venv --exclude frontend --exclude .venv; \
	else \
		echo "ruff not installed — pip install ruff"; \
	fi
	@echo "TypeScript (eslint)..."
	@cd frontend && npm run lint

format:
	@if command -v ruff >/dev/null 2>&1; then \
		ruff format . --exclude venv --exclude frontend --exclude .venv; \
	else \
		echo "ruff not installed — pip install ruff"; \
	fi

test:
	@PYTHONPATH="$(CURDIR)/src:$(CURDIR):$$PYTHONPATH" \
	OFFLINE_MODE=1 \
	pytest tests/ -v

security-check:
	@./scripts/check-secrets.sh

clean:
	@rm -rf frontend/.next frontend/out .pytest_cache .ruff_cache
	@rm -rf memory/chroma_db
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete"
