# AgentOps AI Platform

A **demo multi-agent AI system** built with LangGraph, FastAPI, and Next.js.

Supervisor, Research, Execution, and Evaluator agents collaborate to complete a user goal — with optional semantic memory and observability.

[![CI](https://github.com/guptaom31619-prog/AgentOps-AI-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/guptaom31619-prog/AgentOps-AI-Platform/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Demo project** — designed for local exploration and learning, not production deployment.

---

## Features

- **Multi-agent orchestration** — LangGraph workflow: Supervisor → Research → Execution → Evaluator
- **Web search tool** — DuckDuckGo search for research context
- **Semantic memory** — ChromaDB vector store for reusing past high-quality results
- **Optional observability** — LangSmith + Langfuse tracing (off by default)
- **Self-evaluation** — quality scoring with retry on failure
- **Full-stack demo UI** — Next.js frontend + FastAPI backend

---

## Architecture

```
┌─────────────────────────────────────┐
│         Frontend (Next.js)          │
└──────────────────┬──────────────────┘
                   │ HTTP
┌──────────────────┴──────────────────┐
│          Backend (FastAPI)          │
└──────────────────┬──────────────────┘
                   │
         ┌─────────┴─────────┐
         │   LangGraph flow  │
         └─────────┬─────────┘
                   │
     ┌─────────────┼─────────────┐
     ▼             ▼             ▼
 Supervisor    Research      Execution
     │             │             │
     └─────────────┴──────┬──────┘
                          ▼
                      Evaluator
```

---

## Project structure

```
AgentOps-AI-Platform/
├── backend/                 # FastAPI app + routers
├── frontend/                # Next.js 14 UI
├── src/agentops_ai_platform/
│   ├── agents/              # Supervisor, Research, Execution, Evaluator
│   └── graphs/              # LangGraph workflow
├── memory/                  # JSON + ChromaDB memory layer
├── observability/           # LangSmith / Langfuse helpers
├── tools/                   # Web search tool
├── tests/                   # API smoke tests (CI)
├── scripts/                 # Secret-scan helper
├── .github/workflows/       # CI pipeline
├── .env.example             # Env var template
├── Makefile                 # Local commands
└── requirements.txt
```

---

## Quick start

### Prerequisites

- Python **3.10+**
- Node.js **18+**
- A free [Google Gemini API key](https://aistudio.google.com/apikey)

### Setup

```bash
git clone https://github.com/guptaom31619-prog/AgentOps-AI-Platform.git
cd AgentOps-AI-Platform

make venv
source venv/bin/activate

cp .env.example .env.local
# Add your GOOGLE_API_KEY to .env.local

make install
make dev
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |

### Offline mode (no LLM calls)

```bash
# In .env.local
OFFLINE_MODE=1
```

Useful for UI / API testing without burning API quota.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Gemini API key |
| `GEMINI_MODEL` | No | Supervisor model (default `gemini-2.0-flash`) |
| `GEMINI_RESEARCH_MODEL` | No | Research agent model |
| `GEMINI_EXECUTION_MODEL` | No | Execution agent model |
| `GEMINI_EVALUATOR_MODEL` | No | Evaluator agent model |
| `OFFLINE_MODE` | No | `1` skips live LLM calls |
| `OBSERVABILITY_ENABLED` | No | `1` enables LangSmith + Langfuse |
| `LANGSMITH_API_KEY` | No | LangSmith tracing |
| `LANGSMITH_PROJECT` | No | LangSmith project name |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | No | Langfuse keys |
| `LANGFUSE_HOST` | No | Langfuse host URL |

See [`.env.example`](.env.example) for the full template.

---

## Useful commands

```bash
make help      # List all commands
make dev       # Backend + frontend
make backend   # API only (port 8000)
make frontend  # UI only (port 3000)
make test      # Run pytest
make lint      # Ruff + ESLint
make check     # Verify env + deps
make kill      # Free ports 8000 / 3000
```

---

## CI

Every push and pull request to `main` runs:

1. **Security scan** — blocks committed secrets
2. **Python lint** — Ruff
3. **TypeScript lint** — ESLint
4. **Backend tests** — pytest (offline mode)
5. **Frontend build** — `next build`

Workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

---

## Tech stack

| Layer | Tech |
|-------|------|
| Agents | LangChain, LangGraph, Google Gemini |
| API | FastAPI, Pydantic, Uvicorn |
| Memory | ChromaDB |
| Tools | DuckDuckGo Search |
| Frontend | Next.js 14, React 18, TypeScript |
| Observability | LangSmith, Langfuse (optional) |

---

## License

MIT — see [LICENSE](LICENSE).
