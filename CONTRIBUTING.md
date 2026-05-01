# Contributing

Thanks for checking out this demo project.

## Setup

```bash
git clone https://github.com/guptaom31619-prog/AgentOps-AI-Platform.git
cd AgentOps-AI-Platform

make venv && source venv/bin/activate
cp .env.example .env.local   # add GOOGLE_API_KEY
make install
make dev
```

## Before opening a PR

```bash
make lint
make test
./scripts/check-secrets.sh
```

## Guidelines

- Keep changes focused — avoid unrelated refactors
- Never commit `.env.local`, API keys, or `memory/chroma_db/`
- Prefer small PRs with a clear description
- CI must pass before merge

## Project layout

| Path | Role |
|------|------|
| `backend/` | FastAPI routers |
| `src/agentops_ai_platform/` | Agents + LangGraph |
| `frontend/` | Next.js UI |
| `memory/` | Memory store + vector DB |
| `observability/` | LangSmith / Langfuse |
| `tests/` | Pytest suite |
