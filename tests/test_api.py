"""API smoke tests for AgentOps AI Platform (offline-safe)."""

from __future__ import annotations

import os

from fastapi.testclient import TestClient

# Keep CI deterministic — no live LLM calls.
os.environ.setdefault("OFFLINE_MODE", "1")

from backend.main import app  # noqa: E402

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["service"] == "agentops-ai-backend"
    assert "version" in payload


def test_root_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert "docs" in payload
    assert payload["health"] == "/health"


def test_history_endpoint() -> None:
    response = client.get("/history")
    assert response.status_code == 200
    payload = response.json()
    assert "memories" in payload
    assert "total" in payload
    assert isinstance(payload["memories"], list)
    assert isinstance(payload["total"], int)


def test_run_rejects_empty_goal() -> None:
    response = client.post("/run", json={"goal": ""})
    assert response.status_code == 422


def test_run_rejects_short_goal() -> None:
    response = client.post("/run", json={"goal": "ab"})
    assert response.status_code == 422


def test_run_offline_mode() -> None:
    response = client.post(
        "/run",
        json={"goal": "Write a short summary of multi-agent systems"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "final_output" in payload
    assert "evaluation" in payload
    assert "memory_used" in payload
    assert isinstance(payload["final_output"], str)
    assert payload["final_output"]
    assert "score" in payload["evaluation"]
    assert "passed" in payload["evaluation"]
