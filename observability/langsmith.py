"""
LangSmith observability scaffolding.

When to use LangSmith vs Langfuse (high-level):
- LangSmith: best when you want deep LangChain/LangGraph-native tracing, run trees,
  prompt/version experiments, and evaluation workflows tightly coupled to the LangChain ecosystem.
- Langfuse: best when you want product-analytics-style tracing across many services,
  model/provider-agnostic dashboards, and flexible spans/events that extend beyond LangChain.

IMPORTANT:
- This module is scaffolding only. It does NOT send any data by default.
- Emission is disabled unless `OBSERVABILITY_ENABLED=1`.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Optional


def _enabled() -> bool:
    # Central kill-switch so local dev/tests don’t accidentally emit traces.
    return os.getenv("OBSERVABILITY_ENABLED", "0") == "1"


def get_langsmith_client():
    """
    Initialize and return a LangSmith client from environment variables.

    Common env vars (LangSmith supports multiple conventions):
    - LANGSMITH_API_KEY or LANGCHAIN_API_KEY
    - LANGSMITH_ENDPOINT or LANGCHAIN_ENDPOINT (optional)
    - LANGSMITH_PROJECT or LANGCHAIN_PROJECT (optional)

    Note: this returns a client instance but does not emit any data on its own.
    """

    try:
        from langsmith import Client  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "LangSmith client not installed. Install with:\n"
            "  python -m pip install langsmith"
        ) from e

    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing LangSmith API key. Set one of:\n"
            "- LANGSMITH_API_KEY\n"
            "- LANGCHAIN_API_KEY"
        )

    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")

    # The LangSmith client accepts environment-driven configuration; we pass explicitly
    # to make behavior obvious and testable.
    kwargs: dict[str, Any] = {"api_key": api_key}
    if endpoint:
        kwargs["api_url"] = endpoint

    return Client(**kwargs)


@dataclass(frozen=True)
class NoOpSpan:
    name: str
    metadata: dict[str, Any]

    @contextmanager
    def span(self, name: str, metadata: Optional[dict[str, Any]] = None) -> Iterator["NoOpSpan"]:
        yield NoOpSpan(name=name, metadata=metadata or {})

    def end(self) -> None:
        return


@dataclass(frozen=True)
class LangSmithTrace:
    """
    Minimal LangSmith run tree handle.

    This uses `langsmith.run_trees.RunTree` to represent a trace with child spans.
    Metadata is attached to `extra.metadata` only (no prompts, no full outputs).
    """

    run_tree: Any

    @contextmanager
    def span(self, name: str, metadata: Optional[dict[str, Any]] = None) -> Iterator["LangSmithSpan"]:
        span = LangSmithSpan(parent=self.run_tree, name=name, metadata=metadata or {})
        try:
            yield span
        finally:
            span.end()

    def end(self) -> None:
        try:
            self.run_tree.end(outputs={}, end_time=datetime.now(timezone.utc))
        except Exception:
            pass
        
        # Post full tree (including child spans) best-effort.
        # If LangSmith API returns 403 or any error, silently ignore.
        try:
            self.run_tree.post(exclude_child_runs=False)
        except Exception:
            # Silently ignore errors when posting to LangSmith
            # (e.g., 403 Forbidden, network issues, etc.)
            pass


@dataclass
class LangSmithSpan:
    parent: Any
    name: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        extra = {"metadata": self.metadata}
        self._child = self.parent.create_child(
            name=self.name,
            run_type="chain",
            inputs={},
            extra=extra,
            start_time=datetime.now(timezone.utc),
        )

    @contextmanager
    def span(self, name: str, metadata: Optional[dict[str, Any]] = None) -> Iterator["LangSmithSpan"]:
        # Nested spans under this span.
        span = LangSmithSpan(parent=self._child, name=name, metadata=metadata or {})
        try:
            yield span
        finally:
            span.end()

    def end(self) -> None:
        try:
            self._child.end(outputs={}, end_time=datetime.now(timezone.utc))
        except Exception:
            return


def create_trace(name: str, metadata: Optional[dict[str, Any]] = None) -> LangSmithTrace | NoOpSpan:
    """
    Create a top-level trace handle.

    - Returns a no-op handle unless `OBSERVABILITY_ENABLED=1`.
    - Emits a LangSmith run tree when enabled and credentials are present.
    - If LangSmith API returns errors (e.g., 403 Forbidden), returns a no-op handle.
    """

    if not _enabled():
        return NoOpSpan(name=name, metadata=metadata or {})

    try:
        from langsmith.run_trees import RunTree  # type: ignore
    except Exception:  # pragma: no cover
        # LangSmith not installed - silently fall back to no-op
        return NoOpSpan(name=name, metadata=metadata or {})

    try:
        client = get_langsmith_client()

        # Project is used by LangSmith to group runs; it's safe to default.
        project = (
            os.getenv("LANGSMITH_PROJECT")
            or os.getenv("LANGCHAIN_PROJECT")
            or os.getenv("LANGCHAIN_SESSION")
            or "agentops-ai"
        )

        run_tree = RunTree(
            name=name,
            run_type="chain",
            inputs={},
            extra={"metadata": (metadata or {})},
            start_time=datetime.now(timezone.utc),
            client=client,
            project_name=project,
        )

        return LangSmithTrace(run_tree=run_tree)
    except Exception as e:
        # LangSmith client error (invalid API key, 403, network issue, etc.)
        # Log warning but don't crash - return no-op handle
        print(f"⚠️  LangSmith tracing disabled due to error: {e}")
        return NoOpSpan(name=name, metadata=metadata or {})

