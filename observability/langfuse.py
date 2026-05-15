"""
Langfuse observability scaffolding.

When to use LangSmith vs Langfuse (high-level):
- Langfuse: best when you want product/ops-centric tracing across many services,
  model/provider-agnostic dashboards, and flexible traces/spans that extend beyond LangChain.
- LangSmith: best when you want deep LangChain/LangGraph-native tracing, run trees,
  and evaluation workflows tightly coupled to the LangChain ecosystem.

IMPORTANT:
- This module is scaffolding only. It does NOT send any data by default.
- Emission is disabled unless `OBSERVABILITY_ENABLED=1`.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional


def _enabled() -> bool:
    # Central kill-switch so local dev/tests don’t accidentally emit traces.
    return os.getenv("OBSERVABILITY_ENABLED", "0") == "1"


def get_langfuse_client():
    """
    Initialize and return a Langfuse client from environment variables.

    Expected env vars:
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_HOST (optional; e.g., https://cloud.langfuse.com)

    Note: this returns a client instance but does not emit any data on its own.
    """

    try:
        from langfuse import Langfuse  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Langfuse client not installed. Install with:\n"
            "  python -m pip install langfuse"
        ) from e

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        raise RuntimeError(
            "Missing Langfuse keys. Set:\n"
            "- LANGFUSE_PUBLIC_KEY\n"
            "- LANGFUSE_SECRET_KEY"
        )

    host = os.getenv("LANGFUSE_HOST")
    kwargs: dict[str, Any] = {"public_key": public_key, "secret_key": secret_key}
    if host:
        kwargs["host"] = host

    return Langfuse(**kwargs)


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
class LangfuseTrace:
    """Minimal Langfuse trace handle using the v3 SDK span API."""

    client: Any
    _root_span: Any = None

    @property
    def trace_id(self) -> Optional[str]:
        if self._root_span and hasattr(self._root_span, "trace_id"):
            return self._root_span.trace_id
        return None

    @contextmanager
    def span(self, name: str, metadata: Optional[dict[str, Any]] = None) -> Iterator["LangfuseSpan"]:
        span = LangfuseSpan(parent=self._root_span or self.client, name=name, metadata=metadata or {})
        try:
            yield span
        finally:
            span.end()

    def end(self) -> None:
        try:
            if self._root_span is not None:
                self._root_span.end()
        except Exception:
            pass
        try:
            self.client.flush()
        except Exception:
            pass


@dataclass
class LangfuseSpan:
    parent: Any
    name: str
    metadata: dict[str, Any]
    _span_obj: Any = None

    def __post_init__(self) -> None:
        try:
            self._span_obj = self.parent.start_span(name=self.name, metadata=self.metadata)
        except Exception:
            self._span_obj = None

    @contextmanager
    def span(self, name: str, metadata: Optional[dict[str, Any]] = None) -> Iterator["LangfuseSpan"]:
        parent = self._span_obj if self._span_obj is not None else self.parent
        span = LangfuseSpan(parent=parent, name=name, metadata=metadata or {})
        try:
            yield span
        finally:
            span.end()

    def end(self) -> None:
        try:
            if self._span_obj is not None:
                self._span_obj.end()
        except Exception:
            pass


def create_trace(name: str, metadata: Optional[dict[str, Any]] = None) -> LangfuseTrace | NoOpSpan:
    """
    Create a top-level trace handle.

    Returns a no-op handle unless OBSERVABILITY_ENABLED=1 and credentials are present.
    """

    if not _enabled():
        return NoOpSpan(name=name, metadata=metadata or {})

    try:
        client = get_langfuse_client()
        root_span = client.start_span(name=name, metadata=metadata or {})
        return LangfuseTrace(client=client, _root_span=root_span)
    except Exception as e:
        print(f"⚠️  Langfuse tracing disabled due to error: {e}")
        return NoOpSpan(name=name, metadata=metadata or {})


# =============================================================================
# Metric logging helpers
# =============================================================================
#
# CRITICAL SAFETY REQUIREMENT:
# These helpers MUST fail silently and NEVER crash the application.
#
# Why this matters:
# 1. Observability is optional — missing credentials should not block execution.
# 2. Network issues, rate limits, or API changes must not break agent logic.
# 3. Metrics are for monitoring/analysis — losing a metric is acceptable, but
#    losing a user request or causing a runtime crash is not.
#
# All functions below wrap operations in try/except and return immediately on error.
# =============================================================================


def log_evaluation_score(score: int, trace_id: Optional[str] = None, name: str = "evaluation_score") -> None:
    """
    Log an evaluation score metric to Langfuse.

    Args:
        score: Integer score (typically 1-10) representing evaluation quality.
        trace_id: Optional trace ID to attach the score to. If None, logs as a standalone metric.
        name: Name/label for the score metric (default: "evaluation_score").

    Safety:
        - Only logs if OBSERVABILITY_ENABLED=1 and Langfuse credentials are present.
        - Fails silently if Langfuse client cannot be initialized or if API call fails.
        - Never raises exceptions — this function is guaranteed to be safe to call.
    """

    if not _enabled():
        return

    try:
        client = get_langfuse_client()

        kwargs: dict[str, Any] = {"name": name, "value": score}
        if trace_id:
            kwargs["trace_id"] = trace_id

        client.create_score(**kwargs)

        try:
            client.flush()
        except Exception:
            pass

    except Exception:
        # Silently ignore all errors:
        # - Missing credentials
        # - Network failures
        # - API rate limits
        # - SDK changes
        # Metrics should NEVER break production workflows.
        return


def log_evaluation_pass(pass_value: bool, trace_id: Optional[str] = None, name: str = "evaluation_pass") -> None:
    """
    Log a pass/fail evaluation result to Langfuse.

    Args:
        pass_value: Boolean indicating whether evaluation passed (True) or failed (False).
        trace_id: Optional trace ID to attach the metric to.
        name: Name/label for the pass/fail metric (default: "evaluation_pass").

    Safety:
        - Only logs if OBSERVABILITY_ENABLED=1 and Langfuse credentials are present.
        - Fails silently if Langfuse client cannot be initialized or if API call fails.
        - Never raises exceptions — this function is guaranteed to be safe to call.

    Note:
        - Pass/fail is logged as a boolean score (1.0 for pass, 0.0 for fail).
        - This format is compatible with Langfuse's score aggregation and dashboards.
    """

    if not _enabled():
        return

    try:
        client = get_langfuse_client()

        # Convert boolean to numeric score (1.0 = pass, 0.0 = fail)
        # This allows Langfuse to aggregate pass rates as percentages.
        numeric_value = 1.0 if pass_value else 0.0

        kwargs: dict[str, Any] = {"name": name, "value": numeric_value}
        if trace_id:
            kwargs["trace_id"] = trace_id

        client.create_score(**kwargs)

        try:
            client.flush()
        except Exception:
            pass

    except Exception:
        return


def log_retry_count(iteration_count: int, trace_id: Optional[str] = None, name: str = "retry_count") -> None:
    """
    Log the number of retry iterations to Langfuse.

    Args:
        iteration_count: Number of times the workflow has retried (0 = first attempt, 1+ = retries).
        trace_id: Optional trace ID to attach the metric to.
        name: Name/label for the retry count metric (default: "retry_count").

    Safety:
        - Only logs if OBSERVABILITY_ENABLED=1 and Langfuse credentials are present.
        - Fails silently if Langfuse client cannot be initialized or if API call fails.
        - Never raises exceptions — this function is guaranteed to be safe to call.

    Use cases:
        - Track retry behavior for debugging infinite loops or flaky evaluations.
        - Monitor agent success rates (iteration_count=0 means first-try success).
        - Identify workflows that frequently hit max retry limits.
    """

    if not _enabled():
        return

    try:
        client = get_langfuse_client()

        kwargs: dict[str, Any] = {"name": name, "value": iteration_count}
        if trace_id:
            kwargs["trace_id"] = trace_id

        client.create_score(**kwargs)

        try:
            client.flush()
        except Exception:
            pass

    except Exception:
        return

