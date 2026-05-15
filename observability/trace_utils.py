"""
Trace utilities (LangSmith + Langfuse).

Sends spans to both backends simultaneously when OBSERVABILITY_ENABLED=1.
Missing deps/credentials degrade gracefully to no-ops per backend.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional


def _enabled() -> bool:
    return os.getenv("OBSERVABILITY_ENABLED", "0") == "1"


class _NoOp:
    @contextmanager
    def span(self, _name: str, metadata: Optional[dict[str, Any]] = None) -> Iterator["_NoOp"]:
        yield self

    def end(self) -> None:
        return


class _MultiTrace:
    """Fans out span/end calls to multiple backend traces."""

    def __init__(self, traces: list[Any]):
        self._traces = traces

    @property
    def langfuse_trace_id(self) -> Optional[str]:
        """Return the Langfuse trace_id if a Langfuse backend is active."""
        for t in self._traces:
            tid = getattr(t, "trace_id", None)
            if tid:
                return tid
        return None

    @contextmanager
    def span(self, name: str, metadata: Optional[dict[str, Any]] = None) -> Iterator["_MultiTrace"]:
        spans = []
        for t in self._traces:
            try:
                cm = t.span(name, metadata=metadata)
                s = cm.__enter__()
                spans.append((cm, s))
            except Exception:
                pass
        try:
            yield _MultiTrace([s for _, s in spans])
        finally:
            for cm, _ in spans:
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    pass

    def end(self) -> None:
        for t in self._traces:
            try:
                t.end()
            except Exception:
                pass


@contextmanager
def traced(
    name: str,
    metadata: Optional[dict[str, Any]] = None,
    **_kwargs: Any,
) -> Iterator[Any]:
    """
    Start a trace/span and yield a handle that fans out to all available backends.

    Safe-by-default: missing deps/credentials degrade to no-ops per backend.
    """

    if not _enabled():
        yield _NoOp()
        return

    meta = metadata or {}
    traces: list[Any] = []

    # LangSmith
    try:
        from observability.langsmith import create_trace as ls_create
        t = ls_create(name=name, metadata=meta)
        if not isinstance(t, type(None)):
            traces.append(t)
    except Exception:
        pass

    # Langfuse
    try:
        from observability.langfuse import create_trace as lf_create
        t = lf_create(name=name, metadata=meta)
        if not isinstance(t, type(None)):
            traces.append(t)
    except Exception:
        pass

    handle = _MultiTrace(traces) if traces else _NoOp()

    try:
        yield handle
    finally:
        try:
            handle.end()
        except Exception:
            pass

