"""
Tools package for AgentOps AI Platform.

This package contains tool implementations that can be used by agents.
All tools follow strict schemas using Pydantic and fail safely.
"""

from tools.web_search import web_search, WebSearchInput, WebSearchResult

__all__ = [
    "web_search",
    "WebSearchInput",
    "WebSearchResult",
]
