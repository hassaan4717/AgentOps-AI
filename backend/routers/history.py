"""
History Router

This router handles endpoints for retrieving task history from memory storage.

Endpoints:
- GET /history - List all stored memory records (successful tasks)

Responsibilities:
1. Load memory records from memory store
2. Return lightweight summaries (without full outputs)
3. Support future pagination and filtering

NOT responsible for:
- Executing tasks (that's run.py)
- Storing memories (that's done automatically by evaluator_node)
- Real-time updates (that's WebSocket streaming)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter()

# =============================================================================
# Response Models
# =============================================================================


class MemoryRecordSummary(BaseModel):
    """
    Lightweight summary of a stored memory record.

    Fields:
    - user_goal: User's original request (max 500 chars)
    - summary: Brief summary of the task (max 200 chars for list view)
    - score: Evaluation score (8-10, only high-quality tasks stored)
    - created_at: ISO 8601 timestamp (UTC)

    Note:
    - Full outputs are NOT included (kept lightweight)
    - Only successful tasks are stored in memory
    - All timestamps are in UTC

    Example:
        {
            "user_goal": "Explain the benefits of vector databases",
            "summary": "Vector databases excel at semantic search...",
            "score": 9,
            "created_at": "2026-02-10T12:34:56Z"
        }
    """

    user_goal: str = Field(..., description="Original user request")
    summary: str = Field(..., description="Brief summary (truncated for list view)")
    score: int = Field(..., ge=8, le=10, description="Evaluation score (8-10)")
    created_at: str = Field(..., description="Creation timestamp (ISO 8601, UTC)")


class HistoryResponse(BaseModel):
    """
    List of stored memory records.

    Fields:
    - memories: List of MemoryRecordSummary objects (lightweight)
    - total: Total number of memories

    Example:
        {
            "memories": [
                {
                    "user_goal": "Explain vector databases",
                    "summary": "Vector databases excel at...",
                    "score": 9,
                    "created_at": "2026-02-10T12:34:56Z"
                }
            ],
            "total": 1
        }

    Note:
    - Results are sorted by created_at (newest first)
    - Only high-quality tasks (score >= 8) are stored
    - Full outputs are NOT included (lightweight response)
    """

    memories: List[MemoryRecordSummary] = Field(..., description="List of memory records")
    total: int = Field(..., description="Total number of memories")


# Full output endpoint removed for now (lightweight API only)
# Future: Add GET /history/{memory_id} for full details if needed


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/history", response_model=HistoryResponse)
async def list_history():
    """
    List all stored memory records (successful tasks).

    This endpoint returns lightweight summaries of all successful tasks
    that have been stored in memory. Only high-quality tasks (score >= 8)
    are stored, so this represents the "best" executions.

    Returns:
        HistoryResponse with list of memory records

    Response Structure:
    - memories: List of lightweight memory summaries
    - total: Total count of stored memories

    Privacy & Performance:
    - Full outputs are NOT included (kept lightweight)
    - Summaries are truncated to 200 chars for list view
    - Results are sorted by created_at (newest first)

    Raises:
        HTTPException 500: Failed to load memories from storage
        HTTPException 503: Memory storage not available

    Future Enhancements:
    - Pagination (page, page_size query params)
    - Filtering (by score, date range, search term)
    - Sorting (by created_at, score, user_goal)
    """

    try:
        # Import memory store
        from memory.memory_store import load_all_memories

        # Load all memories from storage
        all_memories = load_all_memories()

        # Sort by created_at (newest first)
        sorted_memories = sorted(
            all_memories,
            key=lambda m: m.created_at,
            reverse=True
        )

        # Convert to lightweight response format
        # Truncate summaries to 200 chars for list view
        memory_summaries = []
        for mem in sorted_memories:
            # Truncate summary if too long (keep lightweight)
            summary_text = mem.summary
            if len(summary_text) > 200:
                summary_text = summary_text[:197] + "..."

            memory_summaries.append(
                MemoryRecordSummary(
                    user_goal=mem.user_goal,
                    summary=summary_text,
                    score=mem.score,
                    created_at=mem.created_at.isoformat(),
                )
            )

        return HistoryResponse(
            memories=memory_summaries,
            total=len(memory_summaries),
        )

    except ImportError:
        # Memory store not available
        raise HTTPException(
            status_code=503,
            detail={
                "error": "service_unavailable",
                "message": "Memory storage is not available",
                "details": "Memory store dependencies not installed",
            },
        )

    except Exception as e:
        # Unexpected error loading memories
        error_type = type(e).__name__
        print(f"❌ Failed to load history: {error_type}: {e}")

        raise HTTPException(
            status_code=500,
            detail={
                "error": "storage_error",
                "message": "Failed to load task history",
                "details": f"{error_type}",
            },
        )


# Removed: GET /history/{memory_id} endpoint (not needed for v1)
# Removed: GET /history/stats endpoint (can be added later if needed)
#
# Future Enhancements:
# 1. Add GET /history/{memory_id} for full details (with final_output)
# 2. Add pagination to GET /history (page, page_size params)
# 3. Add filtering (by score, date range, search term)
# 4. Add sorting (by created_at, score, relevance)
# 5. Add DELETE /history/{memory_id} for cleanup


# =============================================================================
# Future Enhancements (Not Yet Implemented)
# =============================================================================
#
# 1. PAGINATION:
#    - Add page and page_size query parameters
#    - Return has_next flag for client-side pagination
#    - Default: 20 results per page, max 100
#
# 2. FILTERING:
#    - Filter by score (min_score, max_score)
#    - Filter by date range (created_after, created_before)
#    - Search user_goal (case-insensitive substring match)
#
# 3. SORTING:
#    - Sort by created_at (asc/desc)
#    - Sort by score (asc/desc)
#    - Sort by user_goal (alphabetical)
#
# 4. DETAIL ENDPOINT:
#    - GET /history/{memory_id}
#    - Return full memory record (with final_output)
#    - Useful for inspecting specific tasks
#
# 5. DELETE ENDPOINT:
#    - DELETE /history/{memory_id}
#    - Remove specific memory from storage
#    - Require authentication/authorization
#
# 6. STATS ENDPOINT:
#    - GET /history/stats
#    - Aggregate metrics (total, avg score, etc.)
#    - Useful for dashboard/analytics
#
# 7. EXPORT ENDPOINT:
#    - GET /history/export?format=json|csv
#    - Export all memories for backup/analysis
#    - Rate limited
