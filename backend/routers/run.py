"""
Run Router

This router handles endpoints for executing agent tasks.

Endpoints:
- POST /run - Execute a new agent task synchronously

Responsibilities:
1. Accept user requests (goal string)
2. Validate input (length limits, format, malicious content)
3. Execute LangGraph synchronously (supervisor → research → execution → evaluator)
4. Return final output, evaluation, and memory usage
5. Handle errors gracefully (LLM failures, timeouts, rate limits)

NOT responsible for:
- Exposing internal state (plan, research, traces)
- Storing task history (that's the responsibility of a database layer)
- Managing user authentication (future: separate auth middleware)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))

# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter()

# =============================================================================
# Request/Response Models
# =============================================================================


class RunRequest(BaseModel):
    """
    Request body for executing a new agent task.

    Fields:
    - goal: The task the user wants to accomplish (10-500 characters)

    Example:
        {
            "goal": "Explain the benefits of vector databases for semantic search"
        }
    """

    goal: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The task to execute (3-500 characters)",
        examples=["Explain the benefits of vector databases for semantic search"],
    )

    @field_validator("goal")
    @classmethod
    def validate_goal(cls, v: str) -> str:
        """
        Validate and sanitize goal.

        Checks:
        - Not empty after stripping whitespace
        - No suspicious patterns (future: content moderation)
        """
        v = v.strip()
        if not v:
            raise ValueError("goal cannot be empty or whitespace-only")

        # Future: Add content moderation here
        # - Check for prompt injection attempts
        # - Check for malicious content
        # - Check for PII leakage attempts

        return v


class EvaluationResult(BaseModel):
    """
    Evaluation results for the task output.

    Fields:
    - passed: Whether the output passed evaluation
    - score: Quality score (1-10)
    - reasons: List of evaluation reasons (why it passed/failed)
    """

    passed: bool = Field(..., description="Whether output passed evaluation")
    score: int = Field(..., ge=1, le=10, description="Quality score (1-10)")
    reasons: list[str] = Field(..., description="Evaluation reasons")


class RunResponse(BaseModel):
    """
    Response body for an executed agent task.

    Fields:
    - final_output: The final generated output from the execution agent
    - evaluation: Evaluation results (passed, score, reasons)
    - memory_used: Whether memory was used in planning

    Example:
        {
            "final_output": "Vector databases are specialized systems...",
            "evaluation": {
                "passed": true,
                "score": 9,
                "reasons": ["Clear explanation", "Covers key concepts"]
            },
            "memory_used": true
        }

    Note:
    - Internal state (plan, research, traces) is NOT exposed
    - Only final output and evaluation are returned
    """

    final_output: str = Field(..., description="Final output from execution agent")
    evaluation: EvaluationResult = Field(..., description="Evaluation results")
    memory_used: bool = Field(..., description="Whether memory was used in planning")


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/run")
async def execute_task(request: RunRequest):
    """
    Execute a new agent task.

    This endpoint:
    1. Validates the request (Pydantic handles this)
    2. Executes LangGraph workflow (supervisor → research → execution → evaluator)
    3. Returns final output, evaluation, and memory usage
    4. Handles errors gracefully without exposing internal state

    Args:
        request: RunRequest with goal string

    Returns:
        RunResponse with final_output, evaluation, and memory_used

    Raises:
        HTTPException 422: Invalid request (validation error)
        HTTPException 500: Execution failed (LLM errors, timeouts, etc.)
        HTTPException 503: Service unavailable (API keys missing, etc.)

    Privacy & Security:
    - Internal state (plan, research, traces) is NOT exposed
    - Only final output and evaluation are returned
    - Stack traces are logged but not sent to client
    """
    return await _execute_task(request)


async def _execute_task(request: RunRequest) -> RunResponse:
    """
    Execute a task using the LangGraph workflow.
    """

    try:
        from agentops_ai_platform.graphs.main_graph import GraphState, build_main_graph
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "service_unavailable",
                "message": "Agent execution service is not available",
                "details": f"Required dependencies not installed: {e}",
            },
        ) from e

    try:
        # Initialize graph state with user goal
        initial_state: GraphState = {
            "user_goal": request.goal,
            "requires_research": None,  # Set by supervisor
            "plan": None,  # Set by supervisor
            "success_criteria": None,  # Set by supervisor
            "research_results": None,  # Set by research (if needed)
            "draft_output": None,  # Set by execution
            "evaluation": None,  # Set by evaluator
            "iteration_count": 0,
            "memory_used": False,  # Set by supervisor
        }

        # Create and execute the graph
        # This runs: supervisor → research (if needed) → execution → evaluator
        graph = build_main_graph().compile()
        final_state = graph.invoke(initial_state)

        # Extract results from final state
        # Only expose: final output, evaluation, memory usage
        # Do NOT expose: plan, research, internal traces
        draft_output = final_state.get("draft_output") or ""
        evaluation_data = final_state.get("evaluation")

        # Check if memory was used (tracked by supervisor_node)
        memory_used = final_state.get("memory_used", False)

        # Validate that we have required fields
        if not draft_output:
            raise ValueError("No output generated by execution agent")

        if not evaluation_data or not isinstance(evaluation_data, dict):
            raise ValueError("No evaluation performed by evaluator agent")

        # Build evaluation result from evaluator output
        # evaluator_agent.py returns: { "pass": bool, "score": int, "reasons": list, ... }
        evaluation_result = EvaluationResult(
            passed=evaluation_data.get("pass_", False) or evaluation_data.get("pass", False),
            score=max(1, min(10, int(evaluation_data.get("score") or 1))),
            reasons=evaluation_data.get("reasons") or ["No reasons provided"],
        )

        # Return final response
        return RunResponse(
            final_output=draft_output,
            evaluation=evaluation_result,
            memory_used=memory_used,
        )

    except ValueError as e:
        # Validation error in LangGraph execution
        # e.g., empty output, missing evaluation
        raise HTTPException(
            status_code=500,
            detail={
                "error": "execution_failed",
                "message": "Task execution failed due to internal error",
                "details": str(e),
            },
        )

    except Exception as e:
        # Catch-all for unexpected errors
        # Log the full error internally, return sanitized message to client
        error_type = type(e).__name__

        # Classify error for user-friendly messaging
        if "API" in error_type or "RateLimit" in error_type or "Timeout" in error_type:
            # LLM API errors (rate limits, timeouts, etc.)
            error_category = "llm_api_error"
            message = "Failed to execute task due to LLM API error. Please try again."
        elif "ValidationError" in error_type:
            # Pydantic validation errors (malformed data)
            error_category = "validation_error"
            message = "Task execution produced invalid data. Please report this issue."
        else:
            # Unknown error
            error_category = "unknown_error"
            message = "An unexpected error occurred. Please try again or contact support."

        # Log full error internally (not exposed to client)
        print(f"❌ Task execution error: {error_type}: {e}")

        # Return sanitized error to client
        raise HTTPException(
            status_code=500,
            detail={
                "error": error_category,
                "message": message,
                "details": f"{error_type}",  # Type only, not full message
            },
        )


