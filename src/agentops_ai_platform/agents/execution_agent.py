"""
Execution agent contracts.

This module defines schema/contracts for execution outputs.
LLM calls are kept minimal and strictly for producing validated schemas.

TOOL INTEGRATION PHILOSOPHY:
- Tools are only called when explicitly indicated in the Supervisor's plan
- Tool results are treated as external information and clearly labeled
- No autonomous tool calling - the Executor follows Supervisor intent only
- This ensures accountability: if a tool is misused, the issue traces to planning
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


class ExecutionResult(BaseModel):
    """
    Structured execution output produced by the Execution Agent.

    This contract is designed to make runs auditable and evaluator-friendly by
    separating final output from the steps and assumptions used to produce it.
    """

    # The user-facing (or downstream-facing) output produced by the executor.
    output: str = Field(..., description="Final output produced by the Execution Agent.")

    # A normalized list of actions the executor performed (e.g., "called X API",
    # "ran retrieval query Y", "transformed data Z").
    #
    # Observability value: actions_taken provides an audit trail that is easy to
    # trace, debug, and measure (counts, latency buckets, error rates) without
    # relying on brittle log parsing or full tool payload dumps.
    actions_taken: list[str] = Field(
        ...,
        description="Ordered list of notable actions performed during execution.",
    )

    # Explicit assumptions made during execution (e.g., "Assuming X is true because…").
    #
    # Evaluator value: assumptions highlight where the executor lacked certainty or
    # used heuristics, allowing the Evaluator to catch errors, request evidence,
    # or require qualification instead of treating uncertain claims as facts.
    assumptions: list[str] = Field(
        ...,
        description="Assumptions made that could impact correctness or completeness.",
    )


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _should_use_tool(plan_steps: list[str], tool_name: str) -> bool:
    """
    Determine if a tool should be used based on Supervisor's plan.
    
    WHY WE CHECK THE PLAN:
    - Tools are only called when the Supervisor explicitly intended them
    - This prevents autonomous tool usage, which could lead to unintended actions
    - Makes the system auditable: tool usage traces back to planning decisions
    - Aligns with the principle: Executor follows, Supervisor leads
    
    Args:
        plan_steps: List of plan steps from the Supervisor
        tool_name: Name of the tool to check for
        
    Returns:
        True if the plan indicates the tool should be used
    """
    plan_text = " ".join(plan_steps).lower()
    
    # Keywords that indicate tool usage intent
    tool_keywords = {
        "web_search": ["search", "look up", "find information", "web search", "research online"],
    }
    
    keywords = tool_keywords.get(tool_name, [])
    return any(keyword in plan_text for keyword in keywords)


def _call_tool_safely(
    tool_func: Callable,
    tool_name: str,
    tool_input: BaseModel,
) -> tuple[Optional[Any], str]:
    """
    Call a tool with validated input and handle errors gracefully.
    
    WHY SAFE TOOL CALLING MATTERS:
    - External APIs can fail (network, rate limits, malformed data)
    - Tool failures should not crash the entire execution pipeline
    - Failed tool calls should be logged and reported clearly
    - The Executor should be able to continue with partial results
    
    Args:
        tool_func: The tool function to call
        tool_name: Name of the tool (for logging)
        tool_input: Validated Pydantic input model
        
    Returns:
        Tuple of (result, log_message)
        - result: Tool output or None if failed
        - log_message: Human-readable description of what happened
    """
    try:
        result = tool_func(tool_input)
        if result and hasattr(result, 'results') and result.results:
            count = len(result.results)
            log_msg = f"Called {tool_name} → returned {count} result(s) [EXTERNAL INFORMATION]"
        else:
            log_msg = f"Called {tool_name} → no results returned [EXTERNAL INFORMATION]"
        return result, log_msg
    except Exception as e:
        log_msg = f"Called {tool_name} → FAILED: {str(e)} [TOOL ERROR]"
        return None, log_msg


def _extract_json_object(text: str) -> dict[str, Any]:
    """
    Best-effort extraction of a JSON object from model output.

    Even with strict instructions, providers may occasionally wrap responses in
    code fences or add leading/trailing whitespace. This keeps parsing resilient.
    """

    candidate = text.strip()

    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()

    match = _JSON_OBJECT_RE.search(candidate)
    if not match:
        raise ValueError("Model did not return a JSON object.")

    return json.loads(match.group(0))


def execute_task(
    user_goal: str,
    plan_steps: list[str],
    research_results: Optional[str],
    tool_registry: Optional[dict[str, Callable]] = None,
) -> ExecutionResult:
    """
    Execute the task using Gemini and return a validated `ExecutionResult`.

    This agent is execution-only: it follows the provided plan steps and uses
    provided research context; it must not re-plan or do new research.
    
    TOOL USAGE:
    - Tools are only called if indicated in the Supervisor's plan_steps
    - Tool results are injected into research_results with clear labels
    - Tool calls are logged in actions_taken for full auditability
    
    Args:
        user_goal: The user's original request
        plan_steps: Ordered steps from the Supervisor
        research_results: Context from the Research Agent (if any)
        tool_registry: Optional dict mapping tool names to callable functions
        
    Returns:
        ExecutionResult with output, actions, and assumptions
    
    For streaming support, use `execute_task_streaming()` instead.
    """
    
    tool_registry = tool_registry or {}
    
    # Track tool actions for auditability
    tool_actions = []
    tool_results_text = ""
    
    # Check if tools should be called based on Supervisor's plan
    for tool_name, tool_func in tool_registry.items():
        if _should_use_tool(plan_steps, tool_name):
            # Extract parameters from plan (simplified - in production, use LLM to extract)
            # For web_search, look for quoted query or use the user_goal
            if tool_name == "web_search":
                # Import the input model
                try:
                    from tools.web_search import WebSearchInput
                    
                    # Simple extraction: use user_goal as query (production should parse plan)
                    search_input = WebSearchInput(query=user_goal, max_results=5)
                    result, log_msg = _call_tool_safely(tool_func, tool_name, search_input)
                    tool_actions.append(log_msg)
                    
                    if result and hasattr(result, 'results'):
                        # Format tool results with clear labeling
                        tool_results_text += "\n\n--- WEB SEARCH RESULTS (EXTERNAL INFORMATION) ---\n"
                        tool_results_text += f"Query: {search_input.query}\n"
                        tool_results_text += f"Notes: {result.notes}\n\n"
                        for i, res in enumerate(result.results, 1):
                            tool_results_text += f"{i}. {res['title']}\n"
                            tool_results_text += f"   {res['snippet']}\n"
                            tool_results_text += f"   Source: {res['source']}\n\n"
                except ImportError:
                    tool_actions.append(f"Could not import {tool_name} input model [TOOL ERROR]")
    
    # Merge tool results with research results
    if tool_results_text:
        if research_results:
            research_results = research_results + tool_results_text
        else:
            research_results = tool_results_text.strip()

    # Offline structural testing mode: avoids network calls and secrets entirely.
    if os.getenv("OFFLINE_MODE") == "1":
        offline_actions = [
            "OFFLINE_MODE enabled (no model call performed).",
            f"Received {len(plan_steps)} plan steps.",
            "Produced placeholder output without tool usage.",
        ]
        offline_actions.extend(tool_actions)  # Include any tool calls that happened
        
        return ExecutionResult(
            output="OFFLINE_MODE: execution is disabled; this is a placeholder output.",
            actions_taken=offline_actions,
            assumptions=[
                "OFFLINE_MODE: no research or execution was actually performed.",
            ],
        )

    # Import locally to keep module import-time light and to avoid requiring the
    # full LangChain provider stack when only schemas are used.
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Missing GOOGLE_API_KEY. Set it in your environment, e.g.:\n"
            "  export GOOGLE_API_KEY=\"...\""
        )

    # Use Gemini 2.5 Flash for low-latency execution drafting.
    model_name = os.getenv("GEMINI_EXECUTION_MODEL", "gemini-2.0-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, max_retries=3, timeout=30)

    # --- Prompt section: role boundary (execution-only) ---
    # Prevents the model from “helpfully” re-planning, expanding scope, or inventing
    # research. This keeps responsibilities separate and makes failures attributable.
    system_role = SystemMessage(
        content=(
            "You are the Execution Agent. Your role is strictly execution-only.\n"
            "- Follow the given plan steps.\n"
            "- Use the provided research context if present.\n"
            "- Do NOT re-plan, do new research, browse the web, or recommend next steps.\n"
            "- Do NOT add tool usage; produce a draft output only."
        )
    )

    # --- Prompt section: follow plan steps ---
    # Ensures the executor is deterministic relative to a supervisor-produced plan.
    #
    # --- Prompt section: require use of research_results ---
    # Prevents ignoring retrieved context (a common failure mode in agent pipelines).
    #
    # --- Prompt section: structured JSON output ---
    # Allows the platform to trace actions/assumptions and evaluate quality reliably.
    user_request = HumanMessage(
        content=(
            "User goal:\n"
            f"{user_goal}\n\n"
            "Plan steps (follow in order; do not re-plan):\n"
            + "\n".join(f"- {s}" for s in plan_steps)
            + "\n\n"
            "Research context (must use if provided):\n"
            f"{research_results if research_results else '(none provided)'}\n\n"
            "Return ONLY a JSON object matching this schema:\n"
            "{\n"
            '  \"output\": string,\n'
            '  \"actions_taken\": string[],\n'
            '  \"assumptions\": string[]\n'
            "}\n\n"
            "Constraints:\n"
            "- Execution-only: do not add new plan steps.\n"
            "- No researching: do not claim you looked up sources not present above.\n"
            "- If research context is missing, list assumptions explicitly.\n"
            "- actions_taken must reflect how you followed the provided plan steps.\n"
            "- IMPORTANT: you MUST include ALL three keys (output, actions_taken, assumptions) even if a list is empty.\n"
            "- Do not include extra keys, commentary, markdown, or code fences."
        )
    )

    response = llm.invoke([system_role, user_request])
    payload = _extract_json_object(getattr(response, "content", str(response)))
    # Defensive normalization: some models occasionally omit fields even when instructed.
    # We fail closed on `output` (must exist), but we can reconstruct the audit fields
    # from the provided plan/context to keep the pipeline running deterministically.
    if "actions_taken" not in payload or not isinstance(payload.get("actions_taken"), list):
        payload["actions_taken"] = [f"Followed plan step: {s}" for s in plan_steps]
    
    # Merge tool actions into actions_taken (tool calls happen before LLM execution)
    payload["actions_taken"] = tool_actions + payload["actions_taken"]
    
    if "assumptions" not in payload or not isinstance(payload.get("assumptions"), list):
        payload["assumptions"] = []
    if not payload["assumptions"]:
        if research_results:
            payload["assumptions"].append("Used only the provided research context; did not perform new research.")
        else:
            payload["assumptions"].append("No research context was provided; relied on general knowledge.")
    
    # Add assumption about tool usage if tools were called
    if tool_actions:
        payload["assumptions"].append("Tool results are external information and may be incomplete or outdated.")
    
    return ExecutionResult.model_validate(payload)


def execute_task_streaming(
    user_goal: str,
    plan_steps: list[str],
    research_results: Optional[str],
    tool_registry: Optional[dict[str, Callable]] = None,
):
    """
    Execute the task using Gemini with streaming output.
    
    This function yields chunks of the execution output as they are generated,
    then yields a final ExecutionResult at the end.
    
    TOOL USAGE (same as non-streaming):
    - Tools are only called if indicated in the Supervisor's plan_steps
    - Tool results are injected into research_results with clear labels
    - Tool calls are logged in actions_taken for full auditability
    
    Args:
        user_goal: The user's original request
        plan_steps: Ordered steps from the Supervisor
        research_results: Context from the Research Agent (if any)
        tool_registry: Optional dict mapping tool names to callable functions
    
    Yields:
        str: Chunks of the execution output text
        ExecutionResult: Final result with complete output and metadata (last yield)
    
    Why streaming is only for execution:
    - Planning (Supervisor): Must complete before execution starts (dependencies)
    - Research: Must complete before execution uses it (dependencies)
    - Execution: Can stream because it's the main user-facing output
    - Evaluation: Must analyze the complete output (can't evaluate partial text)
    
    Streaming execution output provides immediate feedback to users while
    maintaining the integrity of the multi-agent pipeline.
    """
    
    tool_registry = tool_registry or {}
    
    # Track tool actions for auditability (same as non-streaming)
    tool_actions = []
    tool_results_text = ""
    
    # Check if tools should be called based on Supervisor's plan
    for tool_name, tool_func in tool_registry.items():
        if _should_use_tool(plan_steps, tool_name):
            # Extract parameters from plan (simplified - in production, use LLM to extract)
            # For web_search, look for quoted query or use the user_goal
            if tool_name == "web_search":
                # Import the input model
                try:
                    from tools.web_search import WebSearchInput
                    
                    # Simple extraction: use user_goal as query (production should parse plan)
                    search_input = WebSearchInput(query=user_goal, max_results=5)
                    result, log_msg = _call_tool_safely(tool_func, tool_name, search_input)
                    tool_actions.append(log_msg)
                    
                    if result and hasattr(result, 'results'):
                        # Format tool results with clear labeling
                        tool_results_text += "\n\n--- WEB SEARCH RESULTS (EXTERNAL INFORMATION) ---\n"
                        tool_results_text += f"Query: {search_input.query}\n"
                        tool_results_text += f"Notes: {result.notes}\n\n"
                        for i, res in enumerate(result.results, 1):
                            tool_results_text += f"{i}. {res['title']}\n"
                            tool_results_text += f"   {res['snippet']}\n"
                            tool_results_text += f"   Source: {res['source']}\n\n"
                except ImportError:
                    tool_actions.append(f"Could not import {tool_name} input model [TOOL ERROR]")
    
    # Merge tool results with research results
    if tool_results_text:
        if research_results:
            research_results = research_results + tool_results_text
        else:
            research_results = tool_results_text.strip()
    
    # Offline mode: return immediately without streaming
    if os.getenv("OFFLINE_MODE") == "1":
        offline_actions = [
            "OFFLINE_MODE enabled (no model call performed).",
            f"Received {len(plan_steps)} plan steps.",
            "Produced placeholder output without tool usage.",
        ]
        offline_actions.extend(tool_actions)  # Include any tool calls that happened
        
        result = ExecutionResult(
            output="OFFLINE_MODE: execution is disabled; this is a placeholder output.",
            actions_taken=offline_actions,
            assumptions=[
                "OFFLINE_MODE: no research or execution was actually performed.",
            ],
        )
        yield result.output
        yield result
        return
    
    # Import locally to keep module import-time light
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Missing GOOGLE_API_KEY. Set it in your environment, e.g.:\n"
            "  export GOOGLE_API_KEY=\"...\""
        )
    
    # Use Gemini 2.5 Flash for low-latency execution
    model_name = os.getenv("GEMINI_EXECUTION_MODEL", "gemini-2.0-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, streaming=True, max_retries=3, timeout=30)
    
    # Same prompts as non-streaming version
    system_role = SystemMessage(
        content=(
            "You are the Execution Agent. Your role is strictly execution-only.\n"
            "- Follow the given plan steps.\n"
            "- Use the provided research context if present.\n"
            "- Do NOT re-plan, do new research, browse the web, or recommend next steps.\n"
            "- Do NOT add tool usage; produce a draft output only."
        )
    )
    
    user_request = HumanMessage(
        content=(
            "User goal:\n"
            f"{user_goal}\n\n"
            "Plan steps (follow in order; do not re-plan):\n"
            + "\n".join(f"- {s}" for s in plan_steps)
            + "\n\n"
            "Research context (must use if provided):\n"
            f"{research_results if research_results else '(none provided)'}\n\n"
            "Return ONLY a JSON object matching this schema:\n"
            "{\n"
            '  \"output\": string,\n'
            '  \"actions_taken\": string[],\n'
            '  \"assumptions\": string[]\n'
            "}\n\n"
            "Constraints:\n"
            "- Execution-only: do not add new plan steps.\n"
            "- No researching: do not claim you looked up sources not present above.\n"
            "- If research context is missing, list assumptions explicitly.\n"
            "- actions_taken must reflect how you followed the provided plan steps.\n"
            "- IMPORTANT: you MUST include ALL three keys (output, actions_taken, assumptions) even if a list is empty.\n"
            "- Do not include extra keys, commentary, markdown, or code fences."
        )
    )
    
    # Stream the response and accumulate full text
    full_response = ""
    try:
        for chunk in llm.stream([system_role, user_request]):
            content = getattr(chunk, "content", "")
            if content:
                full_response += content
                # Only yield the content chunks, not the full response yet
                # We'll parse and yield the ExecutionResult at the end
                yield content
    except Exception as e:
        # If streaming fails, log and re-raise to trigger fallback
        print(f"❌ Streaming execution failed: {e}")
        raise
    
    # Parse the complete response into ExecutionResult
    try:
        payload = _extract_json_object(full_response)
        
        # Defensive normalization (same as non-streaming version)
        if "actions_taken" not in payload or not isinstance(payload.get("actions_taken"), list):
            payload["actions_taken"] = [f"Followed plan step: {s}" for s in plan_steps]
        
        # Merge tool actions into actions_taken (tool calls happen before LLM execution)
        payload["actions_taken"] = tool_actions + payload["actions_taken"]
        
        if "assumptions" not in payload or not isinstance(payload.get("assumptions"), list):
            payload["assumptions"] = []
        if not payload["assumptions"]:
            if research_results:
                payload["assumptions"].append("Used only the provided research context; did not perform new research.")
            else:
                payload["assumptions"].append("No research context was provided; relied on general knowledge.")
        
        # Add assumption about tool usage if tools were called
        if tool_actions:
            payload["assumptions"].append("Tool results are external information and may be incomplete or outdated.")
        
        result = ExecutionResult.model_validate(payload)
        
        # Yield the final ExecutionResult as the last item
        # The endpoint will use this to run the evaluator
        yield result
        
    except Exception as e:
        # If parsing fails, create a fallback result with the raw output
        print(f"⚠️ Failed to parse streaming response as JSON, using raw output: {e}")
        fallback_actions = tool_actions + [f"Followed plan step: {s}" for s in plan_steps]
        fallback_assumptions = ["Output was generated but JSON parsing failed; using raw response."]
        if tool_actions:
            fallback_assumptions.append("Tool results are external information and may be incomplete or outdated.")
        
        result = ExecutionResult(
            output=full_response,
            actions_taken=fallback_actions,
            assumptions=fallback_assumptions,
        )
        yield result
