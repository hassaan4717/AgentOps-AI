"""
Supervisor agent contracts.

This module contains schema/contracts used by the supervisor/orchestrator layer.
LLM calls are kept minimal and strictly for producing validated schemas.

WHY PLANNING ≠ EXECUTION:
- Planning: The Supervisor decides WHAT to do and WHICH tools are needed
- Execution: The Executor follows the plan and actually USES the tools
- This separation ensures:
  1. Clear accountability: tool misuse traces to either bad planning or bad execution
  2. Testability: plans can be validated before expensive execution
  3. Observability: we see WHAT was planned vs WHAT was executed
  4. Safety: plans can be reviewed/approved before tools are called
  5. Reusability: same plan can be executed multiple times with different tools
  
The Supervisor declares tool requirements in the plan; the Executor respects those
declarations and calls the tools. Neither should do the other's job.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import BaseModel, Field


class SupervisorPlan(BaseModel):
    """
    Structured plan output expected from the Supervisor (or a planning sub-step).

    This model is intentionally small and explicit so it can be:
    - validated at runtime
    - logged/traced as structured metadata
    - used to drive deterministic routing in the LangGraph
    """

    # Whether the workflow needs a dedicated research step before execution.
    # This is used by the graph router to decide if it should call the Research Agent.
    requires_research: bool = Field(
        ...,
        description="True if the request requires external context retrieval before execution.",
    )

    # A step-by-step task breakdown describing what to do next.
    # This becomes the canonical checklist for downstream execution and repair loops.
    plan_steps: list[str] = Field(
        ...,
        description="Ordered list of plan steps to satisfy the user goal.",
        min_length=1,
    )

    # A concrete definition of “done”.
    # The Evaluator uses these criteria to judge pass/fail, and the Supervisor uses them to stop looping.
    success_criteria: list[str] = Field(
        ...,
        description="Acceptance criteria used to evaluate whether the plan has succeeded.",
        min_length=1,
    )

    # List of tool names required to execute this plan.
    # The Supervisor declares WHICH tools are needed; the Executor decides HOW to use them.
    #
    # Why the Supervisor declares tools:
    # 1. Planning phase: decide if tools are needed BEFORE expensive execution
    # 2. Resource validation: check if required tools are available before starting
    # 3. Security review: plans requiring sensitive tools can be reviewed/approved
    # 4. Observability: see what was planned vs what was executed
    # 5. Clear separation: Supervisor says WHAT, Executor says HOW
    #
    # The Supervisor should:
    # - Declare tool names (e.g., ["web_search"])
    # - NOT describe how to use the tools (that's execution)
    # - Return empty list [] if no tools are needed
    required_tools: list[str] = Field(
        default_factory=list,
        description="List of tool names needed to execute this plan (e.g., ['web_search']). Empty if no tools needed.",
    )


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json_object(text: str) -> dict[str, Any]:
    """
    Best-effort extraction of a JSON object from model output.

    We instruct the model to return raw JSON only, but this keeps the system
    resilient if a provider wraps output in fences or adds stray text.
    """

    candidate = text.strip()

    # If the model accidentally returns code fences, strip them.
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()

    match = _JSON_OBJECT_RE.search(candidate)
    if not match:
        raise ValueError("Model did not return a JSON object.")

    return json.loads(match.group(0))


def generate_plan(user_goal: str) -> SupervisorPlan:
    """
    Generate a `SupervisorPlan` for the given user goal using an LLM.

    Notes:
    - This function does NOT perform any tool usage or side effects.
    - It requests strictly structured JSON matching `SupervisorPlan` and validates it.
    """

    # Offline structural testing mode: avoids network calls and secrets entirely.
    # This is useful for CI scaffolding and for quickly validating graph wiring.
    if os.getenv("OFFLINE_MODE") == "1":
        goal_lower = user_goal.lower()
        requires_research = any(
            k in goal_lower
            for k in (
                "research",
                "compare",
                "pros and cons",
                "trade-off",
                "best practices",
                "find",
                "search",
                "look up",
                "latest",
                "recent",
            )
        )
        # Detect if tools are needed based on keywords
        required_tools = []
        needs_search = any(k in goal_lower for k in (
            "search", "look up", "find", "latest", "recent", 
            "research", "differences", "compare", "best practices"
        ))
        if needs_search:
            required_tools.append("web_search")
        
        # Generate realistic plan steps that include search keywords when tools are declared
        if "web_search" in required_tools:
            plan_steps = [
                f"Search for information about: {user_goal}",
                "Analyze and synthesize the search results",
                "Draft a comprehensive response based on findings",
                "Verify all claims are supported by search results",
                "Refine and structure the final output",
            ]
        else:
            plan_steps = [
                "Clarify the goal and required output shape",
                "Gather any required background context",
                "Draft the response",
                "Evaluate against success criteria",
                "Refine if needed",
            ]
        
        return SupervisorPlan(
            requires_research=requires_research,
            plan_steps=plan_steps,
            success_criteria=[
                "Directly addresses the stated goal",
                "Is structured, clear, and complete",
                "Avoids unsupported claims or clearly labels uncertainty",
            ],
            required_tools=required_tools,
        )

    # Import locally to keep module import-time light and to avoid requiring the
    # full LangChain provider stack when only schemas are used.
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    # --- Prompt section: role and objective ---
    # Establishes the Supervisor’s responsibility: produce an executable plan
    # and explicit completion criteria that downstream agents can follow.
    system_role = SystemMessage(
        content=(
            "You are the Supervisor Agent for an agentic AI platform. "
            "Your job is to produce a concise, step-by-step execution plan "
            "and clear success criteria so other agents can do the work deterministically."
        )
    )

    # --- Prompt section: forced research decision ---
    # Forces a binary decision to drive routing (research_node vs execution_node).
    # The model must commit to `requires_research` rather than hedging.
    #
    # --- Prompt section: structured plan steps ---
    # Requests an ordered list of steps; each step should be action-oriented and testable.
    #
    # --- Prompt section: explicit success criteria ---
    # Defines “done” so the Evaluator can judge pass/fail and the graph can terminate.
    #
    # --- Prompt section: strict JSON output contract ---
    # Emphasizes machine-parseable output that matches the `SupervisorPlan` schema exactly.
    # --- Prompt section: memory context handling (optional) ---
    # If user_goal contains "[CONTEXT: Relevant past successful tasks]", the model
    # is instructed to treat it as ADVISORY ONLY.
    #
    # Why memory is advisory:
    # 1. Past context may not apply to current goal (different requirements)
    # 2. Copying past plans blindly prevents adaptation to new scenarios
    # 3. Memory should refine decisions, not override task analysis
    # 4. Supervisor must remain capable of handling novel tasks
    #
    # This design ensures memory enhances (not replaces) critical thinking.
    user_request = HumanMessage(
        content=(
            "User goal:\n"
            f"{user_goal}\n\n"
            "Return ONLY a JSON object matching this schema:\n"
            "{\n"
            '  "requires_research": boolean,\n'
            '  "plan_steps": string[],\n'
            '  "success_criteria": string[],\n'
            '  "required_tools": string[]\n'
            "}\n\n"
            "Requirements:\n"
            "- Clearly state a step-by-step plan (2–5 steps, keep it concise).\n"
            "- Make an explicit TRUE/FALSE decision for requires_research.\n"
            "- Provide 2–4 concrete success criteria (keep them simple and achievable).\n"
            "- Declare which tools (if any) are needed in required_tools.\n"
            "- Do not include any extra keys, commentary, markdown, or code fences.\n\n"
            "Tool Declaration Guidelines:\n"
            "- Available tools: web_search\n"
            "- Declare tool names ONLY (e.g., [\"web_search\"] or []).\n"
            "- Do NOT describe how to use the tools — that's execution, not planning.\n"
            "- You can choose zero tools if the task doesn't need external data.\n"
            "- Use web_search when: the plan needs current information, facts, or data from the web.\n"
            "- Do NOT use web_search for: general knowledge, code generation, reasoning tasks.\n\n"
            "Past Relevant Experience (if provided above):\n"
            "- Memory is ADVISORY ONLY — use it to refine your plan, not copy it.\n"
            "- DO NOT replicate past outputs or plans verbatim.\n"
            "- Use past tasks to: inform plan structure, avoid repeating mistakes, "
            "assess if research is needed.\n"
            "- IGNORE memory if the current goal differs meaningfully from past tasks."
        )
    )

    # Provider: Google Gemini (Generative Language API key).
    # The key is read from the environment by default.
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Missing GOOGLE_API_KEY. Set it in your environment, e.g.:\n"
            "  export GOOGLE_API_KEY=\"...\"\n"
            "Then re-run your graph."
        )

    # Model choice is configurable for easy experimentation.
    # Availability can vary by account/API version, so we default to a commonly
    # available lightweight model and allow override via GEMINI_MODEL.
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        max_retries=3,
        timeout=30,
    )

    # We intentionally parse/validate ourselves instead of relying on provider-specific
    # structured output features. This keeps the contract stable across LLM backends.
    try:
        response = llm.invoke([system_role, user_request])
    except Exception as e:
        # Common failure: requesting a model name that isn't available for the
        # Generative Language API key (404 NOT_FOUND). Encourage discovery via
        # ListModels and selecting a supported model.
        raise RuntimeError(
            "Gemini call failed. This is commonly caused by an unsupported model name.\n"
            f"- Requested GEMINI_MODEL={model_name!r}\n"
            "- Tip: set GEMINI_MODEL to a model your key supports (e.g. 'gemini-1.5-flash' or 'gemini-pro').\n"
            "- If unsure, list models with:\n"
            "    PYTHONPATH=src python -c \"import os; from google import genai; "
            "c=genai.Client(api_key=os.environ['GOOGLE_API_KEY']); "
            "print([m.name for m in c.models.list()][:30])\"\n"
            "\n"
            "Original error:\n"
            f"{e}"
        ) from e
    payload = _extract_json_object(getattr(response, "content", str(response)))
    return SupervisorPlan.model_validate(payload)
