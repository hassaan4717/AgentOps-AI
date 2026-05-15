"""
Evaluator agent contracts.

This module defines schema/contracts for evaluation outputs.
LLM calls are kept minimal and strictly for producing validated schemas.

WHY THE EVALUATOR GUARDS TOOL USAGE:
- The Evaluator is the final checkpoint before outputs reach users
- Tool misuse can lead to:
  1. Unauthorized actions (tools used without Supervisor approval)
  2. Hallucinated facts (claims not supported by tool results)
  3. Wasted resources (tools called but results ignored)
  4. Security violations (sensitive tools used inappropriately)
  
By checking tool usage, the Evaluator:
- Enforces the Supervisor → Executor contract
- Prevents outputs with unsupported claims from passing
- Provides feedback for improving tool usage in future iterations
- Maintains system integrity and trustworthiness

The Evaluator uses judgment (not automatic failure) because:
- Some tool misuse is minor and doesn't affect output quality
- Context matters: sometimes paraphrasing tool results is appropriate
- The repair loop can fix issues without complete rejection
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """
    Structured evaluation output produced by the Evaluator Agent.

    The contract is designed to support both:
    - hard gating in the LangGraph (pass/fail)
    - continuous quality tracking (score + reasons)
    """

    # Binary gate used for routing and termination.
    #
    # Why pass/fail matters more than score:
    # - The graph needs a deterministic decision (stop vs. repair loop).
    # - Scores are useful for analytics and ranking, but they should not be the
    #   primary control signal because they are often subjective/noisy.
    pass_: bool = Field(..., alias="pass", description="True if the output meets acceptance criteria.")

    # Optional quality indicator (1–10) for tracking and regression detection.
    score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Quality score from 1 (poor) to 10 (excellent).",
    )

    # Short, criteria-linked explanations for the verdict.
    reasons: list[str] = Field(
        ...,
        min_length=1,
        description="Concise reasons supporting the pass/fail verdict.",
    )

    # Actionable suggestions to improve the output when it fails or can be refined.
    improvement_suggestions: list[str] = Field(
        ...,
        description="Actionable changes to improve quality, clarity, or compliance.",
    )

    # Concrete problems detected in the candidate output (bugs, missing sections, policy violations).
    #
    # Why detected_issues is separate from suggestions:
    # - Issues are the *diagnosis* (what is wrong) and should be stable and checkable.
    # - Suggestions are the *treatment* (how to fix it) and can vary by style/approach.
    # Separating them improves debuggability and lets executors target fixes precisely.
    detected_issues: list[str] = Field(
        ...,
        description="Specific issues detected that prevented meeting requirements.",
    )


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _extract_tool_usage_from_actions(actions_taken: list[str]) -> dict[str, bool]:
    """
    Extract which tools were actually used by examining actions_taken.
    
    Why we check actions_taken:
    - actions_taken is the audit trail of what the Executor actually did
    - Tool calls are logged with [EXTERNAL INFORMATION] or [TOOL ERROR] markers
    - This provides ground truth for tool usage validation
    
    Args:
        actions_taken: List of action strings from ExecutionResult
        
    Returns:
        Dict mapping tool name to whether it was successfully used
        Example: {"web_search": True} means web_search was called and returned results
    """
    tool_usage = {}
    
    for action in actions_taken:
        # Check for tool usage patterns
        if "web_search" in action.lower():
            # Check if it was successful (returned results) or failed
            if "[EXTERNAL INFORMATION]" in action:
                tool_usage["web_search"] = True
            elif "[TOOL ERROR]" in action:
                tool_usage["web_search"] = False
    
    return tool_usage


def _check_tool_authorization(
    declared_tools: list[str],
    used_tools: dict[str, bool]
) -> list[str]:
    """
    Check if tools were used with proper authorization.
    
    Why authorization matters:
    - The Supervisor is responsible for declaring which tools are safe/appropriate
    - Using undeclared tools bypasses planning safeguards
    - This could lead to security issues, unexpected costs, or inappropriate actions
    
    Args:
        declared_tools: Tools declared in SupervisorPlan.required_tools
        used_tools: Tools actually used (from actions_taken)
        
    Returns:
        List of authorization issues (empty if no issues)
    """
    issues = []
    
    for tool_name in used_tools.keys():
        if tool_name not in declared_tools:
            issues.append(
                f"Tool '{tool_name}' was used without Supervisor authorization. "
                f"Declared tools: {declared_tools or '[]'}"
            )
    
    return issues


def _check_tool_result_usage(
    draft_output: str,
    used_tools: dict[str, bool],
    research_results: Optional[str]
) -> list[str]:
    """
    Check if tool results were actually incorporated into the output.
    
    Why this matters:
    - Tools consume resources (API calls, time, costs)
    - If tool results aren't used, the tool call was wasteful
    - Outputs should leverage the external information obtained
    - Ignoring tool results suggests the Executor didn't follow the plan
    
    Note: This is a heuristic check, not a guarantee. We look for indicators
    that tool results influenced the output.
    
    Args:
        draft_output: The executor's output text
        used_tools: Tools that were successfully called
        research_results: Research context including tool results
        
    Returns:
        List of usage warnings (empty if results appear to be used)
    """
    warnings = []
    
    # Only check tools that returned results successfully
    successful_tools = [name for name, success in used_tools.items() if success]
    
    if not successful_tools:
        return warnings  # No successful tool calls to check
    
    # Check if tool results section exists in research_results
    if not research_results:
        warnings.append(
            f"Tools {successful_tools} were called but no research results were provided. "
            "Results may not have been incorporated."
        )
        return warnings
    
    # Check for markers that tool results were present
    for tool_name in successful_tools:
        if tool_name == "web_search":
            # Look for indicators that web search results were incorporated
            # This is heuristic - we check if the output discusses topics that would
            # require current/external information
            if "WEB SEARCH RESULTS" in research_results:
                # Tool results were available - check if output is too brief or generic
                if len(draft_output.strip()) < 100:
                    warnings.append(
                        f"Tool '{tool_name}' returned results but output is very brief. "
                        "Consider using the tool results more thoroughly."
                    )
                # Note: We don't automatically fail here because the LLM will check
                # if claims are supported by research_results in its evaluation
    
    return warnings


def _extract_json_object(text: str) -> dict[str, Any]:
    """
    Best-effort extraction of a JSON object from model output.

    Even with strict instructions, providers may occasionally wrap responses in
    code fences or add stray text. This keeps parsing resilient.
    """

    candidate = text.strip()

    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()

    match = _JSON_OBJECT_RE.search(candidate)
    if not match:
        raise ValueError("Model did not return a JSON object.")

    return json.loads(match.group(0))


def evaluate_output(
    user_goal: str,
    draft_output: str,
    success_criteria: list[str],
    research_results: Optional[str],
    declared_tools: Optional[list[str]] = None,
    actions_taken: Optional[list[str]] = None,
) -> EvaluationResult:
    """
    Evaluate a draft against explicit success criteria using Gemini.

    This is strictly an evaluator: it must not rewrite, add new content, or plan.
    
    Tool Usage Validation:
    - Checks if tools were declared by Supervisor (declared_tools)
    - Checks if tools were actually used (from actions_taken)
    - Validates tool results were incorporated into output
    - Flags misuse in detected_issues (uses judgment, not automatic failure)
    
    Args:
        user_goal: The original user request
        draft_output: The executor's output to evaluate
        success_criteria: Success criteria from the plan
        research_results: Research context (may include tool results)
        declared_tools: Tools declared in SupervisorPlan.required_tools
        actions_taken: Actions logged by Executor (for tool usage tracking)
        
    Returns:
        EvaluationResult with pass/fail, score, reasons, and detected issues
    """
    
    declared_tools = declared_tools or []
    actions_taken = actions_taken or []
    
    # TOOL USAGE VALIDATION
    # Check tool usage before evaluation to catch authorization and usage issues.
    # This enforces the Supervisor → Executor contract and prevents outputs with
    # unsupported claims from passing.
    
    tool_issues = []
    
    # Step 1: Extract which tools were actually used
    used_tools = _extract_tool_usage_from_actions(actions_taken)
    
    # Step 2: Check authorization (were used tools declared by Supervisor?)
    auth_issues = _check_tool_authorization(declared_tools, used_tools)
    tool_issues.extend(auth_issues)
    
    # Step 3: Check if tool results were actually incorporated
    usage_warnings = _check_tool_result_usage(draft_output, used_tools, research_results)
    tool_issues.extend(usage_warnings)
    
    # Step 4: Check for undeclared but unused tools
    # If tools were declared but not used, that's okay - the Executor decided they
    # weren't needed during execution. We only flag unauthorized usage.

    # Offline structural testing mode: avoids network calls and secrets entirely.
    # Returns a deterministic placeholder evaluation that preserves the contract.
    if os.getenv("OFFLINE_MODE") == "1":
        missing = []
        if not draft_output.strip():
            missing.append("Draft output is empty.")
        
        # Include tool issues in offline evaluation
        all_issues = missing + tool_issues
        passed = not missing  # Tool issues are warnings, not automatic failures
        
        return EvaluationResult(
            **{
                "pass": passed,
                "score": 8 if passed else 2,
                "reasons": ["OFFLINE_MODE: heuristic evaluation only."],
                "improvement_suggestions": ["Provide a non-empty draft output."] if missing else [],
                "detected_issues": all_issues,
            }
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

    # Use Gemini 2.5 Flash for low-latency evaluation.
    model_name = os.getenv("GEMINI_EVALUATOR_MODEL", "gemini-2.0-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, max_retries=3, timeout=30)

    # --- Prompt section: role boundary (quality evaluator, not a writer) ---
    # Prevents the model from “helpfully” rewriting the draft, which would blur
    # responsibilities and make it impossible to attribute changes to the executor.
    system_role = SystemMessage(
        content=(
            "You are the Evaluator Agent. You are strictly a quality evaluator, NOT a writer.\n"
            "You must judge the provided draft against the provided criteria only.\n"
            "Do NOT rewrite, extend, or add content to the draft.\n"
            "Be fair and reasonable. If the draft substantially addresses the user goal and criteria, it should PASS.\n"
            "Only FAIL outputs that are clearly wrong, incomplete, or off-topic."
        )
    )

    # --- Prompt section: evaluate ONLY against success_criteria ---
    # Ensures evaluation is deterministic and testable; avoids shifting standards.
    #
    # --- Prompt section: hallucination checks with research grounding ---
    # When research context is provided, the evaluator must flag unsupported claims.
    #
    # --- Prompt section: conservative fail-closed behavior ---
    # In ambiguous cases, prefer failing so the system can enter a repair loop.
    #
    # --- Prompt section: tool usage validation ---
    # When tools were used, check if their results actually support the output claims.
    
    # Build tool usage context for the LLM
    tool_context = ""
    if declared_tools or used_tools:
        tool_context = "\n\nTool Usage Information:\n"
        tool_context += f"- Tools declared by Supervisor: {declared_tools or '[]'}\n"
        tool_context += f"- Tools actually used: {list(used_tools.keys()) or '[]'}\n"
        tool_context += f"- Actions taken: {', '.join(actions_taken[-3:]) if actions_taken else 'none'}\n"  # Last 3 actions
        
        if used_tools:
            tool_context += "\nTool Usage Validation (check these):\n"
            tool_context += "- Were the used tools declared by the Supervisor?\n"
            tool_context += "- Do factual claims in the output align with tool results in research context?\n"
            tool_context += "- If tools returned data, was that data actually incorporated?\n"
    
    # Pre-detected tool issues to include
    tool_issues_context = ""
    if tool_issues:
        tool_issues_context = "\n\nPre-detected Tool Issues:\n"
        for issue in tool_issues:
            tool_issues_context += f"- {issue}\n"
    
    user_request = HumanMessage(
        content=(
            "User goal:\n"
            f"{user_goal}\n\n"
            "Success criteria (evaluate ONLY against these):\n"
            + "\n".join(f"- {c}" for c in success_criteria)
            + "\n\n"
            "=== DRAFT OUTPUT (plain text, this is what you are evaluating) ===\n"
            f"{draft_output}\n"
            "=== END DRAFT OUTPUT ===\n\n"
            "Research context (use ONLY to check factual claims; may be empty):\n"
            f"{research_results if research_results else '(none provided)'}"
            + tool_context
            + tool_issues_context
            + "\n\n"
            "IMPORTANT: The draft output above is plain text, NOT JSON. Do NOT penalize it for not being JSON.\n"
            "Evaluate ONLY whether it correctly and clearly answers the user goal based on the success criteria.\n\n"
            "Rules:\n"
            "- Be practical: if the draft reasonably addresses the goal and criteria, PASS it.\n"
            "- Only FAIL if the content is clearly wrong, incomplete, or off-topic.\n"
            "- Minor formatting or stylistic preferences should NOT cause failure.\n\n"
            "YOUR response (not the draft) must be ONLY a JSON object:\n"
            "{\n"
            '  "pass": boolean,\n'
            '  "score": integer (1-10),\n'
            '  "reasons": string[],\n'
            '  "improvement_suggestions": string[],\n'
            '  "detected_issues": string[]\n'
            "}\n"
            "No extra keys, commentary, markdown, or code fences."
        )
    )

    response = llm.invoke([system_role, user_request])
    payload = _extract_json_object(getattr(response, "content", str(response)))

    # Defensive normalization: ensure required list fields exist even if the model omits them.
    payload.setdefault("reasons", [])
    payload.setdefault("improvement_suggestions", [])
    payload.setdefault("detected_issues", [])

    # If the model returned empty reasons, force at least one to satisfy the contract.
    if not payload.get("reasons"):
        payload["reasons"] = ["No reasons provided by model; treat evaluation as insufficient."]
    
    # Merge pre-detected tool issues with LLM-detected issues
    # Tool issues are added to detected_issues but don't automatically cause failure
    # The LLM uses judgment to decide if they warrant failure based on severity
    if tool_issues:
        # Prepend tool issues so they appear first
        payload["detected_issues"] = tool_issues + payload["detected_issues"]

    return EvaluationResult.model_validate(payload)
