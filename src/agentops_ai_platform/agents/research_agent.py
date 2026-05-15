"""
Research agent contracts.

This module defines schema/contracts for research outputs.
LLM calls are kept minimal and strictly for producing validated schemas.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import BaseModel, Field


class ResearchResult(BaseModel):
    """
    Structured research output designed to reduce hallucination risk.

    The core idea: force the system to separate *what we think is true* (summary),
    *what matters* (key points), and *why we believe it* (sources + confidence notes).
    """

    # A concise synthesis of findings.
    # Hallucination risk reduction: requires the agent to compress into a coherent
    # statement that can later be cross-checked by the Evaluator and compared to sources.
    summary: str = Field(..., description="Concise synthesis of the most relevant findings.")

    # Bulletized claims extracted from the research.
    # Hallucination risk reduction: encourages atomic, checkable statements rather than
    # sprawling narratives; each point can be validated or rejected independently.
    key_points: list[str] = Field(
        ...,
        min_length=1,
        description="Atomic, checkable claims derived from the research.",
    )

    # Provenance pointers (URLs, doc IDs, internal paths).
    # Hallucination risk reduction: forces explicit attribution so downstream steps can
    # verify where claims came from and avoid “source-less” assertions.
    sources: list[str] = Field(
        ...,
        min_length=1,
        description="Source identifiers/URLs supporting the summary and key points.",
    )

    # Notes about uncertainty, gaps, and assumptions.
    # Hallucination risk reduction: makes uncertainty explicit, preventing the system
    # from presenting weakly-supported claims as facts and guiding follow-up research.
    confidence_notes: str = Field(
        ...,
        description="Uncertainty notes, assumptions, and gaps that affect confidence.",
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


def conduct_research(research_question: str) -> ResearchResult:
    """
    Perform research using Gemini and return a validated `ResearchResult`.

    This is research-only: no planning, no recommendations, and no final answers.
    """

    # Offline structural testing mode: avoids network calls and secrets entirely.
    # This returns a deterministic placeholder that preserves the contract shape.
    if os.getenv("OFFLINE_MODE") == "1":
        return ResearchResult(
            summary="OFFLINE_MODE: research is disabled; this is a placeholder summary.",
            key_points=[
                "OFFLINE_MODE enabled (no external retrieval performed).",
                "Replace this with grounded research results when online.",
            ],
            sources=["offline://no-sources"],
            confidence_notes="OFFLINE_MODE: confidence is unknown; no sources were consulted.",
        )

    # Import locally to keep module import-time light and to avoid requiring the
    # full LangChain provider stack when only schemas are used.
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Provider: Google Gemini (Generative Language API key).
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Missing GOOGLE_API_KEY. Set it in your environment, e.g.:\n"
            "  export GOOGLE_API_KEY=\"...\""
        )

    # Use Gemini 2.5 Flash for low-latency research synthesis.
    # Allow override for experiments, but default is the requested model.
    model_name = os.getenv("GEMINI_RESEARCH_MODEL", "gemini-2.0-flash")

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, max_retries=3, timeout=30)

    # --- Prompt section: role boundary (research-only) ---
    # Prevents the model from drifting into planning/execution or “final answer” mode.
    system_role = SystemMessage(
        content=(
            "You are the Research Agent. Your role is strictly research-only: "
            "collect factual, grounded information and summarize it with provenance. "
            "Do NOT plan, recommend actions, or write a final user-facing answer."
        )
    )

    # --- Prompt section: factual grounding & uncertainty ---
    # Forces grounded claims and explicitly requests uncertainty notes when evidence is weak.
    #
    # --- Prompt section: strict structured JSON output ---
    # Ensures downstream components can validate and use outputs deterministically.
    #
    # --- Prompt section: prohibitions (no planning/recommendations/final answers) ---
    # Keeps this component composable; Supervisor/Executor own planning/execution.
    user_request = HumanMessage(
        content=(
            "Research question:\n"
            f"{research_question}\n\n"
            "Return ONLY a JSON object matching this schema:\n"
            "{\n"
            '  \"summary\": string,\n'
            '  \"key_points\": string[],\n'
            '  \"sources\": string[],\n'
            '  \"confidence_notes\": string\n'
            "}\n\n"
            "Requirements:\n"
            "- Provide factual, grounded information; avoid speculation.\n"
            "- If something is uncertain, say so explicitly in confidence_notes.\n"
            "- sources must be concrete (URLs, doc IDs, or clear source descriptors).\n"
            "- Do NOT include planning steps, recommendations, or a final answer.\n"
            "- Do not include extra keys, commentary, markdown, or code fences."
        )
    )

    response = llm.invoke([system_role, user_request])
    payload = _extract_json_object(getattr(response, "content", str(response)))
    return ResearchResult.model_validate(payload)
