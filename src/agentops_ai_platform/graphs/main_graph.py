"""
Main LangGraph wiring for AgentOps AI.

This module intentionally contains only:
- The global graph state contract (TypedDict)
- Minimal LangGraph imports
- A placeholder graph builder

Agent implementations and node logic are added elsewhere and wired in later.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

try:
    from typing import NotRequired
except ImportError:  # Python < 3.11
    from typing_extensions import NotRequired

from langgraph.graph import END, StateGraph


class GraphState(TypedDict):
    """
    Global state shared across the LangGraph run.

    Why a single state object?
    - It makes the workflow explicit: each node reads/writes specific fields.
    - It improves observability: state transitions can be traced and debugged.
    - It enforces contracts between planning, research, execution, and evaluation.
    """

    # The user's objective in a normalized, actionable form.
    # This is the anchor for planning and for evaluation criteria.
    user_goal: str

    # Explicit routing decision produced by the Supervisor's plan.
    # This allows deterministic branching (research vs execution) without heuristics.
    requires_research: bool | None

    # A step-by-step plan derived from the user_goal (or None if not planned yet).
    # Represented as simple strings initially; can evolve to structured steps later.
    plan: list[str] | None

    # Explicit definition of “done” for this run (or None until planned).
    # The evaluator must grade ONLY against these criteria to keep evaluation deterministic.
    success_criteria: list[str] | None

    # Structured research output (or None if research hasn't run).
    # Stored as a dict (ResearchResult-compatible) to preserve provenance and uncertainty.
    research_results: dict[str, Any] | None

    # The current draft output being iterated on (or None until generation occurs).
    draft_output: str | None

    # Optional execution audit trail (useful for observability and debugging).
    # This stores the Execution Agent's high-level action log (not raw tool payloads).
    execution_actions: NotRequired[list[str]]

    # Latest evaluation result (or None until evaluated).
    # Kept as a dict to allow flexible rubric/checklist outputs early on.
    evaluation: dict[str, Any] | None

    # How many plan/execute/evaluate cycles have occurred.
    # Used to bound loops and enforce termination conditions.
    iteration_count: int

    # Whether memory was used in planning (set by Supervisor).
    # Used for API response and observability (not for routing).
    memory_used: bool


MAX_ITERATIONS = 2
"""
Hard safety cap to prevent infinite loops during plan→act→evaluate cycles.

This limit is intentionally small for the initial skeleton. In production this
will typically be configurable per workflow and/or request.

Why fail-fast?
- Infinite retries waste tokens/time and can amplify hallucinations by repeatedly “trying again”.
- A hard cap forces the system to surface a clear failure reason and stop deterministically,
  which is critical for production reliability and debuggability.
"""


def supervisor_node(state: GraphState) -> GraphState:
    """
    Placeholder supervisor/orchestrator node.

    In the real system this node will route between specialist nodes and enforce
    policies/budgets. For now it only logs that it was invoked.
    """

    print("supervisor_node called")  # structural placeholder (replace with logging later)

    # Ensure a stable loop counter even if callers provide a partial initial state.
    if "iteration_count" not in state:
        state["iteration_count"] = 0

    # Memory: Check for relevant past successes before planning.
    # This enables the Supervisor to learn from past tasks and avoid redundant research.
    #
    # Memory is OPTIONAL and ADVISORY:
    # - If memory retrieval fails, planning proceeds normally
    # - If no relevant memories found, planning proceeds normally
    # - Supervisor can choose to ignore memory context
    # - Memory never blocks or delays the user response
    user_goal = state["user_goal"]
    memory_context: Optional[str] = None
    num_memories: int = 0  # Track count for observability (not content)

    try:
        from memory.memory_store import find_relevant_memories

        # Search for up to 3 similar past tasks
        relevant_memories = find_relevant_memories(user_goal, max_results=3)

        if relevant_memories:
            # Track count for observability metadata (never log actual memory content)
            num_memories = len(relevant_memories)

            # Create a concise summary of past successful approaches
            # Format: one line per memory showing goal + brief summary
            memory_lines = []
            for i, mem in enumerate(relevant_memories, 1):
                # Each line: "Past task: <goal> → <summary>"
                summary_snippet = mem.summary[:80].strip()
                if len(mem.summary) > 80:
                    summary_snippet += "..."
                memory_lines.append(f"{i}. {mem.user_goal} → {summary_snippet}")

            memory_context = "\n".join(memory_lines)

            # Log memory usage for observability (optional)
            print(f"💡 Found {num_memories} relevant memories")

    except Exception as e:
        # Memory retrieval failed, but this is not critical
        # Continue with planning without memory context
        print(f"⚠️  Memory retrieval failed (non-critical): {type(e).__name__}: {e}")
        memory_context = None
        num_memories = 0

    # Generate a structured plan from the user's goal (no tool usage here).
    # Import locally to avoid forcing LLM dependencies at import time for this module.
    from agentops_ai_platform.agents.supervisor_agent import generate_plan

    # Observability: wrap planning in a span.
    # We only attach minimal metadata (no prompts, no full plan text) to reduce leakage risk.
    #
    # NAMING CONVENTION:
    # All trace/span names follow the format "agentops.<agent_name>" for consistent observability.
    # This enables:
    # - Easy filtering in dashboards (e.g., "agentops.*" shows all agent traces)
    # - Consistent grouping across different observability backends (Langfuse, LangSmith)
    # - Clear namespace separation from other services in multi-service deployments
    #
    # MEMORY METADATA:
    # We track WHETHER memory influenced planning, NOT the actual memory contents.
    # This protects user privacy and keeps traces lightweight while still enabling analysis:
    # - Did memory retrieval work? (memory_used: true/false)
    # - How many memories were considered? (num_memories_considered: int)
    # - Did memory affect plan quality? (compare with memory_used=false baseline)
    from observability.trace_utils import traced

    trace_meta: dict[str, Any] = {
        "system_version": "v0.1",  # Track version for A/B tests and rollback correlation
        "user_goal": state["user_goal"],
        "memory_used": memory_context is not None,  # Whether memory influenced planning
        "num_memories_considered": num_memories,  # Count of memories retrieved (not content)
    }

    # Prepare the planning input with optional memory context.
    # We append memory as supplementary context to the user_goal.
    # This allows generate_plan to use the information without changing its signature.
    #
    # The Supervisor can choose to:
    # - Adopt a similar plan structure from memory
    # - Skip research if a similar task succeeded without it
    # - Ignore memory if the current goal differs meaningfully
    if memory_context:
        planning_input = f"""{user_goal}

[CONTEXT: Relevant past successful tasks]
{memory_context}
"""
    else:
        planning_input = user_goal

    with traced(name="agentops.supervisor", metadata=trace_meta):
        supervisor_plan = generate_plan(planning_input)
        trace_meta.update(
            {
                "requires_research": supervisor_plan.requires_research,
                "number_of_plan_steps": len(supervisor_plan.plan_steps),
            }
        )

    # Persist the plan steps into the graph state (drives downstream execution).
    state["plan"] = supervisor_plan.plan_steps

    # Persist success criteria so the evaluator can grade deterministically.
    state["success_criteria"] = supervisor_plan.success_criteria

    # Persist the explicit routing decision (drives supervisor→research/execute branching).
    state["requires_research"] = supervisor_plan.requires_research

    # Persist memory usage flag for API response (not for routing)
    # This tracks whether memory influenced planning (for observability and API)
    state["memory_used"] = memory_context is not None

    return state


def research_node(state: GraphState) -> GraphState:
    """
    Placeholder research node.

    In the real system this node will populate `research_results` using approved
    retrieval sources. For now it only logs that it was invoked.
    """

    print("research_node called")  # structural placeholder (replace with logging later)

    # Read the research intent from the Supervisor’s plan steps.
    # We look for steps that explicitly mention research/gathering context.
    plan_steps = state.get("plan") or []
    research_intent_steps = [
        step
        for step in plan_steps
        if any(keyword in step.lower() for keyword in ("research", "gather", "look up", "find"))
    ]

    # Form a clear research question for the Research Agent:
    # - anchored to the user_goal
    # - aligned with any explicit research intent present in the plan
    intent = research_intent_steps[0] if research_intent_steps else "Gather factual background needed to address the goal."
    research_question = (
        f"{intent}\n\n"
        f"User goal: {state['user_goal']}\n"
        "Provide grounded definitions, key concepts, major tradeoffs, and important caveats. "
        "Include concrete sources when possible."
    )

    # Execute the research-only agent (no tools; returns a structured ResearchResult).
    from agentops_ai_platform.agents.research_agent import conduct_research

    # Observability: wrap research in a span.
    # Keep metadata lightweight; include the research question but truncate to reduce payload size.
    #
    # Using consistent "agentops.research" naming for cross-agent trace filtering.
    from observability.trace_utils import traced

    trace_meta: dict[str, Any] = {
        "system_version": "v0.1",
        "research_question": research_question[:500],
    }
    with traced(name="agentops.research", metadata=trace_meta):
        research_result = conduct_research(research_question)
        trace_meta.update(
            {
                "num_sources": len(research_result.sources),
                "has_confidence_notes": bool(research_result.confidence_notes.strip()),
            }
        )

    # Store the full structured result so downstream nodes can reference it deterministically.
    state["research_results"] = research_result.model_dump()
    return state


def execution_node(state: GraphState) -> GraphState:
    """
    Placeholder execution node.

    In the real system this node will run tool-using steps and update
    `draft_output` and/or `tool_results`. For now it only logs that it was invoked.
    """

    print("execution_node called")  # structural placeholder (replace with logging later)

    user_goal = state["user_goal"]
    plan_steps = state.get("plan") or []

    # Convert structured research results into a compact textual context blob for the executor.
    # (Executor is forbidden from doing new research; it must rely on what is provided here.)
    rr = state.get("research_results")
    if rr is None:
        research_context: str | None = None
    else:
        summary = rr.get("summary", "")
        key_points = rr.get("key_points", []) or []
        sources = rr.get("sources", []) or []
        confidence_notes = rr.get("confidence_notes", "")

        research_context = (
            "RESEARCH SUMMARY:\n"
            f"{summary}\n\n"
            "KEY POINTS:\n"
            + "\n".join(f"- {p}" for p in key_points)
            + "\n\n"
            "SOURCES:\n"
            + "\n".join(f"- {s}" for s in sources)
            + ("\n\nCONFIDENCE NOTES:\n" + confidence_notes if confidence_notes else "")
        )

    # Execute using the execution-only agent (no tools; follows plan + uses provided research).
    from agentops_ai_platform.agents.execution_agent import execute_task

    # Observability: wrap execution in a span.
    # Do not log output text; only log sizes/counters.
    #
    # Consistent naming enables filtering like "agentops.execution" across all workflows.
    from observability.trace_utils import traced

    trace_meta: dict[str, Any] = {
        "system_version": "v0.1",
        "num_plan_steps": len(plan_steps),
    }
    with traced(name="agentops.execution", metadata=trace_meta):
        result = execute_task(
            user_goal=user_goal,
            plan_steps=plan_steps,
            research_results=research_context,
        )
        trace_meta.update(
            {
                "output_length": len(result.output),
                "num_assumptions": len(result.assumptions),
            }
        )

    # Store the draft output for downstream evaluation/iteration.
    state["draft_output"] = result.output

    # Optional: store a high-level action log for later observability and debugging.
    state["execution_actions"] = result.actions_taken
    return state


def evaluator_node(state: GraphState) -> GraphState:
    """
    Placeholder evaluator node.

    In the real system this node will validate `draft_output` and write an
    `evaluation` dict. We increment `iteration_count` here to represent the end
    of a plan→act→evaluate cycle (avoids over-counting if multiple nodes run).
    """

    print("evaluator_node called")  # structural placeholder (replace with logging later)

    # Ensure a stable loop counter even if callers provide a partial initial state.
    if "iteration_count" not in state:
        state["iteration_count"] = 0

    user_goal = state["user_goal"]
    draft_output = state.get("draft_output") or ""
    success_criteria = state.get("success_criteria") or []

    # Convert structured research results into a compact textual context blob for evaluation.
    rr = state.get("research_results")
    if rr is None:
        research_context: str | None = None
    else:
        summary = rr.get("summary", "")
        key_points = rr.get("key_points", []) or []
        sources = rr.get("sources", []) or []
        confidence_notes = rr.get("confidence_notes", "")
        research_context = (
            "RESEARCH SUMMARY:\n"
            f"{summary}\n\n"
            "KEY POINTS:\n"
            + "\n".join(f"- {p}" for p in key_points)
            + "\n\n"
            "SOURCES:\n"
            + "\n".join(f"- {s}" for s in sources)
            + ("\n\nCONFIDENCE NOTES:\n" + confidence_notes if confidence_notes else "")
        )

    from agentops_ai_platform.agents.evaluator_agent import evaluate_output

    # Observability: wrap evaluation in a span.
    # Keep metadata lightweight; avoid including the draft text or full research context.
    #
    # Consistent naming ("agentops.evaluator") is critical for:
    # - Aggregating metrics across all evaluator runs (pass rates, avg scores)
    # - Identifying bottlenecks in the evaluation loop
    # - Correlating retries with specific agent versions
    from observability.trace_utils import traced

    # iteration_count logged here is the cycle that is being completed by this evaluator call.
    current_iter = state.get("iteration_count", 0)
    trace_meta: dict[str, Any] = {
        "system_version": "v0.1",
        "iteration_count": current_iter + 1,
    }
    with traced(name="agentops.evaluator", metadata=trace_meta) as trace_handle:
        evaluation_result = evaluate_output(
            user_goal=user_goal,
            draft_output=draft_output,
            success_criteria=success_criteria,
            research_results=research_context,
        )
        trace_meta.update(
            {
                "pass": bool(evaluation_result.pass_),
                "score": int(evaluation_result.score),
                "num_detected_issues": len(evaluation_result.detected_issues),
            }
        )

        # Log evaluation metrics to Langfuse, attached to this trace.
        langfuse_tid = getattr(trace_handle, "langfuse_trace_id", None)
        from observability.langfuse import (
            log_evaluation_score,
            log_evaluation_pass,
            log_retry_count,
        )

        log_evaluation_score(score=evaluation_result.score, trace_id=langfuse_tid)
        log_evaluation_pass(pass_value=evaluation_result.pass_, trace_id=langfuse_tid)
        log_retry_count(iteration_count=current_iter + 1, trace_id=langfuse_tid)

    # Store the full structured evaluation.
    # IMPORTANT: use by_alias=True so the dict contains a literal "pass" key,
    # which is what the existing routing function `_evaluation_passed` expects.
    state["evaluation"] = evaluation_result.model_dump(by_alias=True)

    # Memory: Save successful high-quality outputs to memory for future planning.
    # This enables the Supervisor to learn from past successes and avoid redundant research.
    #
    # CRITICAL SAFETY REQUIREMENT:
    # Memory writes MUST NEVER crash the graph. If memory storage fails, log the error
    # and continue. The user's response is more important than storing memory.
    #
    # Memory is ONLY written when:
    # - Evaluation passed (pass = True)
    # - High quality score (score >= 8)
    # - This prevents polluting memory with failed or mediocre outputs
    if evaluation_result.pass_ and evaluation_result.score >= 7:
        try:
            from datetime import datetime, timezone
            import uuid
            from memory.memory_store import MemoryRecord, save_memory

            # Create a brief summary for memory (max 1000 chars)
            # Option 1: Use first 1-2 sentences of output
            # Option 2: Use evaluator's reasons as a summary
            # We'll use Option 2 (evaluator reasons) as it's more semantic
            if evaluation_result.reasons:
                # Evaluator reasons capture why it passed, which is valuable for learning
                summary_text = " ".join(evaluation_result.reasons)
            else:
                # Fallback: use first 200 chars of output as summary
                summary_text = draft_output[:200].strip()
                if len(draft_output) > 200:
                    summary_text += "..."

            # Ensure summary meets minimum length (50 chars) required by MemoryRecord
            if len(summary_text) < 50:
                # Pad with a generic completion note
                summary_text = f"Task completed successfully. {summary_text}"

            # Create the memory record
            memory_record = MemoryRecord(
                id=str(uuid.uuid4()),  # Unique ID for this memory entry
                user_goal=user_goal,
                summary=summary_text[:1000],  # Enforce max length
                final_output=draft_output[:2000],  # Truncate to prevent bloat
                score=evaluation_result.score,
                created_at=datetime.now(timezone.utc),
            )

            # Save to memory (returns False on error, never raises)
            if save_memory(memory_record):
                # Optional: log success for debugging
                pass  # Success is silent (avoid log noise)
            else:
                # Memory save failed, but this is not critical
                # The error was already logged by save_memory()
                pass

        except Exception as e:
            # Catch any unexpected errors (Pydantic validation, UUID generation, etc.)
            # Memory failures must NEVER crash the graph or block user responses
            print(f"⚠️  Failed to save memory (non-critical): {type(e).__name__}: {e}")
            # Continue execution - memory is optional, user response is mandatory

    # Increment at end of execute→evaluate cycle.
    state["iteration_count"] += 1

    # Enforce max retry limit: if we are still failing at/after the cap, make sure
    # the terminal state carries an explicit reason for stopping.
    if state["evaluation"].get("pass") is False and state["iteration_count"] >= MAX_ITERATIONS:
        # Fail-fast is better than infinite retries because repeated attempts without new
        # information/tools tend to drift, waste budget, and reduce debuggability.
        state["evaluation"].setdefault("detected_issues", [])
        state["evaluation"]["detected_issues"].append("Max retries reached")
        state["evaluation"].setdefault("reasons", [])
        state["evaluation"]["reasons"].append("Max retries reached (iteration cap hit).")
    return state


def _route_from_supervisor(state: GraphState) -> Literal["research", "execute"]:
    """
    Conditional routing function for the supervisor node.

    For the initial scaffold, we route to research when no research has been
    performed yet. Later this will be based on intent classification, tool
    policy, and whether external facts are required.
    """

    # Prefer the Supervisor’s explicit decision when present.
    if state.get("requires_research") is True:
        return "research"
    if state.get("requires_research") is False:
        return "execute"

    # Fallback heuristic for early scaffolding (before planning logic is fully wired).
    needs_research = state.get("research_results") is None
    return "research" if needs_research else "execute"


def _evaluation_passed(state: GraphState) -> bool:
    """
    Interprets the evaluation dict as a simple pass/fail signal.

    Contract (initial): `state["evaluation"]` is expected to be a dict containing
    a boolean `pass` key. Missing/None evaluations default to failing closed.
    """

    evaluation: dict[str, Any] | None = state.get("evaluation")
    if not evaluation:
        return False
    return bool(evaluation.get("pass", False))


def _route_from_evaluator(state: GraphState) -> Literal["end", "execute"]:
    """
    Conditional routing function for the evaluator node.

    - If evaluation passes, terminate the run (END).
    - If evaluation fails, go back to execution for a repair attempt.
    - If evaluation fails AND MAX_ITERATIONS is reached, terminate to prevent loops.
    """

    # Success always terminates immediately.
    if _evaluation_passed(state):
        return "end"

    # Fail-fast guard: stop once we’ve exhausted the allowed retry budget.
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "end"

    # Otherwise, loop back to execution for a targeted repair attempt.
    return "execute"


def build_main_graph() -> StateGraph:
    """
    Create the main workflow graph (nodes/edges wired elsewhere later).

    This returns the *builder* so the application can register nodes and edges
    without importing agent logic into this module.
    """

    graph = StateGraph(GraphState)

    # Register placeholder nodes (no agent logic yet).
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", research_node)
    graph.add_node("execute", execution_node)
    graph.add_node("evaluate", evaluator_node)

    # Entry point: always start with the supervisor/router.
    graph.set_entry_point("supervisor")

    # Conditional routing: supervisor decides whether to research or execute.
    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "research": "research",
            "execute": "execute",
        },
    )

    # Core path edges.
    graph.add_edge("research", "execute")
    graph.add_edge("execute", "evaluate")

    # Conditional routing: evaluator either ends (pass) or loops back to execute (fail),
    # with a hard MAX_ITERATIONS cap as a safety termination condition.
    graph.add_conditional_edges(
        "evaluate",
        _route_from_evaluator,
        {
            "end": END,
            "execute": "execute",
        },
    )

    return graph


if __name__ == "__main__":
    # Structural smoke test:
    # - create a minimal initial state
    # - compile and run the graph
    # - print the final state after routing/looping completes

    initial_state: GraphState = {
        "user_goal": "Draft a concise technical overview of this system.",
        "requires_research": None,
        "plan": None,
        "research_results": None,
        "draft_output": None,
        # Seed a passing evaluation so the skeleton terminates cleanly after the
        # first evaluate step. (Real evaluation logic will populate this.)
        "evaluation": {"pass": True},
        "iteration_count": 0,
        "memory_used": False,
    }

    app = build_main_graph().compile()
    final_state = app.invoke(initial_state)
    print("Final state:")
    print(final_state)
