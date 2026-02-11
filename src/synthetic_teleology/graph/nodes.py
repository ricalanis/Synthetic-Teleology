"""LangGraph node functions for the teleological loop.

Each function takes a ``TeleologicalState`` and returns a partial update dict.
The nodes delegate to the service classes (LLM-backed or numeric) rather than
reimplementing any logic.

In LLM mode, nodes append reasoning to the ``reasoning_trace`` channel and
capture hypothesis metadata.  In numeric mode, nodes behave identically to
v0.2.x for backward compatibility.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from synthetic_teleology.domain.values import EvalSignal, StateSnapshot

logger = logging.getLogger(__name__)


def _build_enriched_observation(base_observation: str, state: dict[str, Any]) -> str:
    """Append action results, eval trends, and goal revision count to the observation.

    Returns ``base_observation`` unchanged on step 1 (no history exists yet).
    """
    step = state.get("step", 0)
    if step <= 1:
        return base_observation

    parts: list[str] = []

    # Recent action results (last 3)
    feedback = state.get("action_feedback", [])
    if feedback:
        recent = feedback[-3:]
        lines = []
        for fb in recent:
            result_str = str(fb.get("result", ""))[:300]
            tool = fb.get("tool_name") or "direct"
            lines.append(f"  - {fb.get('action', '?')} (via {tool}): {result_str}")
        parts.append("Recent action results:\n" + "\n".join(lines))

    # Eval score trend (last 5)
    eval_history = state.get("eval_history", [])
    if eval_history:
        recent_evals = eval_history[-5:]
        scores = [
            f"{getattr(e, 'score', e.get('score', 0) if isinstance(e, dict) else 0):.2f}"
            for e in recent_evals
        ]
        parts.append(f"Eval score trend: {' -> '.join(scores)}")

    # Goal revision count
    goal_history = state.get("goal_history", [])
    if goal_history:
        n = len(goal_history)
        parts.append(f"{n} goal revision(s) so far")

    if not parts:
        return base_observation

    enrichment = "\n\n".join(parts)
    return f"{base_observation}\n\n--- History ---\n{enrichment}"


def perceive_node(state: dict[str, Any]) -> dict[str, Any]:
    """Observe the environment and produce a StateSnapshot.

    Reads ``perceive_fn`` and ``step`` from state.
    Writes ``state_snapshot``, ``observation``, and incremented ``step``.

    After step 1, enriches the observation with action feedback, eval score
    trends, and goal revision counts so that downstream LLM services
    (evaluator, planner, reviser) see history through the ``{observation}``
    prompt variable.
    """
    perceive_fn = state["perceive_fn"]
    step = state.get("step", 0) + 1
    snapshot = perceive_fn()

    # Build observation text
    observation = getattr(snapshot, "observation", "") or ""
    if not observation and snapshot.values:
        observation = f"State values at step {step}: {snapshot.values}"

    # Enrich observation with history
    observation = _build_enriched_observation(observation, state)

    # Inject recent action results into snapshot context
    feedback = state.get("action_feedback", [])
    context = dict(snapshot.context) if snapshot.context else {}
    if feedback:
        context["recent_action_results"] = feedback[-3:]

    # Reconstruct snapshot with enriched observation + context (frozen dataclass)
    snapshot = StateSnapshot(
        timestamp=snapshot.timestamp,
        values=snapshot.values,
        observation=observation,
        context=context,
        source=snapshot.source,
        metadata=snapshot.metadata,
    )

    logger.debug("perceive_node: step=%d dim=%d", step, snapshot.dimension)
    return {
        "state_snapshot": snapshot,
        "observation": observation,
        "step": step,
    }


def evaluate_node(state: dict[str, Any]) -> dict[str, Any]:
    """Compute Delta(G_t, S_t) using the evaluator strategy.

    In LLM mode, the evaluator produces reasoning and criteria scores.
    In numeric mode, works the same as v0.2.x.

    Reads ``evaluator``, ``goal``, ``state_snapshot``.
    Writes ``eval_signal``, appends to ``eval_history``, ``events``,
    and ``reasoning_trace``.
    """
    evaluator = state["evaluator"]
    goal = state["goal"]
    snapshot = state["state_snapshot"]

    if evaluator.validate(goal, snapshot):
        signal = evaluator.evaluate(goal, snapshot)
    else:
        signal = EvalSignal(
            score=0.0,
            confidence=0.0,
            explanation="Evaluator validation failed",
        )

    logger.debug(
        "evaluate_node: score=%.4f conf=%.4f", signal.score, signal.confidence
    )

    event = {
        "type": "evaluation_completed",
        "step": state.get("step", 0),
        "score": signal.score,
        "confidence": signal.confidence,
        "timestamp": time.time(),
    }

    result: dict[str, Any] = {
        "eval_signal": signal,
        "eval_history": [signal],
        "events": [event],
    }

    # Append reasoning to trace if available (LLM mode)
    if signal.reasoning:
        result["reasoning_trace"] = [
            {
                "node": "evaluate",
                "step": state.get("step", 0),
                "reasoning": signal.reasoning,
                "timestamp": time.time(),
            }
        ]

    return result


def revise_node(state: dict[str, Any]) -> dict[str, Any]:
    """Optionally revise the goal based on the evaluation signal.

    In LLM mode, the LLMReviser reasons about whether revision is warranted
    and can revise the description/criteria as well as numeric values.

    Reads ``goal_updater``, ``goal``, ``state_snapshot``, ``eval_signal``.
    Writes ``goal``, appends to ``goal_history``, ``events``,
    and ``reasoning_trace``.
    """
    updater = state["goal_updater"]
    goal = state["goal"]
    snapshot = state["state_snapshot"]
    signal = state["eval_signal"]

    revised = updater.update(goal, snapshot, signal)

    if revised is not None:
        logger.info(
            "revise_node: goal revised %s -> %s",
            goal.goal_id,
            revised.goal_id,
        )
        event = {
            "type": "goal_revised",
            "step": state.get("step", 0),
            "previous_goal_id": goal.goal_id,
            "new_goal_id": revised.goal_id,
            "timestamp": time.time(),
        }
        result: dict[str, Any] = {
            "goal": revised,
            "goal_history": [revised],
            "events": [event],
        }

        # Capture revision reasoning in trace
        if hasattr(revised, "metadata") and revised.metadata.get("revision_reasoning"):
            result["reasoning_trace"] = [
                {
                    "node": "revise",
                    "step": state.get("step", 0),
                    "reasoning": revised.metadata["revision_reasoning"],
                    "timestamp": time.time(),
                }
            ]

        # Record to audit trail if available
        audit_trail = state.get("audit_trail")
        if audit_trail is not None:
            try:
                audit_trail.record(
                    goal_id=revised.goal_id,
                    previous_goal_id=goal.goal_id,
                    revision_reason=event.get("type", ""),
                    eval_score=signal.score,
                    eval_confidence=signal.confidence,
                    provenance=getattr(revised, "provenance", None),
                )
            except Exception as exc:
                logger.warning("revise_node: audit trail record failed: %s", exc)

        return result

    return {}


def ground_goal_node(state: dict[str, Any]) -> dict[str, Any]:
    """Apply intentional grounding from external directives.

    Only active when a ``grounding_manager`` is present in state.
    The grounding manager assesses accumulated directives and may
    adjust the goal description or criteria.

    Reads ``grounding_manager``, ``goal``, ``eval_signal``.
    Writes ``goal`` (if grounded), appends to ``reasoning_trace``.
    """
    grounding_manager = state.get("grounding_manager")
    if grounding_manager is None:
        return {}

    goal = state["goal"]
    signal = state.get("eval_signal")

    try:
        grounded_goal = grounding_manager.ground(goal, signal)
    except Exception as exc:
        logger.warning("ground_goal_node: grounding failed: %s", exc)
        return {}

    if grounded_goal is not None and grounded_goal.goal_id != goal.goal_id:
        logger.info("ground_goal_node: goal grounded %s -> %s", goal.goal_id, grounded_goal.goal_id)
        return {
            "goal": grounded_goal,
            "goal_history": [grounded_goal],
            "reasoning_trace": [
                {
                    "node": "ground_goal",
                    "step": state.get("step", 0),
                    "reasoning": "Goal adjusted by intentional grounding",
                    "timestamp": time.time(),
                }
            ],
        }
    return {}


def check_constraints_node(state: dict[str, Any]) -> dict[str, Any]:
    """Validate the current state against the constraint pipeline.

    Supports both legacy pipeline checkers and LLM constraint checkers.

    Reads ``constraint_pipeline``, ``goal``, ``state_snapshot``.
    Writes ``constraints_ok``, ``constraint_violations``,
    ``constraint_assessments``.
    """
    pipeline = state["constraint_pipeline"]
    goal = state["goal"]
    snapshot = state["state_snapshot"]

    ok, violations = pipeline.check_all(goal, snapshot)

    if not ok:
        logger.warning("check_constraints_node: violations=%s", violations)

    return {
        "constraints_ok": ok,
        "constraint_violations": violations,
    }


def plan_node(state: dict[str, Any]) -> dict[str, Any]:
    """Generate an action policy from the current goal and state.

    In LLM mode, generates multiple hypotheses with confidence scores.
    In numeric mode, works the same as v0.2.x.

    Reads ``planner``, ``goal``, ``state_snapshot``.
    Writes ``policy``, ``hypotheses``, ``selected_plan``, appends to
    ``events`` and ``reasoning_trace``.
    """
    planner = state["planner"]
    goal = state["goal"]
    snapshot = state["state_snapshot"]

    policy = planner.plan(goal, snapshot)
    logger.debug("plan_node: planned %d actions", policy.size)

    event = {
        "type": "plan_generated",
        "step": state.get("step", 0),
        "num_actions": policy.size,
        "timestamp": time.time(),
    }

    result: dict[str, Any] = {
        "policy": policy,
        "events": [event],
    }

    # Extract hypotheses if available (LLM mode)
    hypotheses_data = policy.metadata.get("hypotheses") if policy.metadata else None
    if hypotheses_data:
        result["reasoning_trace"] = [
            {
                "node": "plan",
                "step": state.get("step", 0),
                "num_hypotheses": len(hypotheses_data),
                "reasoning": policy.metadata.get("selection_reasoning", ""),
                "timestamp": time.time(),
            }
        ]

    return result


def filter_policy_node(state: dict[str, Any]) -> dict[str, Any]:
    """Filter the policy through the constraint pipeline.

    Reads ``policy_filter``, ``policy``, ``goal``, ``state_snapshot``.
    Writes ``filtered_policy``.
    """
    policy_filter = state["policy_filter"]
    policy = state["policy"]
    goal = state["goal"]
    snapshot = state["state_snapshot"]

    filtered = policy_filter.filter(policy, goal, snapshot)
    logger.debug("filter_policy_node: %d -> %d actions", policy.size, filtered.size)
    return {"filtered_policy": filtered}


def act_node(state: dict[str, Any]) -> dict[str, Any]:
    """Execute the filtered policy and trigger environment transition.

    In LLM mode, if the selected action has a ``tool_name``, invokes the
    corresponding LangChain tool from the tools list in state.

    Reads ``act_fn``, ``transition_fn``, ``filtered_policy``, ``state_snapshot``,
    ``tools``.
    Writes ``executed_action``, appends to ``action_history`` and ``events``.
    """
    act_fn = state.get("act_fn")
    transition_fn = state.get("transition_fn")
    filtered_policy = state["filtered_policy"]
    snapshot = state["state_snapshot"]
    tools = state.get("tools", [])

    # Select action
    if act_fn is not None:
        action = act_fn(filtered_policy, snapshot)
    else:
        action = filtered_policy.actions[0] if filtered_policy.size > 0 else None

    # Execute LangChain tool if action has tool_name
    tool_result = None
    if action is not None and getattr(action, "tool_name", None) and tools:
        tool_map = {getattr(t, "name", ""): t for t in tools}
        tool = tool_map.get(action.tool_name)
        if tool is not None:
            try:
                tool_result = tool.invoke(dict(action.parameters))
                logger.debug("act_node: tool '%s' returned: %s", action.tool_name, tool_result)
            except Exception as exc:
                logger.warning("act_node: tool '%s' failed: %s", action.tool_name, exc)

    # Transition — supports constraint-conditioned 2-arg signature
    if transition_fn is not None and action is not None:
        import inspect

        sig = inspect.signature(transition_fn)
        if len(sig.parameters) >= 2:
            constraints_context = {
                "constraints_ok": state.get("constraints_ok", True),
                "constraint_violations": state.get("constraint_violations", []),
                "constraint_assessments": state.get("constraint_assessments", []),
            }
            transition_fn(action, constraints_context)
        else:
            transition_fn(action)

    logger.debug(
        "act_node: executed action=%s", action.name if action else "None"
    )

    event: dict[str, Any] = {
        "type": "action_executed",
        "step": state.get("step", 0),
        "action_name": action.name if action else None,
        "timestamp": time.time(),
    }
    if tool_result is not None:
        event["tool_result"] = str(tool_result)[:500]

    # Build structured feedback for the perception-action loop
    feedback_entry = []
    if action is not None:
        feedback_entry = [{
            "action": action.name,
            "tool_name": getattr(action, "tool_name", None),
            "result": tool_result,
            "step": state.get("step", 0),
            "timestamp": time.time(),
        }]

    history_entry = [action] if action is not None else []
    return {
        "executed_action": action,
        "action_history": history_entry,
        "action_feedback": feedback_entry,
        "events": [event],
    }


def reflect_node(state: dict[str, Any]) -> dict[str, Any]:
    """Evaluate whether the loop should terminate.

    In LLM mode, also captures the full reasoning trace summary.

    Reads ``step``, ``goal``, ``eval_signal``, ``max_steps``,
    ``goal_achieved_threshold``, ``filtered_policy``.
    Writes ``events``, ``stop_reason`` (if termination warranted),
    and ``reasoning_trace``.
    """
    from synthetic_teleology.domain.enums import GoalStatus

    step = state.get("step", 0)
    goal = state["goal"]
    signal = state["eval_signal"]
    max_steps = state.get("max_steps", 100)
    threshold = state.get("goal_achieved_threshold", 0.9)
    filtered_policy = state.get("filtered_policy")

    stop_reason: str | None = None

    if step >= max_steps:
        stop_reason = "max_steps"
    elif goal.status == GoalStatus.ACHIEVED:
        stop_reason = "goal_achieved"
    elif goal.status == GoalStatus.ABANDONED:
        stop_reason = "goal_abandoned"
    elif signal.score >= threshold:
        goal.achieve()
        stop_reason = "goal_achieved"
    elif filtered_policy is not None and filtered_policy.size == 0:
        stop_reason = "empty_policy"

    event = {
        "type": "reflection",
        "step": step,
        "eval_score": signal.score,
        "stop_reason": stop_reason,
        "timestamp": time.time(),
    }

    result: dict[str, Any] = {"events": [event]}
    if stop_reason is not None:
        result["stop_reason"] = stop_reason

    # Log reflection in reasoning trace
    result["reasoning_trace"] = [
        {
            "node": "reflect",
            "step": step,
            "eval_score": signal.score,
            "stop_reason": stop_reason,
            "timestamp": time.time(),
        }
    ]

    return result
