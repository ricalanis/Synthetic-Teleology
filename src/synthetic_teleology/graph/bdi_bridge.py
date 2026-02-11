"""BDI-LangGraph bridge.

Factory functions that wrap BDIAgent methods as LangGraph node functions,
enabling BDI-augmented teleological graphs per Haidemariam (2026) Section 5.3.

The bridge maps the BDI deliberation cycle onto graph nodes:
- ``make_bdi_perceive_node`` — updates beliefs from environment snapshot
- ``make_bdi_revise_node`` — desire reconsideration + goal revision
- ``make_bdi_plan_node`` — intention reuse + planning

``build_bdi_teleological_graph`` compiles a full graph with BDI-augmented nodes.

Note: No ``from __future__ import annotations`` — LangGraph needs runtime
type resolution for TypedDict schemas.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def make_bdi_perceive_node(bdi_agent: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a perceive node that updates BDI agent beliefs.

    The node delegates to the standard perceive_fn but also updates
    the BDI agent's belief base with the resulting snapshot.

    Parameters
    ----------
    bdi_agent:
        A BDIAgent instance with a ``beliefs`` attribute and
        ``update_beliefs`` method (or equivalent).

    Returns
    -------
    Callable
        A LangGraph node function.
    """

    def bdi_perceive_node(state: dict[str, Any]) -> dict[str, Any]:
        perceive_fn = state["perceive_fn"]
        step = state.get("step", 0) + 1
        snapshot = perceive_fn()

        # Update BDI beliefs
        if hasattr(bdi_agent, "beliefs"):
            bdi_agent.beliefs["state_values"] = snapshot.values
            bdi_agent.beliefs["step"] = step
            bdi_agent.beliefs["observation"] = getattr(snapshot, "observation", "")

        observation = getattr(snapshot, "observation", "") or ""
        if not observation and snapshot.values:
            observation = f"State values at step {step}: {snapshot.values}"

        logger.debug("bdi_perceive_node: step=%d, beliefs updated", step)
        return {
            "state_snapshot": snapshot,
            "observation": observation,
            "step": step,
        }

    return bdi_perceive_node


def make_bdi_revise_node(bdi_agent: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a revise node that uses BDI desire reconsideration.

    The node first lets the BDI agent reconsider its desires based on
    current beliefs and evaluation signal. If desires change, the node
    performs goal revision. Otherwise, falls back to the standard updater.

    Parameters
    ----------
    bdi_agent:
        A BDIAgent instance with ``desires``, ``reconsider``,
        and ``revise_goal`` methods.

    Returns
    -------
    Callable
        A LangGraph node function.
    """

    def bdi_revise_node(state: dict[str, Any]) -> dict[str, Any]:
        updater = state["goal_updater"]
        goal = state["goal"]
        snapshot = state["state_snapshot"]
        signal = state["eval_signal"]

        # Update BDI beliefs with evaluation
        if hasattr(bdi_agent, "beliefs"):
            bdi_agent.beliefs["eval_score"] = signal.score
            bdi_agent.beliefs["eval_confidence"] = signal.confidence

        # Try BDI desire reconsideration
        bdi_revised = False
        if hasattr(bdi_agent, "reconsider") and hasattr(bdi_agent, "desires"):
            try:
                old_desires = (
                    list(bdi_agent.desires) if hasattr(bdi_agent.desires, "__iter__") else []
                )
                bdi_agent.reconsider(snapshot, signal)
                new_desires = (
                    list(bdi_agent.desires) if hasattr(bdi_agent.desires, "__iter__") else []
                )

                # Check if desires changed (different top desire)
                if old_desires and new_desires:
                    old_top = old_desires[0] if old_desires else None
                    new_top = new_desires[0] if new_desires else None
                    if old_top != new_top:
                        bdi_revised = True
                        logger.info("bdi_revise_node: BDI desire reconsideration triggered")
            except Exception as exc:
                logger.warning("bdi_revise_node: BDI reconsider failed: %s", exc)

        # If BDI reconsideration changed desires, use BDI revision
        if bdi_revised and hasattr(bdi_agent, "revise_goal"):
            try:
                bdi_goal = bdi_agent.revise_goal(snapshot, signal)
                if bdi_goal is not None and bdi_goal.goal_id != goal.goal_id:
                    event = {
                        "type": "bdi_goal_revised",
                        "step": state.get("step", 0),
                        "previous_goal_id": goal.goal_id,
                        "new_goal_id": bdi_goal.goal_id,
                        "source": "bdi_desire_reconsideration",
                        "timestamp": time.time(),
                    }
                    return {
                        "goal": bdi_goal,
                        "goal_history": [bdi_goal],
                        "events": [event],
                        "reasoning_trace": [
                            {
                                "node": "bdi_revise",
                                "step": state.get("step", 0),
                                "reasoning": "BDI desire reconsideration triggered goal revision",
                                "timestamp": time.time(),
                            }
                        ],
                    }
            except Exception as exc:
                logger.warning("bdi_revise_node: BDI revise_goal failed: %s", exc)

        # Fallback: standard updater
        revised = updater.update(goal, snapshot, signal)
        if revised is not None:
            event = {
                "type": "goal_revised",
                "step": state.get("step", 0),
                "previous_goal_id": goal.goal_id,
                "new_goal_id": revised.goal_id,
                "timestamp": time.time(),
            }
            return {
                "goal": revised,
                "goal_history": [revised],
                "events": [event],
            }

        return {}

    return bdi_revise_node


def make_bdi_plan_node(bdi_agent: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a plan node that leverages BDI intention reuse.

    The node first checks if the BDI agent has a reusable intention
    (current plan). If the intention is still valid for the current
    goal, it reuses it. Otherwise, falls back to the standard planner.

    Parameters
    ----------
    bdi_agent:
        A BDIAgent instance with ``intentions`` and ``plan`` methods.

    Returns
    -------
    Callable
        A LangGraph node function.
    """

    def bdi_plan_node(state: dict[str, Any]) -> dict[str, Any]:
        planner = state["planner"]
        goal = state["goal"]
        snapshot = state["state_snapshot"]

        # Try BDI intention reuse
        if hasattr(bdi_agent, "intentions") and bdi_agent.intentions is not None:
            try:
                intention = bdi_agent.intentions
                # Check if intention is still valid (has actions)
                if hasattr(intention, "actions") and intention.actions:
                    n_acts = len(intention.actions)
                    logger.debug("bdi_plan_node: reusing BDI intention (%d actions)", n_acts)
                    event = {
                        "type": "bdi_intention_reused",
                        "step": state.get("step", 0),
                        "num_actions": len(intention.actions),
                        "timestamp": time.time(),
                    }
                    return {
                        "policy": intention,
                        "events": [event],
                        "reasoning_trace": [
                            {
                                "node": "bdi_plan",
                                "step": state.get("step", 0),
                                "reasoning": "Reused existing BDI intention",
                                "timestamp": time.time(),
                            }
                        ],
                    }
            except Exception as exc:
                logger.warning("bdi_plan_node: BDI intention reuse failed: %s", exc)

        # Fallback: standard planner + update BDI intentions
        policy = planner.plan(goal, snapshot)

        if hasattr(bdi_agent, "intentions"):
            import contextlib

            with contextlib.suppress(Exception):
                bdi_agent.intentions = policy

        event = {
            "type": "plan_generated",
            "step": state.get("step", 0),
            "num_actions": policy.size,
            "timestamp": time.time(),
        }

        return {
            "policy": policy,
            "events": [event],
        }

    return bdi_plan_node


def build_bdi_teleological_graph(
    bdi_agent: Any,
    *,
    checkpointer: Any = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    evaluator: Any = None,
    constraint_pipeline: Any = None,
    policy_filter: Any = None,
) -> Any:
    """Build a LangGraph with BDI-augmented nodes.

    Replaces the standard perceive, revise, and plan nodes with
    BDI-aware versions that leverage belief updating, desire
    reconsideration, and intention reuse.

    Parameters
    ----------
    bdi_agent:
        A BDIAgent instance.
    checkpointer:
        Optional LangGraph checkpointer.
    interrupt_before:
        Nodes to interrupt before (human-in-the-loop).
    interrupt_after:
        Nodes to interrupt after.
    evaluator:
        Optional evaluator for closure injection (enables checkpointing).
    constraint_pipeline:
        Optional constraint pipeline for closure injection.
    policy_filter:
        Optional policy filter for closure injection.

    Returns
    -------
    CompiledStateGraph
        A compiled BDI-augmented teleological graph.
    """
    from langgraph.graph import END, START, StateGraph

    from synthetic_teleology.graph.edges import should_continue, should_revise
    from synthetic_teleology.graph.nodes import (
        act_node,
        check_constraints_node,
        evaluate_node,
        filter_policy_node,
        make_check_constraints_node,
        make_evaluate_node,
        make_filter_policy_node,
        reflect_node,
    )
    from synthetic_teleology.graph.state import TeleologicalState

    eval_fn = make_evaluate_node(evaluator) if evaluator is not None else evaluate_node
    constraints_fn = (
        make_check_constraints_node(constraint_pipeline)
        if constraint_pipeline is not None
        else check_constraints_node
    )
    filter_fn = (
        make_filter_policy_node(policy_filter)
        if policy_filter is not None
        else filter_policy_node
    )

    graph = StateGraph(TeleologicalState)

    # Use BDI-augmented nodes for perceive, revise, plan
    graph.add_node("perceive", make_bdi_perceive_node(bdi_agent))
    graph.add_node("evaluate", eval_fn)
    graph.add_node("revise", make_bdi_revise_node(bdi_agent))
    graph.add_node("check_constraints", constraints_fn)
    graph.add_node("plan", make_bdi_plan_node(bdi_agent))
    graph.add_node("filter_policy", filter_fn)
    graph.add_node("act", act_node)
    graph.add_node("reflect", reflect_node)

    # Edges — same structure as standard teleological graph
    graph.add_edge(START, "perceive")
    graph.add_edge("perceive", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        should_revise,
        {"revise": "revise", "check_constraints": "check_constraints"},
    )
    graph.add_edge("revise", "check_constraints")
    graph.add_edge("check_constraints", "plan")
    graph.add_edge("plan", "filter_policy")
    graph.add_edge("filter_policy", "act")
    graph.add_edge("act", "reflect")
    graph.add_conditional_edges(
        "reflect",
        should_continue,
        {"perceive": "perceive", "__end__": END},
    )

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before
    if interrupt_after:
        compile_kwargs["interrupt_after"] = interrupt_after

    return graph.compile(**compile_kwargs)
