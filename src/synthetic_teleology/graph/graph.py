"""Build the teleological StateGraph.

``build_teleological_graph()`` wires the 8 nodes and conditional edges
into a compiled LangGraph that implements the full Perceive-Evaluate-
Revise-Plan-Filter-Act-Reflect loop.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from synthetic_teleology.graph.edges import should_continue, should_revise
from synthetic_teleology.graph.nodes import (
    act_node,
    check_constraints_node,
    evaluate_node,
    filter_policy_node,
    ground_goal_node,
    perceive_node,
    plan_node,
    reflect_node,
    revise_node,
)
from synthetic_teleology.graph.state import TeleologicalState


def build_teleological_graph(
    checkpointer: Any | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    enable_grounding: bool = False,
) -> Any:
    """Build and compile the teleological StateGraph.

    Parameters
    ----------
    checkpointer:
        Optional LangGraph checkpointer for persistence.
    interrupt_before:
        Node names to interrupt before (human-in-the-loop).
    interrupt_after:
        Node names to interrupt after (human-in-the-loop).
    enable_grounding:
        If True, insert a ``ground_goal`` node between evaluate and revise.

    Returns
    -------
    CompiledStateGraph
        A compiled graph ready for ``.invoke()`` or ``.stream()``.
    """
    graph = StateGraph(TeleologicalState)

    # Add all core nodes
    graph.add_node("perceive", perceive_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("revise", revise_node)
    graph.add_node("check_constraints", check_constraints_node)
    graph.add_node("plan", plan_node)
    graph.add_node("filter_policy", filter_policy_node)
    graph.add_node("act", act_node)
    graph.add_node("reflect", reflect_node)

    if enable_grounding:
        graph.add_node("ground_goal", ground_goal_node)

    # Edges
    graph.add_edge(START, "perceive")
    graph.add_edge("perceive", "evaluate")

    if enable_grounding:
        # evaluate -> ground_goal -> (conditional: revise | check_constraints)
        graph.add_edge("evaluate", "ground_goal")
        graph.add_conditional_edges(
            "ground_goal",
            should_revise,
            {"revise": "revise", "check_constraints": "check_constraints"},
        )
    else:
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
