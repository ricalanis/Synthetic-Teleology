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
    perceive_node,
    plan_node,
    reflect_node,
    revise_node,
)
from synthetic_teleology.graph.state import TeleologicalState


def build_teleological_graph(
    checkpointer: Any | None = None,
) -> Any:
    """Build and compile the teleological StateGraph.

    The graph implements the invariant loop:

    .. code-block:: text

        START → perceive → evaluate →(conditional)→ revise → check_constraints → plan
                                          |                                       |
                                          +→ check_constraints → plan ←-----------+
                                                                  ↓
                                                           filter_policy → act → reflect
                                                                                   ↓
                                                                     (conditional: perceive | END)

    Parameters
    ----------
    checkpointer:
        Optional LangGraph checkpointer for persistence (e.g. ``MemorySaver``).

    Returns
    -------
    CompiledStateGraph
        A compiled graph ready for ``.invoke()`` or ``.stream()``.
    """
    graph = StateGraph(TeleologicalState)

    # Add all 8 nodes
    graph.add_node("perceive", perceive_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("revise", revise_node)
    graph.add_node("check_constraints", check_constraints_node)
    graph.add_node("plan", plan_node)
    graph.add_node("filter_policy", filter_policy_node)
    graph.add_node("act", act_node)
    graph.add_node("reflect", reflect_node)

    # Edges
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

    return graph.compile(**compile_kwargs)
