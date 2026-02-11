"""Build the teleological StateGraph.

``build_teleological_graph()`` wires the 8 nodes and conditional edges
into a compiled LangGraph that implements the full Perceive-Evaluate-
Revise-Plan-Filter-Act-Reflect loop.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from synthetic_teleology.graph.edges import should_continue, should_revise
from synthetic_teleology.graph.nodes import (
    act_node,
    check_constraints_node,
    evaluate_node,
    filter_policy_node,
    ground_goal_node,
    make_check_constraints_node,
    make_evaluate_node,
    make_filter_policy_node,
    make_plan_node,
    make_revise_node,
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
    evaluator: Any | None = None,
    goal_updater: Any | None = None,
    planner: Any | None = None,
    constraint_pipeline: Any | None = None,
    policy_filter: Any | None = None,
    enable_evolving_constraints: bool = False,
    state_schema: type | None = None,
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
    evaluator:
        Optional evaluator instance. When provided, the evaluate node uses a
        closure instead of reading from state (enables checkpointing).
    goal_updater:
        Optional goal updater instance (closure injection).
    planner:
        Optional planner instance (closure injection).
    constraint_pipeline:
        Optional constraint pipeline instance (closure injection).
    policy_filter:
        Optional policy filter instance (closure injection).
    enable_evolving_constraints:
        If True, wire an ``evolve_constraints`` node after check_constraints.
    state_schema:
        Optional state TypedDict to use instead of the default
        ``TeleologicalState``.  Pass a bounded variant (from
        ``make_bounded_state()``) to cap accumulation channels.

    Returns
    -------
    CompiledStateGraph
        A compiled graph ready for ``.invoke()`` or ``.stream()``.
    """
    schema = state_schema if state_schema is not None else TeleologicalState
    graph = StateGraph(schema)

    # Use closure-based nodes when strategies are provided as kwargs,
    # otherwise fall back to module-level functions that read from state.
    eval_fn = make_evaluate_node(evaluator) if evaluator is not None else evaluate_node
    revise_fn = make_revise_node(goal_updater) if goal_updater is not None else revise_node
    plan_fn = make_plan_node(planner) if planner is not None else plan_node
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

    # Add all core nodes
    graph.add_node("perceive", perceive_node)
    graph.add_node("evaluate", eval_fn)
    graph.add_node("revise", revise_fn)
    graph.add_node("check_constraints", constraints_fn)
    graph.add_node("plan", plan_fn)
    graph.add_node("filter_policy", filter_fn)
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

    if enable_evolving_constraints:
        from synthetic_teleology.graph.nodes import evolve_constraints_node

        graph.add_node("evolve_constraints", evolve_constraints_node)
        graph.add_edge("check_constraints", "evolve_constraints")
        graph.add_edge("evolve_constraints", "plan")
    else:
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
