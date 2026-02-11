#!/usr/bin/env python3
"""Example 04: Human-in-the-loop with custom review node.

Demonstrates:
- Customizing the teleological graph with a human review step
- Using node-level hooks for human approval logic
- Swapping individual nodes in the graph

Note: For full LangGraph interrupt() / Command(resume=...) support,
the state must be fully serializable. This example demonstrates the
pattern using a simulated approval callback.

Run:
    PYTHONPATH=src python examples/04_human_in_the_loop.py
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph import (
    TeleologicalState,
    act_node,
    check_constraints_node,
    evaluate_node,
    filter_policy_node,
    perceive_node,
    plan_node,
    reflect_node,
    revise_node,
    should_continue,
    should_revise,
)
from synthetic_teleology.graph.builder import GraphBuilder

# Simulated human approval
_approval_log: list[dict[str, Any]] = []


def human_review_revise_node(state: dict) -> dict:
    """Wraps revise_node with simulated human approval."""
    signal = state.get("eval_signal")
    goal = state.get("goal")
    step = state.get("step", 0)

    if signal and abs(signal.score) >= 0.5:
        # Simulate human review
        review = {
            "step": step,
            "goal": goal.name if goal else "unknown",
            "eval_score": signal.score,
            "decision": "approve" if signal.score < 0.8 else "skip",
        }
        _approval_log.append(review)

        if review["decision"] == "skip":
            print(
                f"  [HUMAN] Step {step}: Skipping revision "
                f"(score {signal.score:.3f} good enough)"
            )
            return {}
        else:
            print(f"  [HUMAN] Step {step}: Approved revision (score {signal.score:.3f})")

    return revise_node(state)


def build_hitl_graph():
    """Build a teleological graph with human-in-the-loop at revision."""
    graph = StateGraph(TeleologicalState)

    graph.add_node("perceive", perceive_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("revise", human_review_revise_node)
    graph.add_node("check_constraints", check_constraints_node)
    graph.add_node("plan", plan_node)
    graph.add_node("filter_policy", filter_policy_node)
    graph.add_node("act", act_node)
    graph.add_node("reflect", reflect_node)

    graph.add_edge(START, "perceive")
    graph.add_edge("perceive", "evaluate")
    graph.add_conditional_edges(
        "evaluate", should_revise,
        {"revise": "revise", "check_constraints": "check_constraints"},
    )
    graph.add_edge("revise", "check_constraints")
    graph.add_edge("check_constraints", "plan")
    graph.add_edge("plan", "filter_policy")
    graph.add_edge("filter_policy", "act")
    graph.add_edge("act", "reflect")
    graph.add_conditional_edges(
        "reflect", should_continue,
        {"perceive": "perceive", "__end__": END},
    )

    return graph.compile()


def main() -> None:
    env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))

    # Build initial state using the builder
    _, initial_state = (
        GraphBuilder("hitl-agent")
        .with_objective((5.0, 5.0))
        .with_max_steps(10)
        .with_environment(
            perceive_fn=lambda: env.observe(),
            act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
            transition_fn=lambda a: env.step(a) if a else None,
        )
        .build()
    )

    # Build custom HITL graph (without checkpointer for simplicity)
    app = build_hitl_graph()

    print("=== Human-in-the-Loop Teleological Agent ===")
    print("Human reviews goal revisions when |eval_score| >= 0.5")
    print()

    result = app.invoke(initial_state)

    print()
    print(f"Steps completed: {result['step']}")
    print(f"Stop reason: {result.get('stop_reason', 'none')}")
    print(f"Final eval score: {result['eval_signal'].score:.4f}")
    print(f"Human reviews: {len(_approval_log)}")

    for review in _approval_log:
        print(f"  Step {review['step']}: {review['decision']} "
              f"(score={review['eval_score']:.3f})")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
