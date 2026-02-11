#!/usr/bin/env python3
"""Example 01: Basic teleological loop as a LangGraph StateGraph.

Demonstrates:
- Building the teleological graph with ``build_teleological_graph()``
- Invoking it with ``.invoke()``
- Inspecting the resulting state

Run:
    PYTHONPATH=src python examples/conceptual/01_basic_loop.py
"""

from __future__ import annotations

from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph import GraphBuilder


def main() -> None:
    # -- Environment --
    env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))

    # -- Build graph with the fluent builder --
    app, initial_state = (
        GraphBuilder("agent-basic")
        .with_objective((5.0, 5.0))
        .with_max_steps(20)
        .with_goal_achieved_threshold(0.9)
        .with_environment(
            perceive_fn=lambda: env.observe(),
            act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
            transition_fn=lambda a: env.step(a) if a else None,
        )
        .build()
    )

    print("=== LangGraph Basic Teleological Loop ===")
    print(f"Goal: approach {initial_state['goal'].objective.values}")
    print(f"Start state: {env.observe().values}")
    print()

    # -- Run --
    result = app.invoke(initial_state)

    print(f"Stop reason: {result.get('stop_reason', 'none')}")
    print(f"Steps completed: {result['step']}")
    print(f"Final state: {result['state_snapshot'].values}")
    print(f"Final eval score: {result['eval_signal'].score:.4f}")
    print(f"Events emitted: {len(result['events'])}")
    print(f"Actions taken: {len(result['action_history'])}")
    print()

    # Inspect goal history
    if result["goal_history"]:
        print(f"Goal revisions: {len(result['goal_history'])}")
    else:
        print("No goal revisions occurred.")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
