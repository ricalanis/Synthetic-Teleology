#!/usr/bin/env python3
"""Example 05: Hierarchical goals with GoalTree + nested subgraphs.

Demonstrates:
- Using GoalTree for hierarchical goal management
- Running separate teleological subgraphs for each active leaf goal
- Revision propagation through the tree

Run:
    PYTHONPATH=src python examples/05_hierarchical_goals.py
"""

from __future__ import annotations

from synthetic_teleology.domain.aggregates import GoalTree
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import ObjectiveVector
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph import GraphBuilder


def main() -> None:
    # -- Build goal tree --
    root_goal = Goal(
        name="master-plan",
        objective=ObjectiveVector(
            values=(10.0, 10.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        ),
    )
    sub_goal_a = Goal(
        name="sub-a-reach-x",
        objective=ObjectiveVector(
            values=(10.0, 5.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        ),
    )
    sub_goal_b = Goal(
        name="sub-b-reach-y",
        objective=ObjectiveVector(
            values=(5.0, 10.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        ),
    )

    tree = GoalTree(root=root_goal)
    tree.add_subgoal(root_goal.goal_id, sub_goal_a)
    tree.add_subgoal(root_goal.goal_id, sub_goal_b)

    print("=== Hierarchical Goals ===")
    print(f"Root: {root_goal.name} -> {root_goal.objective.values}")
    print(f"  Sub-A: {sub_goal_a.name} -> {sub_goal_a.objective.values}")
    print(f"  Sub-B: {sub_goal_b.name} -> {sub_goal_b.objective.values}")
    print(f"Tree size: {tree.size}")
    print(f"Active leaves: {[g.name for g in tree.get_active_leaves()]}")
    print()

    # -- Run a subgraph for each active leaf --
    for leaf in tree.get_active_leaves():
        env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))

        app, state = (
            GraphBuilder(f"agent-{leaf.name}")
            .with_goal(leaf)
            .with_max_steps(15)
            .with_goal_achieved_threshold(0.85)
            .with_environment(
                perceive_fn=lambda env=env: env.observe(),
                act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
                transition_fn=lambda a, env=env: env.step(a) if a else None,
            )
            .build()
        )

        result = app.invoke(state)

        print(f"Leaf '{leaf.name}':")
        print(f"  Steps: {result['step']}")
        print(f"  Stop reason: {result.get('stop_reason', 'none')}")
        print(f"  Final eval: {result['eval_signal'].score:.4f}")
        print(f"  Final state: {result['state_snapshot'].values}")
        print()

    # -- Check tree coherence --
    issues = tree.validate_coherence()
    coherent = len(issues) == 0
    print(f"Tree coherent: {coherent}")
    if issues:
        for issue in issues:
            print(f"  Issue: {issue}")

    # -- Lineage --
    lineage = tree.get_lineage(sub_goal_a.goal_id)
    print(f"Lineage for sub-a: {[g.name for g in lineage]}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
