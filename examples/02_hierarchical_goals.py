#!/usr/bin/env python3
"""Example 02: Hierarchical goals with GoalTree.

Demonstrates:
- Building a GoalTree with subgoals
- Querying lineage, flatten, active_leaves
- Propagating revisions down the tree
- Validating coherence

Run:
    PYTHONPATH=src python examples/02_hierarchical_goals.py
"""

from __future__ import annotations

from synthetic_teleology.domain.aggregates import GoalTree
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import ObjectiveVector


def main() -> None:
    print("=== Hierarchical Goals ===\n")

    # -- Build goal tree ------------------------------------------------------
    root_obj = ObjectiveVector(
        values=(10.0, 10.0),
        directions=(Direction.APPROACH, Direction.APPROACH),
    )
    root = Goal(name="master-goal", objective=root_obj)

    sub_a_obj = ObjectiveVector(
        values=(5.0, 10.0),
        directions=(Direction.APPROACH, Direction.APPROACH),
    )
    sub_a = Goal(name="sub-A", objective=sub_a_obj)

    sub_b_obj = ObjectiveVector(
        values=(10.0, 5.0),
        directions=(Direction.APPROACH, Direction.APPROACH),
    )
    sub_b = Goal(name="sub-B", objective=sub_b_obj)

    sub_a1_obj = ObjectiveVector(
        values=(2.0, 10.0),
        directions=(Direction.APPROACH, Direction.APPROACH),
    )
    sub_a1 = Goal(name="sub-A1", objective=sub_a1_obj)

    tree = GoalTree(root)
    tree.add_subgoal(parent_id=root.goal_id, child=sub_a)
    tree.add_subgoal(parent_id=root.goal_id, child=sub_b)
    tree.add_subgoal(parent_id=sub_a.goal_id, child=sub_a1)

    # -- Query ----------------------------------------------------------------
    print(f"Root: {root.name} ({root.goal_id})")
    print(f"Flatten (all goals): {[g.name for g in tree.flatten()]}")
    print(f"Active leaves: {[g.name for g in tree.get_active_leaves()]}")
    print(f"Lineage of sub-A1: {[g.name for g in tree.get_lineage(sub_a1.goal_id)]}")
    print()

    # -- Coherence check ------------------------------------------------------
    coherent, issues = tree.validate_coherence()
    print(f"Coherence check: {'PASS' if coherent else 'FAIL'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    print()

    # -- Revision propagation -------------------------------------------------
    print("Revising root goal...")
    new_root_obj = ObjectiveVector(
        values=(8.0, 8.0),
        directions=(Direction.APPROACH, Direction.APPROACH),
    )
    new_root, revision = root.revise(new_objective=new_root_obj, reason="tighten target")
    print(f"  Old root objective: {root_obj.values}")
    print(f"  New root objective: {new_root_obj.values}")
    print(f"  Revision: {revision.previous_goal_id} -> {revision.new_goal_id}")
    print()

    # Propagate revision (replace root in tree)
    propagated = tree.propagate_revision(root.goal_id, new_root_obj, learning_rate=0.5)
    print(f"Propagated revision to {propagated} subgoals")

    # Print all goals after propagation
    print("\nGoal tree after propagation:")
    for g in tree.flatten():
        if g.objective is not None:
            print(f"  {g.name}: values={g.objective.values}")
        else:
            print(f"  {g.name}: no objective")

    print("\nDone.")


if __name__ == "__main__":
    main()
