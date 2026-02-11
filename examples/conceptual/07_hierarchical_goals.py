#!/usr/bin/env python3
"""Example 07: Hierarchical goal structure — GoalTree, coherence, propagation.

Demonstrates:
- GoalTree construction with root + sub-goals
- validate_coherence(): checking parent-child structural consistency
- propagate_revision(): parent revision cascades to children
- get_lineage(): tracing goal ancestry from leaf to root
- HierarchicalUpdater: revision with parent regularization
- Running a full agent with hierarchical goal structure

Run:
    PYTHONPATH=src python examples/conceptual/07_hierarchical_goals.py
"""

from __future__ import annotations

from synthetic_teleology.domain.aggregates import GoalTree
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import ActionSpec, ObjectiveVector
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph import GraphBuilder, collect_stream_events
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.goal_revision import GradientUpdater, HierarchicalUpdater


def main() -> None:
    print("=== Hierarchical Goals ===")
    print()

    # -- 1. Build a GoalTree: root (4D) -> two sub-goals (2D each) --
    root_goal = Goal(
        name="master-goal",
        description="Reach (8, 8, 4, 4) in 4D space",
        objective=ObjectiveVector(
            values=(8.0, 8.0, 4.0, 4.0),
            directions=(Direction.APPROACH, Direction.APPROACH,
                        Direction.APPROACH, Direction.APPROACH),
        ),
    )

    sub_goal_a = Goal(
        name="spatial-sub",
        description="Handle dimensions 0-1",
        objective=ObjectiveVector(
            values=(8.0, 8.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        ),
    )

    sub_goal_b = Goal(
        name="resource-sub",
        description="Handle dimensions 2-3",
        objective=ObjectiveVector(
            values=(4.0, 4.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        ),
    )

    tree = GoalTree(root=root_goal)
    tree.add_subgoal(root_goal.goal_id, sub_goal_a)
    tree.add_subgoal(root_goal.goal_id, sub_goal_b)

    print(f"GoalTree: {tree.size} goals")
    print(f"  Root: {root_goal.name} -> {root_goal.objective.values}")
    for child in tree.get_children(root_goal.goal_id):
        print(f"  Sub: {child.name} -> {child.objective.values}")
    print()

    # -- 2. Validate coherence --
    issues = tree.validate_coherence()
    print(f"Coherence check: {'PASS (no issues)' if not issues else 'ISSUES FOUND'}")
    for issue in issues:
        print(f"  - {issue}")
    print()

    # -- 3. Get lineage --
    lineage = tree.get_lineage(sub_goal_a.goal_id)
    print(f"Lineage for '{sub_goal_a.name}':")
    for i, g in enumerate(lineage):
        indent = "  " * i
        print(f"  {indent}-> {g.name} (id={g.goal_id[:6]}...)")
    print()

    # -- 4. Propagate revision --
    print("--- Propagating revision from root ---")
    new_root_objective = ObjectiveVector(
        values=(10.0, 10.0, 6.0, 6.0),
        directions=root_goal.objective.directions,
    )
    revised_root, revision_record = root_goal.revise(
        new_objective=new_root_objective,
        reason="Environment changed, adjusting targets",
    )
    # Update tree with revised root (note: propagate_revision works on children)
    revisions = tree.propagate_revision(revised_root)
    print(f"Root revised: {root_goal.objective.values} -> {revised_root.objective.values}")
    print(f"Propagated revisions: {len(revisions)}")
    for rev in revisions:
        print(f"  {rev.previous_goal_id[:6]}... -> {rev.new_goal_id[:6]}... ({rev.reason})")
    print()

    # -- 5. Run agent with HierarchicalUpdater on sub-goal A --
    print("--- Running agent with HierarchicalUpdater ---")
    env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))

    # Rebuild tree for the agent run (fresh goals)
    run_root = Goal(
        name="run-root",
        objective=ObjectiveVector(
            values=(8.0, 8.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        ),
    )
    run_child = Goal(
        name="run-child",
        objective=ObjectiveVector(
            values=(8.0, 8.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        ),
        parent_id=run_root.goal_id,
    )
    run_tree = GoalTree(root=run_root)
    run_tree.add_subgoal(run_root.goal_id, run_child)

    hierarchical_updater = HierarchicalUpdater(
        goal_tree=run_tree,
        inner=GradientUpdater(learning_rate=0.1),
        regularization_strength=0.3,
    )

    actions: list[ActionSpec] = []
    for d in range(2):
        for sign, label in [(1.0, "pos"), (-1.0, "neg")]:
            effect = tuple(sign if i == d else 0.0 for i in range(2))
            actions.append(ActionSpec(
                name=f"move_d{d}_{label}",
                parameters={"effect": effect, "delta": effect},
            ))
    actions.append(ActionSpec(name="noop", parameters={"effect": (0.0, 0.0), "delta": (0.0, 0.0)}))

    from synthetic_teleology.services.planning import GreedyPlanner

    app, initial_state = (
        GraphBuilder("hierarchical-agent")
        .with_goal(run_child)
        .with_evaluator(NumericEvaluator(max_distance=12.0))
        .with_goal_updater(hierarchical_updater)
        .with_planner(GreedyPlanner(action_space=actions))
        .with_max_steps(15)
        .with_goal_achieved_threshold(0.85)
        .with_environment(
            perceive_fn=lambda: env.observe(),
            act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
            transition_fn=lambda a: env.step(a) if a else None,
        )
        .build()
    )

    stream = app.stream(initial_state, stream_mode="updates")
    events = collect_stream_events(stream)

    # Extract results
    revisions_count = sum(
        1 for ev in events
        if ev.get("node") == "revise" and ev.get("goal_history")
    )
    scores = [
        ev["eval_signal"].score
        for ev in events
        if ev.get("node") == "evaluate" and ev.get("eval_signal")
    ]

    print(f"Agent steps: {sum(1 for ev in events if ev.get('node') == 'reflect')}")
    print(f"Goal revisions (with parent regularization): {revisions_count}")
    print(f"Final state: {env.observe().values}")
    if scores:
        print(f"Score trajectory: {' -> '.join(f'{s:.3f}' for s in scores[:8])}"
              + (" ..." if len(scores) > 8 else ""))
    print()
    print("Key insight: HierarchicalUpdater regularizes child goal revisions")
    print("toward the parent's objective, preventing sub-goals from drifting")
    print("too far from the overall strategy.")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
