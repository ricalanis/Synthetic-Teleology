#!/usr/bin/env python3
"""Example 10: Ethical constraints — EthicalChecker, ConstraintPipeline, PolicyFilter.

Demonstrates:
- EthicalChecker with custom rule predicates
- ConstraintPipeline with fail_fast=True and hard_checker_indices
- PolicyFilter: filtering unsafe actions before execution
- Combining EthicalChecker + SafetyChecker + BudgetChecker in a single pipeline
- Streaming constraint violations

Run:
    PYTHONPATH=src python examples/conceptual/10_ethical_constraints.py
"""

from __future__ import annotations

from synthetic_teleology.domain.values import ActionSpec
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph import GraphBuilder, collect_stream_events
from synthetic_teleology.services.constraint_engine import (
    BudgetChecker,
    ConstraintPipeline,
    EthicalChecker,
    PolicyFilter,
    SafetyChecker,
)
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.planning import GreedyPlanner


def main() -> None:
    print("=== Ethical Constraints & Policy Filtering ===")
    print()

    env = NumericEnvironment(dimensions=2, initial_state=(5.0, 5.0))

    # -- Build action space with some "ethically questionable" actions --
    actions: list[ActionSpec] = []
    for d in range(2):
        # Normal positive step
        effect_pos = tuple(1.0 if i == d else 0.0 for i in range(2))
        actions.append(ActionSpec(
            name=f"step_d{d}_pos",
            parameters={"effect": effect_pos, "delta": effect_pos},
            cost=0.5,
        ))
        # Normal negative step
        effect_neg = tuple(-1.0 if i == d else 0.0 for i in range(2))
        actions.append(ActionSpec(
            name=f"step_d{d}_neg",
            parameters={"effect": effect_neg, "delta": effect_neg},
            cost=0.5,
        ))
    # Large aggressive action (high cost, potentially unsafe)
    actions.append(ActionSpec(
        name="aggressive_jump",
        parameters={"effect": (5.0, 5.0), "delta": (5.0, 5.0)},
        cost=3.0,
    ))
    actions.append(ActionSpec(
        name="noop",
        parameters={"effect": (0.0, 0.0), "delta": (0.0, 0.0)},
        cost=0.0,
    ))

    # -- 1. Define ethical rules as predicates --
    def no_negative_territory(goal, state, action):
        """Ethical rule: don't take actions that move into negative territory."""
        if action is None:
            return True
        effect = action.parameters.get("effect", (0.0, 0.0))
        return all(
            s + e >= 0
            for s, e in zip(state.values, effect, strict=False)
        )

    def no_excessive_cost(goal, state, action):
        """Ethical rule: individual actions should not cost more than 2.0."""
        if action is None:
            return True
        return action.cost <= 2.0

    def progress_toward_goal(goal, state, action):
        """Ethical rule: prefer actions that move toward the goal, not away."""
        if action is None or goal.objective is None:
            return True
        effect = action.parameters.get("effect", (0.0, 0.0))
        # At least one dimension should be moving toward the goal
        improving = False
        for s, e, g in zip(
            state.values, effect, goal.objective.values, strict=False,
        ):
            if abs(s + e - g) < abs(s - g):
                improving = True
        return improving

    ethical_checker = EthicalChecker(
        rules=[
            ("no_negative_territory", no_negative_territory),
            ("no_excessive_cost", no_excessive_cost),
            ("progress_toward_goal", progress_toward_goal),
        ],
        name="EthicalRules",
    )

    # -- 2. Safety and budget checkers --
    safety_checker = SafetyChecker(
        lower_bounds=(0.0, 0.0),
        upper_bounds=(20.0, 20.0),
    )
    budget_checker = BudgetChecker(total_budget=6.0)

    # -- 3. Build ConstraintPipeline with fail_fast --
    # Ethical (index 0) and Safety (index 1) are hard constraints
    pipeline = ConstraintPipeline(
        checkers=[ethical_checker, safety_checker, budget_checker],
        fail_fast=True,
        hard_checker_indices=[0, 1],
    )

    print("Constraint pipeline:")
    print("  [0] EthicalChecker (HARD): 3 rules")
    print("  [1] SafetyChecker (HARD): bounds [0,0] to [20,20]")
    print("  [2] BudgetChecker: total_budget=6.0")
    print("  fail_fast=True (stops on first hard violation)")
    print()

    # -- 4. Demonstrate PolicyFilter standalone --
    print("--- PolicyFilter Demo ---")
    from synthetic_teleology.domain.entities import Goal
    from synthetic_teleology.domain.enums import Direction
    from synthetic_teleology.domain.values import ObjectiveVector, PolicySpec

    test_goal = Goal(
        name="test",
        objective=ObjectiveVector(
            values=(12.0, 12.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        ),
    )
    test_state = env.observe()
    policy_filter = PolicyFilter(pipeline)

    # Create a policy with all actions
    full_policy = PolicySpec(actions=tuple(actions))
    filtered_policy = policy_filter.filter(full_policy, test_goal, test_state)

    print(f"Original policy: {full_policy.size} actions")
    print(f"Filtered policy: {filtered_policy.size} actions")
    print("Kept actions:")
    for a in filtered_policy.actions:
        print(f"  {a.name} (cost={a.cost})")
    orig_names = set(a.name for a in full_policy.actions)
    kept_names = set(a.name for a in filtered_policy.actions)
    removed = orig_names - kept_names
    if removed:
        print(f"Filtered out: {', '.join(sorted(removed))}")
    print()

    # -- 5. Run full agent with streaming constraint checks --
    print("--- Full Agent Run with Ethical Constraints ---")

    # Reset budget checker for the actual run
    budget_checker.reset()

    app, initial_state = (
        GraphBuilder("ethical-agent")
        .with_objective((12.0, 12.0))
        .with_evaluator(NumericEvaluator(max_distance=20.0))
        .with_planner(GreedyPlanner(action_space=actions))
        .with_constraint_checkers(ethical_checker, safety_checker, budget_checker)
        .with_max_steps(15)
        .with_goal_achieved_threshold(0.8)
        .with_environment(
            perceive_fn=lambda: env.observe(),
            act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
            transition_fn=lambda a: env.step(a) if a else None,
        )
        .build()
    )

    stream = app.stream(initial_state, stream_mode="updates")
    events = collect_stream_events(stream)

    # Show what happened
    print(f"Start: {(5.0, 5.0)} -> Goal: (12.0, 12.0)")
    print()

    constraint_checks = 0
    violations = []
    actions_taken = []
    for ev in events:
        if ev.get("node") == "check_constraints":
            constraint_checks += 1
            ok = ev.get("constraints_ok", True)
            viol = ev.get("constraint_violations", [])
            if not ok:
                violations.extend(viol)
        if ev.get("node") == "act" and ev.get("executed_action"):
            a = ev["executed_action"]
            actions_taken.append(a.name)

    print(f"Constraint checks: {constraint_checks}")
    print(f"Violations detected: {len(violations)}")
    for v in violations:
        print(f"  - {v}")
    print()

    print(f"Actions taken ({len(actions_taken)}):")
    for a in actions_taken:
        print(f"  {a}")

    final_state = env.observe()
    print(f"\nFinal state: {final_state.values}")
    print(f"Budget remaining: {budget_checker.budget_remaining:.1f}")
    print()

    print("Key insight: The constraint pipeline enforces ethical rules as")
    print("first-class citizens alongside safety and budget constraints.")
    print("PolicyFilter removes non-compliant actions BEFORE execution,")
    print("while ConstraintPipeline with fail_fast=True stops checking")
    print("on the first hard constraint violation.")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
