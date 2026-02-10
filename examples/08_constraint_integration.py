#!/usr/bin/env python3
"""Example 08: Constraint pipeline with SafetyChecker and BudgetChecker.

Demonstrates:
- Building a ConstraintPipeline with SafetyChecker + BudgetChecker
- Running constraint checks on states and actions
- Using PolicyFilter to remove non-compliant actions from a policy
- Fail-fast mode vs collect-all mode
- Budget tracking across multiple actions

Run:
    PYTHONPATH=src python examples/08_constraint_integration.py
"""

from __future__ import annotations

import time

from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import (
    ActionSpec,
    ObjectiveVector,
    PolicySpec,
    StateSnapshot,
)
from synthetic_teleology.services.constraint_engine import (
    BudgetChecker,
    ConstraintPipeline,
    PolicyFilter,
    SafetyChecker,
)


def main() -> None:
    print("=== Constraint Integration ===\n")

    # -- Setup: goal and state ------------------------------------------------
    objective = ObjectiveVector(
        values=(5.0, 5.0),
        directions=(Direction.APPROACH, Direction.APPROACH),
    )
    goal = Goal(name="constrained-goal", objective=objective)

    # State within safe bounds
    safe_state = StateSnapshot(
        timestamp=time.time(),
        values=(3.0, 4.0),
    )

    # State violating upper bound on dim[0]
    unsafe_state = StateSnapshot(
        timestamp=time.time(),
        values=(12.0, 4.0),
    )

    # -- Safety Checker -------------------------------------------------------
    print("--- SafetyChecker ---\n")

    safety = SafetyChecker(
        lower_bounds=(0.0, 0.0),
        upper_bounds=(10.0, 10.0),
        name="EnvSafety",
    )

    passed, msg = safety.check(goal, safe_state)
    print(f"Safe state   (3.0, 4.0): passed={passed}")

    passed, msg = safety.check(goal, unsafe_state)
    print(f"Unsafe state (12.0, 4.0): passed={passed}")
    if msg:
        print(f"  Violation: {msg}")
    print()

    # Check action that would push state out of bounds
    dangerous_action = ActionSpec(
        name="big_push",
        parameters={"effect": (8.0, 0.0)},
        cost=1.0,
    )
    passed, msg = safety.check(goal, safe_state, action=dangerous_action)
    print(f"Safe state + big_push(+8.0, 0.0): passed={passed}")
    if msg:
        print(f"  Violation: {msg}")

    safe_action = ActionSpec(
        name="small_push",
        parameters={"effect": (1.0, 0.0)},
        cost=0.5,
    )
    passed, msg = safety.check(goal, safe_state, action=safe_action)
    print(f"Safe state + small_push(+1.0, 0.0): passed={passed}")
    print()

    # -- Budget Checker -------------------------------------------------------
    print("--- BudgetChecker ---\n")

    budget = BudgetChecker(total_budget=2.0, name="ActionBudget")
    print(f"Total budget: 2.0")
    print(f"Remaining:    {budget.budget_remaining:.2f}")
    print()

    # Check actions against budget
    cheap_action = ActionSpec(name="cheap", parameters={}, cost=0.3)
    expensive_action = ActionSpec(name="expensive", parameters={}, cost=3.0)

    passed, msg = budget.check(goal, safe_state, action=cheap_action)
    print(f"Cheap action (cost=0.3): passed={passed}")

    passed, msg = budget.check(goal, safe_state, action=expensive_action)
    print(f"Expensive action (cost=3.0): passed={passed}")
    if msg:
        print(f"  Violation: {msg}")

    # Simulate spending
    budget.record_cost(0.3)
    budget.record_cost(0.5)
    budget.record_cost(0.8)
    print(f"\nAfter spending 0.3 + 0.5 + 0.8 = 1.6:")
    print(f"  Total spent: {budget.total_spent:.2f}")
    print(f"  Remaining:   {budget.budget_remaining:.2f}")

    medium_action = ActionSpec(name="medium", parameters={}, cost=0.5)
    passed, msg = budget.check(goal, safe_state, action=medium_action)
    print(f"  Medium action (cost=0.5): passed={passed}")
    if msg:
        print(f"    Violation: {msg}")
    print()

    # -- Constraint Pipeline --------------------------------------------------
    print("--- ConstraintPipeline (collect-all mode) ---\n")

    # Fresh budget checker for pipeline demo
    budget_fresh = BudgetChecker(total_budget=1.0, name="PipelineBudget")

    pipeline = ConstraintPipeline(
        checkers=[safety, budget_fresh],
        fail_fast=False,
    )
    print(f"Pipeline checkers: {len(pipeline.checkers)}")

    # Check: safe state, no action
    passed, violations = pipeline.check_all(goal, safe_state)
    print(f"Safe state, no action: all_passed={passed}, violations={len(violations)}")

    # Check: unsafe state, no action
    passed, violations = pipeline.check_all(goal, unsafe_state)
    print(f"Unsafe state, no action: all_passed={passed}, violations={len(violations)}")
    for v in violations:
        print(f"  - {v}")

    # Check: safe state, over-budget action
    over_budget_action = ActionSpec(name="costly", parameters={"effect": (0.5, 0.5)}, cost=2.0)
    passed, violations = pipeline.check_all(goal, safe_state, action=over_budget_action)
    print(f"Safe state + costly action: all_passed={passed}, violations={len(violations)}")
    for v in violations:
        print(f"  - {v}")
    print()

    # -- Fail-fast Pipeline ---------------------------------------------------
    print("--- ConstraintPipeline (fail-fast mode) ---\n")

    pipeline_ff = ConstraintPipeline(
        checkers=[safety, budget_fresh],
        fail_fast=True,
        hard_checker_indices=[0],  # Only safety is hard
    )

    passed, violations = pipeline_ff.check_all(goal, unsafe_state, action=over_budget_action)
    print(f"Unsafe state + costly action (fail-fast): all_passed={passed}")
    print(f"  Violations collected: {len(violations)} (stops at first hard failure)")
    for v in violations:
        print(f"  - {v}")
    print()

    # -- PolicyFilter ---------------------------------------------------------
    print("--- PolicyFilter ---\n")

    # Build a policy with mixed compliant and non-compliant actions
    policy = PolicySpec(
        actions=(
            ActionSpec(name="safe_small", parameters={"effect": (1.0, 1.0)}, cost=0.2),
            ActionSpec(name="safe_medium", parameters={"effect": (2.0, 2.0)}, cost=0.3),
            ActionSpec(name="unsafe_big", parameters={"effect": (8.0, 8.0)}, cost=0.1),
            ActionSpec(name="over_budget", parameters={"effect": (0.5, 0.5)}, cost=5.0),
        ),
        probabilities=(0.25, 0.25, 0.25, 0.25),
    )
    print(f"Original policy: {policy.size} actions (stochastic={policy.is_stochastic})")
    for a in policy.actions:
        print(f"  {a.name}: effect={a.parameters.get('effect')}, cost={a.cost}")
    print()

    # Fresh pipeline for filter demo
    safety_filter = SafetyChecker(
        lower_bounds=(0.0, 0.0),
        upper_bounds=(10.0, 10.0),
        name="FilterSafety",
    )
    budget_filter = BudgetChecker(total_budget=1.0, name="FilterBudget")
    filter_pipeline = ConstraintPipeline(checkers=[safety_filter, budget_filter])
    policy_filter = PolicyFilter(filter_pipeline)

    filtered = policy_filter.filter(policy, goal, safe_state)
    print(f"Filtered policy: {filtered.size} actions")
    for i, a in enumerate(filtered.actions):
        prob = filtered.probabilities[i] if filtered.probabilities else "N/A"
        print(f"  {a.name}: cost={a.cost}, probability={prob}")
    print(f"  Metadata: filtered_count={filtered.metadata.get('filtered_count', 0)}")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
