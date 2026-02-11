#!/usr/bin/env python3
"""Example 08: Constraint-aware planning with SafetyChecker and BudgetChecker.

Demonstrates:
- Integrating SafetyChecker and BudgetChecker into the LangGraph
- Streaming constraint checks as the agent runs
- Policy filtering removing unsafe actions

Run:
    PYTHONPATH=src python examples/08_constraint_aware_planning.py
"""

from __future__ import annotations

from synthetic_teleology.domain.values import ActionSpec
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph import GraphBuilder, collect_stream_events
from synthetic_teleology.services.constraint_engine import BudgetChecker, SafetyChecker
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.planning import GreedyPlanner


def main() -> None:
    env = NumericEnvironment(dimensions=2, initial_state=(5.0, 5.0))

    # Build action space
    actions: list[ActionSpec] = []
    for d in range(2):
        for sign, label in [(1.0, "pos"), (-1.0, "neg")]:
            effect = tuple(sign if i == d else 0.0 for i in range(2))
            actions.append(ActionSpec(
                name=f"step_{d}_{label}",
                parameters={"effect": effect, "delta": effect},
                cost=0.5,
            ))
    actions.append(ActionSpec(
        name="noop",
        parameters={"effect": (0.0, 0.0), "delta": (0.0, 0.0)},
        cost=0.0,
    ))

    # Constraints
    safety_checker = SafetyChecker(
        lower_bounds=(0.0, 0.0),
        upper_bounds=(15.0, 15.0),
    )
    budget_checker = BudgetChecker(total_budget=8.0)

    app, initial_state = (
        GraphBuilder("constrained-agent")
        .with_objective((12.0, 12.0))
        .with_evaluator(NumericEvaluator(max_distance=20.0))
        .with_planner(GreedyPlanner(action_space=actions))
        .with_constraint_checkers(safety_checker, budget_checker)
        .with_max_steps(20)
        .with_goal_achieved_threshold(0.85)
        .with_environment(
            perceive_fn=lambda: env.observe(),
            act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
            transition_fn=lambda a: env.step(a) if a else None,
        )
        .build()
    )

    print("=== Constraint-Aware Planning ===")
    print(f"Goal: approach {initial_state['goal'].objective.values}")
    print("Safety bounds: [0,0] to [15,15]")
    print("Budget limit: 8.0 (actions cost 0.5 each)")
    print(f"Start: {env.observe().values}")
    print()

    # Stream execution
    stream = app.stream(initial_state, stream_mode="updates")
    events = collect_stream_events(stream)

    # Show constraint checks
    constraint_events = [e for e in events if e.get("node") == "check_constraints"]
    print(f"Constraint checks: {len(constraint_events)}")
    for ce in constraint_events:
        ok = ce.get("constraints_ok", "?")
        violations = ce.get("constraint_violations", [])
        print(f"  OK={ok}, violations={violations}")

    # Show action events
    action_events = [e for e in events if e.get("node") == "act"]
    print(f"\nActions executed: {len(action_events)}")
    for ae in action_events:
        action = ae.get("executed_action")
        if action:
            print(f"  {action.name} (cost={action.cost})")

    # Show final state
    reflect_events = [e for e in events if e.get("node") == "reflect"]
    if reflect_events:
        last_reflect = reflect_events[-1]
        sr = last_reflect.get("stop_reason")
        print(f"\nStop reason: {sr or 'continuing'}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
