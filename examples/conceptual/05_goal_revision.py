#!/usr/bin/env python3
"""Example 05: Goal revision strategies — 5 updater types + GoalUpdaterChain.

Demonstrates:
- ThresholdUpdater: revise only when |score| exceeds threshold
- GradientUpdater: continuous gradient-based goal adjustment
- UncertaintyAwareUpdater: revise when confidence is LOW (active inference)
- GoalUpdaterChain: cascade multiple updaters (first match wins)
- Comparing revision frequency and final goal values

Run:
    PYTHONPATH=src python examples/conceptual/05_goal_revision.py
"""

from __future__ import annotations

from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph import GraphBuilder, collect_stream_events
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.goal_revision import (
    GoalUpdaterChain,
    GradientUpdater,
    ThresholdUpdater,
    UncertaintyAwareUpdater,
)


def _run_with_updater(label: str, updater) -> dict:
    """Run a teleological agent with the given goal updater."""
    env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
    app, initial_state = (
        GraphBuilder(f"revision-{label}")
        .with_objective((8.0, 8.0))
        .with_evaluator(NumericEvaluator(max_distance=15.0))
        .with_goal_updater(updater)
        .with_max_steps(20)
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

    # Count revisions and track goal changes
    revisions = 0
    goal_values = []
    scores = []
    for ev in events:
        if ev.get("node") == "revise":
            goal_history = ev.get("goal_history", [])
            if goal_history:
                revisions += 1
                latest = goal_history[-1]
                if hasattr(latest, "objective") and latest.objective:
                    goal_values.append(latest.objective.values)
        if ev.get("node") == "evaluate" and ev.get("eval_signal"):
            scores.append(ev["eval_signal"].score)

    # Final goal
    final_goal_values = None
    for ev in reversed(events):
        if ev.get("goal") and hasattr(ev["goal"], "objective") and ev["goal"].objective:
            final_goal_values = ev["goal"].objective.values
            break

    final_step = 0
    for ev in events:
        if ev.get("node") == "reflect":
            final_step = ev.get("step", final_step)

    return {
        "label": label,
        "revisions": revisions,
        "final_goal": final_goal_values,
        "scores": scores,
        "final_step": final_step,
        "env": env,
    }


def main() -> None:
    print("=== Goal Revision Strategies Comparison ===")
    print("Initial goal: approach (8.0, 8.0) from (0.0, 0.0)")
    print()

    updaters = [
        ("Threshold(0.3)", ThresholdUpdater(threshold=0.3, learning_rate=0.15)),
        ("Gradient", GradientUpdater(learning_rate=0.08, min_gradient_norm=0.01)),
        ("Uncertainty", UncertaintyAwareUpdater(confidence_threshold=0.5, adaptation_rate=0.2)),
        ("Chain(Threshold+Gradient)", GoalUpdaterChain(updaters=[
            ThresholdUpdater(threshold=0.7, learning_rate=0.1),
            GradientUpdater(learning_rate=0.05),
        ])),
    ]

    results = []
    for label, updater in updaters:
        result = _run_with_updater(label, updater)
        results.append(result)

    # -- Print comparison --
    print(f"{'Updater':<28} {'Revisions':>9} {'Steps':>5}  Final Goal")
    print("-" * 72)
    for r in results:
        goal_str = (
            f"({r['final_goal'][0]:.2f}, {r['final_goal'][1]:.2f})"
            if r["final_goal"] else "unchanged"
        )
        print(f"{r['label']:<28} {r['revisions']:>9} {r['final_step']:>5}  {goal_str}")

    print()

    # -- Detail per strategy --
    for r in results:
        print(f"--- {r['label']} ---")
        if r["scores"]:
            trajectory = " -> ".join(f"{s:.3f}" for s in r["scores"][:6])
            if len(r["scores"]) > 6:
                trajectory += " ..."
            print(f"  Score trajectory: {trajectory}")
        print(f"  Final state: {r['env'].observe().values}")
        print(f"  Revisions: {r['revisions']}")
        print()

    print("Key insight: ThresholdUpdater revises only on large deviations,")
    print("GradientUpdater adjusts continuously, UncertaintyAwareUpdater")
    print("activates under low confidence, and GoalUpdaterChain tries")
    print("each in order (first non-None result wins).")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
