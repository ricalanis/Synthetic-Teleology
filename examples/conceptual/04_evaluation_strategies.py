#!/usr/bin/env python3
"""Example 04: Evaluation strategies — NumericEvaluator, CompositeEvaluator, ReflectiveEvaluator.

Demonstrates:
- NumericEvaluator: distance-based scoring (Delta(G_t, S_t))
- CompositeEvaluator: weighted blend of multiple evaluators
- ReflectiveEvaluator: drift detection + confidence adjustment via EMA
- Running the same agent with each evaluator and comparing results

Run:
    PYTHONPATH=src python examples/conceptual/04_evaluation_strategies.py
"""

from __future__ import annotations

from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph import GraphBuilder, collect_stream_events
from synthetic_teleology.services.evaluation import (
    CompositeEvaluator,
    NumericEvaluator,
    ReflectiveEvaluator,
)


def _run_agent(label: str, evaluator, env: NumericEnvironment) -> dict:
    """Run a teleological agent with the given evaluator and return the result."""
    env_fresh = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
    app, initial_state = (
        GraphBuilder(f"eval-{label}")
        .with_objective((5.0, 5.0))
        .with_evaluator(evaluator)
        .with_max_steps(15)
        .with_goal_achieved_threshold(0.85)
        .with_environment(
            perceive_fn=lambda: env_fresh.observe(),
            act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
            transition_fn=lambda a: env_fresh.step(a) if a else None,
        )
        .build()
    )
    stream = app.stream(initial_state, stream_mode="updates")
    events = collect_stream_events(stream)

    # Extract score trajectory from evaluate events
    scores = []
    confidences = []
    for ev in events:
        if ev.get("node") == "evaluate" and ev.get("eval_signal"):
            sig = ev["eval_signal"]
            scores.append(sig.score)
            confidences.append(sig.confidence)

    # Get final state from reflect events
    final_step = 0
    stop_reason = "unknown"
    for ev in events:
        if ev.get("node") == "reflect":
            final_step = ev.get("step", final_step)
            stop_reason = ev.get("stop_reason", stop_reason) or "continuing"

    return {
        "label": label,
        "scores": scores,
        "confidences": confidences,
        "final_step": final_step,
        "stop_reason": stop_reason,
        "env": env_fresh,
    }


def main() -> None:
    print("=== Evaluation Strategies Comparison ===")
    print("Goal: approach (5.0, 5.0) from (0.0, 0.0)")
    print()

    # -- 1. NumericEvaluator (baseline) --
    numeric = NumericEvaluator(max_distance=10.0)

    # -- 2. CompositeEvaluator (blend two numerics with different max_distance) --
    composite = CompositeEvaluator(evaluators=[
        (NumericEvaluator(max_distance=10.0), 0.7),
        (NumericEvaluator(max_distance=20.0), 0.3),
    ])

    # -- 3. ReflectiveEvaluator (wraps numeric, detects drift) --
    reflective = ReflectiveEvaluator(
        inner=NumericEvaluator(max_distance=10.0),
        history_size=20,
        drift_threshold=0.15,
        smoothing_factor=0.3,
    )

    env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
    evaluators = [
        ("Numeric", numeric),
        ("Composite", composite),
        ("Reflective", reflective),
    ]

    results = []
    for label, evaluator in evaluators:
        result = _run_agent(label, evaluator, env)
        results.append(result)

    # -- Print comparison --
    for r in results:
        print(f"--- {r['label']} Evaluator ---")
        print(f"  Steps: {r['final_step']}")
        print(f"  Stop reason: {r['stop_reason']}")
        if r["scores"]:
            print(f"  Score trajectory: {' -> '.join(f'{s:.3f}' for s in r['scores'][:8])}"
                  + (" ..." if len(r["scores"]) > 8 else ""))
            print(f"  Final score: {r['scores'][-1]:.4f}")
        if r["confidences"]:
            conf_traj = " -> ".join(f"{c:.3f}" for c in r["confidences"][:8])
            suffix = " ..." if len(r["confidences"]) > 8 else ""
            print(f"  Confidence trajectory: {conf_traj}{suffix}")
            print(f"  Final confidence: {r['confidences'][-1]:.4f}")
        print(f"  Final state: {r['env'].observe().values}")
        print()

    # -- Highlight reflective self-model --
    print("Key insight: ReflectiveEvaluator adjusts confidence when scores")
    print("show high variance (drift detection via EMA). The inner evaluator's")
    print("raw scores are preserved, but confidence is reduced during instability.")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
