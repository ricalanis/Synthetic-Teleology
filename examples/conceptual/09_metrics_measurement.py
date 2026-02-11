#!/usr/bin/env python3
"""Example 09: Metrics & Measurement — all 7 teleological metrics + MetricsEngine.

Demonstrates:
- Building AgentLog entries from a teleological graph run
- Computing all 7 metrics via MetricsEngine:
  GP (Goal Persistence), TC (Teleological Coherence), RE (Reflective Efficiency),
  AD (Adaptivity), NF (Normative Fidelity), IY (Innovation Yield), LS (Lyapunov Stability)
- Building a MetricsReport with summary()
- Comparing metrics across agent configurations

Run:
    PYTHONPATH=src python examples/conceptual/09_metrics_measurement.py
"""

from __future__ import annotations

import time

from synthetic_teleology.domain.events import ConstraintViolated, GoalRevised
from synthetic_teleology.domain.values import ActionSpec
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph import GraphBuilder, collect_stream_events
from synthetic_teleology.measurement.collector import AgentLog, AgentLogEntry
from synthetic_teleology.measurement.engine import MetricsEngine
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.goal_revision import GradientUpdater, ThresholdUpdater
from synthetic_teleology.services.planning import GreedyPlanner, StochasticPlanner


def _make_actions() -> list[ActionSpec]:
    """Build a 2D action space."""
    actions: list[ActionSpec] = []
    for d in range(2):
        for sign, label in [(0.5, "pos"), (-0.5, "neg")]:
            effect = tuple(sign if i == d else 0.0 for i in range(2))
            actions.append(ActionSpec(
                name=f"step_d{d}_{label}",
                parameters={"effect": effect, "delta": effect},
                cost=0.1,
            ))
    actions.append(ActionSpec(
        name="noop",
        parameters={"effect": (0.0, 0.0), "delta": (0.0, 0.0)},
        cost=0.0,
    ))
    return actions


def _run_and_build_log(agent_id: str, planner, updater) -> AgentLog:
    """Run a teleological agent and construct an AgentLog from stream events."""
    env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
    app, initial_state = (
        GraphBuilder(agent_id)
        .with_objective((5.0, 5.0))
        .with_evaluator(NumericEvaluator(max_distance=10.0))
        .with_planner(planner)
        .with_goal_updater(updater)
        .with_max_steps(20)
        .with_goal_achieved_threshold(0.9)
        .with_environment(
            perceive_fn=lambda: env.observe(),
            act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
            transition_fn=lambda a: env.step(a) if a else None,
        )
        .build()
    )

    stream = app.stream(initial_state, stream_mode="updates")
    events = collect_stream_events(stream)

    # Build AgentLog from stream events
    log = AgentLog(agent_id=agent_id)
    step = 0
    current_entry = None

    for ev in events:
        node = ev.get("node", "")

        if node == "perceive":
            step += 1
            current_entry = AgentLogEntry(
                step=step,
                timestamp=time.time(),
            )
            if ev.get("state_snapshot"):
                current_entry.state_values = ev["state_snapshot"].values

        elif node == "evaluate" and current_entry:
            if ev.get("eval_signal"):
                current_entry.eval_score = ev["eval_signal"].score
                current_entry.eval_confidence = ev["eval_signal"].confidence

        elif node == "revise" and current_entry:
            if ev.get("goal_history"):
                current_entry.goal_revised = True
                # Create a GoalRevised event for the log
                log.goal_revisions.append(GoalRevised(
                    source_id=agent_id,
                ))

        elif node == "act" and current_entry:
            if ev.get("executed_action"):
                current_entry.action_name = ev["executed_action"].name
                current_entry.action_cost = ev["executed_action"].cost

        elif node == "check_constraints" and current_entry:
            if ev.get("constraint_violations"):
                current_entry.constraint_violated = True
                for v in ev["constraint_violations"]:
                    log.constraint_violations.append(ConstraintViolated(
                        source_id=agent_id,
                        violation_details=str(v),
                    ))

        elif node == "reflect" and current_entry:
            current_entry.goal_id = agent_id
            log.entries.append(current_entry)
            current_entry = None

    return log


def main() -> None:
    print("=== Teleological Metrics & Measurement ===")
    print()

    actions = _make_actions()

    # Two agent configurations: stable vs adaptive
    configs = [
        ("stable-agent", GreedyPlanner(action_space=actions),
         ThresholdUpdater(threshold=0.99, learning_rate=0.05)),
        ("adaptive-agent", StochasticPlanner(action_space=actions, temperature=1.0, seed=42),
         GradientUpdater(learning_rate=0.1)),
    ]

    engine = MetricsEngine()  # includes all 7 default metrics
    print(f"Metrics engine loaded: {', '.join(engine.metric_names)}")
    print()

    reports = {}
    for agent_id, planner, updater in configs:
        log = _run_and_build_log(agent_id, planner, updater)
        report = engine.build_report(agent_id, log)
        reports[agent_id] = report
        print(f"Built log for '{agent_id}': {log.num_steps} steps, "
              f"{log.revision_count} revisions, "
              f"{len(log.constraint_violations)} violations")

    print()

    # -- Print reports --
    for _agent_id, report in reports.items():
        print(report.summary())
        print()

    # -- Comparison table --
    print("=== Side-by-Side Comparison ===")
    metric_names = engine.metric_names
    header = f"{'Metric':<28} {'stable-agent':>12} {'adaptive-agent':>14}"
    print(header)
    print("-" * len(header))
    for name in metric_names:
        stable_val = reports["stable-agent"].get_metric(name)
        adaptive_val = reports["adaptive-agent"].get_metric(name)
        s_str = f"{stable_val.value:.4f}" if stable_val else "N/A"
        a_str = f"{adaptive_val.value:.4f}" if adaptive_val else "N/A"
        print(f"{name:<28} {s_str:>12} {a_str:>14}")
    print()

    # -- Metric interpretations --
    print("Metric Interpretations (Haidemariam 2026):")
    interpretations = {
        "goal_persistence": "GP: Higher = fewer goal revisions (stable strategy)",
        "teleological_coherence": "TC: Score improvement correlates with goal stability",
        "reflective_efficiency": "RE: Cost-weighted score improvement per step",
        "adaptivity": "AD: Recovery rate after perturbations (higher = more adaptive)",
        "normative_fidelity": "NF: Fraction of steps without constraint violations",
        "innovation_yield": "IY: Score improvement from novel (unique) actions",
        "lyapunov_stability": "LS: Score variance convergence (higher = more stable)",
    }
    for _name, description in interpretations.items():
        print(f"  {description}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
