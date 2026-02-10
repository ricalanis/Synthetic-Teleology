#!/usr/bin/env python3
"""Example 05: Benchmark suite with MetricsEngine.

Demonstrates:
- Building an AgentLog manually (synthetic data)
- Computing all 7 canonical metrics via MetricsEngine
- Generating a MetricsReport with summary output
- Running a simple benchmark suite

Run:
    PYTHONPATH=src python examples/05_benchmark_suite.py
"""

from __future__ import annotations

import time

from synthetic_teleology.domain.events import GoalRevised
from synthetic_teleology.domain.values import GoalRevision
from synthetic_teleology.measurement.collector import AgentLog, AgentLogEntry
from synthetic_teleology.measurement.engine import MetricsEngine
from synthetic_teleology.measurement.report import MetricsReport


def _build_synthetic_log(
    agent_id: str = "bench-agent",
    num_steps: int = 50,
    revision_rate: float = 0.1,
    violation_rate: float = 0.05,
    num_distinct_actions: int = 5,
) -> AgentLog:
    """Build a synthetic agent log for benchmarking."""
    import random
    random.seed(42)

    log = AgentLog(agent_id=agent_id)

    action_names = [f"action_{i}" for i in range(num_distinct_actions)]
    score = 0.3

    for step in range(num_steps):
        # Gradually improving scores with some noise
        score = min(1.0, max(-1.0, score + random.gauss(0.01, 0.05)))

        revised = random.random() < revision_rate
        violated = random.random() < violation_rate
        reflected = step % 10 == 0 and step > 0

        entry = AgentLogEntry(
            step=step,
            timestamp=float(step),
            eval_score=score,
            eval_confidence=0.9,
            action_name=random.choice(action_names),
            action_cost=random.uniform(0.1, 1.0),
            goal_revised=revised,
            constraint_violated=violated,
            reflection_triggered=reflected,
        )
        log.entries.append(entry)

        if revised:
            log.goal_revisions.append(
                GoalRevised(
                    source_id=agent_id,
                    revision=GoalRevision(
                        previous_goal_id=f"g-{step}",
                        new_goal_id=f"g-{step + 1}",
                        reason="teleological_update",
                    ),
                    previous_objective=None,
                    new_objective=None,
                    timestamp=float(step),
                )
            )

    return log


def main() -> None:
    print("=== Benchmark Suite ===\n")

    # -- Build synthetic logs -------------------------------------------------
    log_a = _build_synthetic_log(
        agent_id="agent-stable",
        num_steps=50,
        revision_rate=0.05,
        violation_rate=0.02,
        num_distinct_actions=8,
    )
    log_b = _build_synthetic_log(
        agent_id="agent-volatile",
        num_steps=50,
        revision_rate=0.3,
        violation_rate=0.15,
        num_distinct_actions=3,
    )

    print(f"Agent-stable: {log_a.num_steps} steps, {log_a.revision_count} revisions")
    print(f"Agent-volatile: {log_b.num_steps} steps, {log_b.revision_count} revisions")
    print()

    # -- Compute metrics for each agent ---------------------------------------
    engine = MetricsEngine()  # Uses all 7 default metrics

    print(f"Registered metrics: {engine.metric_names}\n")

    report_a = engine.build_report("agent-stable", log_a)
    report_b = engine.build_report("agent-volatile", log_b)

    # -- Print summaries ------------------------------------------------------
    print(report_a.summary())
    print()
    print(report_b.summary())
    print()

    # -- Compute all agents at once -------------------------------------------
    all_results = engine.compute_all_agents({
        "agent-stable": log_a,
        "agent-volatile": log_b,
    })

    print("=== Comparison ===")
    for metric_name in engine.metric_names:
        val_a = report_a.get_metric(metric_name)
        val_b = report_b.get_metric(metric_name)
        if val_a and val_b:
            diff = val_a.value - val_b.value
            winner = "stable" if diff > 0 else "volatile" if diff < 0 else "tie"
            print(
                f"  {metric_name:<30s}  "
                f"stable={val_a.value:.4f}  "
                f"volatile={val_b.value:.4f}  "
                f"({winner})"
            )

    # -- Serialisation demo ---------------------------------------------------
    print("\n=== Report serialisation (dict) ===")
    d = report_a.to_dict()
    print(f"Keys: {list(d.keys())}")
    print(f"Metric names: {list(d['metrics'].keys())}")

    print("\nDone.")


if __name__ == "__main__":
    main()
