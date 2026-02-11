#!/usr/bin/env python3
"""Example 06: Stream events -> measurement metrics.

Demonstrates:
- Streaming the teleological graph execution
- Collecting stream events for measurement
- Using ``stream_to_agent_log_entries()`` to bridge to the metrics layer

Run:
    PYTHONPATH=src python examples/06_benchmark_measurement.py
"""

from __future__ import annotations

from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph import (
    GraphBuilder,
    collect_stream_events,
    stream_to_agent_log_entries,
)


def main() -> None:
    env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))

    app, initial_state = (
        GraphBuilder("benchmark-agent")
        .with_objective((5.0, 5.0))
        .with_max_steps(20)
        .with_goal_achieved_threshold(0.9)
        .with_environment(
            perceive_fn=lambda: env.observe(),
            act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
            transition_fn=lambda a: env.step(a) if a else None,
        )
        .build()
    )

    print("=== Benchmark Measurement ===")
    print()

    # Stream and collect events
    stream = app.stream(initial_state, stream_mode="updates")
    all_events = collect_stream_events(stream)

    print(f"Total stream events: {len(all_events)}")

    # Count by node type
    node_counts: dict[str, int] = {}
    for ev in all_events:
        node = ev.get("node", "unknown")
        node_counts[node] = node_counts.get(node, 0) + 1

    print("Events by node:")
    for node, count in sorted(node_counts.items()):
        print(f"  {node}: {count}")
    print()

    # Collect agent log entries for measurement
    stream2 = app.stream(initial_state, stream_mode="updates")

    # Reset env for second run
    env2 = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
    initial_state2 = dict(initial_state)
    initial_state2["perceive_fn"] = lambda: env2.observe()
    initial_state2["transition_fn"] = lambda a: env2.step(a) if a else None

    stream2 = app.stream(initial_state2, stream_mode="updates")
    log_entries = stream_to_agent_log_entries(stream2)

    print(f"Agent log entries: {len(log_entries)}")
    for entry in log_entries[:10]:
        print(f"  Step {entry.get('step', '?')}: {entry}")
    if len(log_entries) > 10:
        print(f"  ... and {len(log_entries) - 10} more")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
