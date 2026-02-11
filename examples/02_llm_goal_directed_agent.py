#!/usr/bin/env python3
"""Example 02: LLM-powered goal-directed agent.

Demonstrates:
- Using ``create_llm_teleological_agent()`` with a mock LLM
- The teleological loop with LLM metadata stored in state
- Streaming events from the graph

Run:
    PYTHONPATH=src python examples/02_llm_goal_directed_agent.py
"""

from __future__ import annotations

from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph import collect_stream_events, create_llm_teleological_agent


def main() -> None:
    env = NumericEnvironment(dimensions=2, initial_state=(1.0, 1.0))

    # Create an LLM-powered agent (mock model for demo)
    app, initial_state = create_llm_teleological_agent(
        model="mock-claude-3.5-sonnet",
        target_values=(8.0, 8.0),
        perceive_fn=lambda: env.observe(),
        transition_fn=lambda a: env.step(a) if a else None,
        max_steps=15,
        goal_achieved_threshold=0.85,
        agent_id="llm-agent",
    )

    print("=== LLM Goal-Directed Agent ===")
    print(f"Model: {initial_state['metadata']['llm_model']}")
    print(f"Goal: approach {initial_state['goal'].objective.values}")
    print(f"Start: {env.observe().values}")
    print()

    # Stream the execution and collect events
    stream = app.stream(initial_state, stream_mode="updates")
    events = collect_stream_events(stream)

    print(f"Stream events: {len(events)}")
    for ev in events[:5]:
        fields = ', '.join(
            f'{k}={v}' for k, v in ev.items()
            if k not in ('node', 'timestamp') and not callable(v) and k != 'events'
        )
        print(f"  [{ev['node']}] {fields}")
    if len(events) > 5:
        print(f"  ... and {len(events) - 5} more events")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
