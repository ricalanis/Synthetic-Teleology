#!/usr/bin/env python3
"""Example 07: ReAct research agent with tool calling (mock).

Demonstrates:
- ``create_react_teleological_agent()`` with mock tools
- The teleological loop wrapping a ReAct-style agent concept
- Accessing tool metadata in the state

Run:
    PYTHONPATH=src python examples/07_react_research_agent.py
"""

from __future__ import annotations

from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph import create_react_teleological_agent


def main() -> None:
    env = NumericEnvironment(dimensions=1, initial_state=(0.0,))

    # Mock tools (in a real scenario these would be LangChain tools)
    mock_tools = [
        {"name": "web_search", "description": "Search the web for information"},
        {"name": "take_notes", "description": "Save notes about findings"},
        {"name": "summarize", "description": "Summarize collected information"},
    ]

    app, initial_state = create_react_teleological_agent(
        model="mock-claude-3.5-sonnet",
        tools=mock_tools,
        goal_description="Research quantum computing advances in 2025",
        perceive_fn=lambda: env.observe(),
        transition_fn=lambda a: env.step(a) if a else None,
        target_values=(1.0,),
        max_steps=5,
        goal_achieved_threshold=0.9,
        agent_id="researcher",
    )

    print("=== ReAct Research Agent ===")
    print(f"Goal: {initial_state['metadata']['goal_description']}")
    print(f"Model: {initial_state['metadata']['llm_model']}")
    print(f"Tools: {[t['name'] for t in initial_state['metadata']['tools']]}")
    print()

    result = app.invoke(initial_state)

    print(f"Steps: {result['step']}")
    print(f"Stop reason: {result.get('stop_reason', 'none')}")
    print(f"Final eval: {result['eval_signal'].score:.4f}")
    print(f"Actions taken: {len(result['action_history'])}")

    if result["action_history"]:
        print("Action sequence:")
        for i, action in enumerate(result["action_history"]):
            print(f"  {i+1}. {action.name}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
