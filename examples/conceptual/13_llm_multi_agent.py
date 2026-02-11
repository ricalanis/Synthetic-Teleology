#!/usr/bin/env python3
"""Example 13: LLM-powered multi-agent coordination.

Demonstrates:
- Multiple LLM agents with different goals
- LLM-powered negotiation between agents
- Per-agent model and tool configuration
- Shared reasoning traces across agents

Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.

Run:
    PYTHONPATH=src python examples/conceptual/13_llm_multi_agent.py
"""

from __future__ import annotations

import os
import sys

from synthetic_teleology.graph.multi_agent import AgentConfig, build_multi_agent_graph


def main() -> None:
    model = _get_model()
    if model is None:
        print("No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
        print("(See 02_multi_agent.py for numeric multi-agent example.)")
        sys.exit(1)

    # -- Configure agents with different goals --
    agents = [
        AgentConfig(
            agent_id="growth-agent",
            goal="Maximize revenue growth through new customer acquisition",
            model=model,
            criteria=["Customer acquisition cost < $50", "Monthly growth rate > 10%"],
            constraints=["Stay within marketing budget of $100k"],
            max_steps_per_round=3,
        ),
        AgentConfig(
            agent_id="retention-agent",
            goal="Improve customer retention and reduce churn",
            model=model,
            criteria=["Churn rate < 5%", "NPS score > 50"],
            constraints=["Do not discount more than 20%"],
            max_steps_per_round=3,
        ),
    ]

    # -- Build multi-agent graph --
    graph = build_multi_agent_graph(
        agent_configs=agents,
        max_rounds=2,
    )

    print("=== LLM Multi-Agent Coordination ===")
    for a in agents:
        print(f"  Agent '{a.agent_id}': {a.goal}")
    print()

    # -- Run --
    initial_state = {
        "model": model,
        "agent_results": {},
        "shared_objective": None,
        "shared_direction": "",
        "negotiation_round": 0,
        "max_rounds": 2,
        "goal_achieved_threshold": 0.8,
        "events": [],
        "reasoning_trace": [],
    }

    result = graph.invoke(initial_state)

    # -- Inspect results --
    print(f"Negotiation rounds: {result.get('negotiation_round', 0)}")
    print(f"Events: {len(result.get('events', []))}")
    print()

    for agent_id, agent_result in result.get("agent_results", {}).items():
        print(f"--- {agent_id} ---")
        print(f"  Steps: {agent_result.get('steps', 0)}")
        print(f"  Stop reason: {agent_result.get('stop_reason', 'unknown')}")
        if agent_result.get("eval_signal"):
            print(f"  Final score: {agent_result['eval_signal'].score:.4f}")
        trace = agent_result.get("reasoning_trace", [])
        if trace:
            print(f"  Reasoning entries: {len(trace)}")
        print()

    print("Done.")


def _get_model():
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.7)
        except ImportError:
            pass
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o", temperature=0.7)
        except ImportError:
            pass
    return None


if __name__ == "__main__":
    main()
