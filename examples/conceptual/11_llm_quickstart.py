#!/usr/bin/env python3
"""Example 11: LLM-powered teleological agent quickstart.

Demonstrates the new LLM-first API:
- Natural language goals instead of numeric vectors
- LLM-backed evaluation, planning, and revision
- Multi-hypothesis planning with confidence scores
- Reasoning trace for interpretability

Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.

Run:
    PYTHONPATH=src python examples/conceptual/11_llm_quickstart.py
"""

from __future__ import annotations

import os
import sys

from synthetic_teleology.graph import GraphBuilder, WorkingMemory


def main() -> None:
    # -- Choose model based on available API key --
    model = _get_model()
    if model is None:
        print("No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
        print("(This example requires a real LLM. See 01_basic_loop.py for numeric mode.)")
        sys.exit(1)

    # -- Working memory gives the agent an environment to reason about --
    memory = WorkingMemory(
        initial_context=(
            "Team productivity data (Q4 2025):\n"
            "- Sprint velocity: 42 pts/sprint (down from 55 in Q2)\n"
            "- PR review time: avg 3.2 days (target: <1 day)\n"
            "- Meeting load: 18 hrs/week per IC (up 40% YoY)\n"
            "- On-call incidents: 12/month (up from 4 in Q2)\n"
            "- Developer satisfaction (eNPS): 22 (down from 45)"
        ),
    )

    # -- Build LLM-powered teleological agent --
    app, initial_state = (
        GraphBuilder("llm-quickstart")
        .with_model(model)
        .with_goal(
            "Develop a plan to increase team productivity by 25%",
            criteria=[
                "Identify at least 3 bottlenecks",
                "Propose actionable solutions for each",
                "Estimate time-to-impact for each solution",
            ],
        )
        .with_constraints(
            "Do not recommend reducing headcount",
            "Solutions must be implementable within 1 quarter",
        )
        .with_environment(
            perceive_fn=memory.perceive,
            transition_fn=memory.record,
        )
        .with_max_steps(5)
        .with_num_hypotheses(3)
        .build()
    )

    print("=== LLM-Powered Teleological Agent ===")
    print(f"Goal: {initial_state['goal'].description}")
    print(f"Criteria: {initial_state['goal'].success_criteria}")
    print(f"Max steps: {initial_state['max_steps']}")
    print()

    # -- Run the teleological loop --
    result = app.invoke(initial_state)

    # -- Inspect results --
    print(f"Steps completed: {result['step']}")
    print(f"Stop reason: {result.get('stop_reason', 'max_steps')}")
    print()

    if result.get("eval_signal"):
        sig = result["eval_signal"]
        print(f"Final eval score: {sig.score:.4f}")
        print(f"Confidence: {sig.confidence:.4f}")
        if sig.reasoning:
            print(f"Reasoning: {sig.reasoning[:300]}")
    print()

    # -- Show reasoning trace --
    trace = result.get("reasoning_trace", [])
    if trace:
        print(f"Reasoning trace ({len(trace)} entries):")
        for i, entry in enumerate(trace[:5]):
            print(f"  [{i}] {str(entry)[:120]}")
        if len(trace) > 5:
            print(f"  ... and {len(trace) - 5} more entries")
    print()

    print("Done.")


def _get_model():
    """Try to create a LangChain model from available API keys."""
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
