#!/usr/bin/env python3
"""Example 12: LLM agent with LangChain tools.

Demonstrates:
- Defining LangChain-compatible tools for the agent
- Tool-augmented planning (LLM proposes tool-based actions)
- Action execution via tool_name mapping

This example uses mock tools that don't require API keys.

Run:
    PYTHONPATH=src python examples/conceptual/12_llm_tools.py
"""

from __future__ import annotations

import os
import sys

from synthetic_teleology.graph import GraphBuilder, WorkingMemory

# -- Define LangChain-compatible tools --

class SearchTool:
    """Mock search tool for demonstration."""
    name = "web_search"
    description = "Search the web for information on a topic"

    def __call__(self, query: str) -> str:
        return f"Search results for '{query}': [mock results]"


class CalculatorTool:
    """Mock calculator tool."""
    name = "calculator"
    description = "Perform arithmetic calculations"

    def __call__(self, expression: str) -> str:
        return f"Result of '{expression}': [mock calculation]"


class DatabaseTool:
    """Mock database query tool."""
    name = "query_database"
    description = "Query internal databases for metrics and KPIs"

    def __call__(self, query: str) -> str:
        return f"Database result for '{query}': [mock data]"


def main() -> None:
    model = _get_model()
    if model is None:
        print("No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
        sys.exit(1)

    tools = [SearchTool(), CalculatorTool(), DatabaseTool()]

    # Working memory provides initial sales context for the agent
    memory = WorkingMemory(
        initial_context=(
            "Q4 Sales Summary (preliminary):\n"
            "- Total revenue: $2.1M (Q3: $1.9M)\n"
            "- New customers: 34 (Q3: 28)\n"
            "- Churn rate: 4.2% (Q3: 3.8%)\n"
            "- Average deal size: $18.5k (Q3: $16.2k)\n"
            "- Pipeline value: $4.8M"
        ),
    )

    app, initial_state = (
        GraphBuilder("tool-agent")
        .with_model(model)
        .with_goal(
            "Analyze Q4 sales performance and identify growth opportunities",
            criteria=[
                "Retrieve Q4 sales data",
                "Calculate year-over-year growth rate",
                "Identify top 3 growth opportunities",
            ],
        )
        .with_tools(*tools)
        .with_constraints("Use only authorized data sources")
        .with_environment(
            perceive_fn=memory.perceive,
            transition_fn=memory.record,
        )
        .with_max_steps(4)
        .with_num_hypotheses(2)
        .build()
    )

    print("=== LLM Agent with Tools ===")
    print(f"Goal: {initial_state['goal'].description}")
    print(f"Available tools: {[t.name for t in tools]}")
    print()

    result = app.invoke(initial_state)

    print(f"Steps: {result['step']}")
    print(f"Actions taken: {len(result.get('action_history', []))}")
    for i, action in enumerate(result.get("action_history", [])[:5]):
        print(f"  [{i}] {action}")
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
