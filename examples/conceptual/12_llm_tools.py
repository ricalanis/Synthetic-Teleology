#!/usr/bin/env python3
"""Example 12: LLM agent with LangChain tools.

Demonstrates:
- Tool-augmented planning (LLM proposes actions with tool_name)
- Tool results flow into action_feedback and enrich subsequent observations
- WorkingMemory accumulates tool execution history

Self-contained: runs with a mock LLM by default (no API key required).
Set ANTHROPIC_API_KEY or OPENAI_API_KEY for real LLM mode.

Run:
    PYTHONPATH=src python examples/conceptual/12_llm_tools.py
"""

from __future__ import annotations

import os

from synthetic_teleology.graph import GraphBuilder, WorkingMemory

# -- Mock LangChain-compatible tools -----------------------------------------


class SearchTool:
    """Mock web search tool."""

    name = "web_search"
    description = "Search the web for information on a topic"

    def invoke(self, params: dict) -> str:
        return (
            "Search results: Found 3 relevant articles"
            " on sales growth strategies"
        )

    def __call__(self, query: str) -> str:
        return self.invoke({"query": query})


class CalculatorTool:
    """Mock calculator tool."""

    name = "calculator"
    description = "Perform arithmetic calculations"

    def invoke(self, params: dict) -> str:
        return (
            "Calculation result: Q4 YoY growth = 10.5%,"
            " projected annual revenue = $8.8M"
        )

    def __call__(self, expression: str) -> str:
        return self.invoke({"expression": expression})


class DatabaseTool:
    """Mock database query tool."""

    name = "query_database"
    description = "Query internal databases for metrics and KPIs"

    def invoke(self, params: dict) -> str:
        return (
            "Database result: Top customers by revenue — "
            "Acme Corp ($450k), Beta Inc ($380k), Gamma Ltd ($290k). "
            "Churn risk: 12 accounts flagged."
        )

    def __call__(self, query: str) -> str:
        return self.invoke({"query": query})


# -- Mock model --------------------------------------------------------------


def _build_mock_model():
    """Build mock responses for 3-step tool-augmented agent.

    Step 1: score=0.25 (no revision) → query_database tool
    Step 2: score=0.60 (revision)    → web_search tool
    Step 3: score=0.93 (revision)    → calculator tool → goal achieved
    """
    from synthetic_teleology.services.llm_evaluation import EvaluationOutput
    from synthetic_teleology.services.llm_planning import (
        ActionProposal,
        PlanHypothesis,
        PlanningOutput,
    )
    from synthetic_teleology.services.llm_revision import RevisionOutput
    from synthetic_teleology.testing import MockStructuredChatModel

    responses = [
        # --- Step 1: eval (score=0.25, no revision) ---
        EvaluationOutput(
            score=0.25,
            confidence=0.55,
            reasoning=(
                "Initial state: have Q4 summary but need"
                " detailed data from internal databases."
            ),
            criteria_scores={
                "Retrieve Q4 sales data": 0.3,
                "Calculate year-over-year growth rate": 0.0,
                "Identify top 3 growth opportunities": 0.0,
            },
        ),
        # --- Step 1: plan (tool: query_database) ---
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="fetch_sales_data",
                            description="Query database for detailed Q4 sales breakdown",
                            tool_name="query_database",
                            parameters={"query": "Q4 sales by customer segment"},
                        )
                    ],
                    reasoning="Need granular data before analysis",
                    expected_outcome="Detailed sales breakdown by customer and segment",
                    confidence=0.8,
                ),
            ],
            selected_index=0,
            selection_reasoning="Database query is the essential first step",
        ),
        # --- Step 2: eval (score=0.60, revision triggered: abs(0.60) >= 0.5) ---
        EvaluationOutput(
            score=0.60,
            confidence=0.75,
            reasoning=(
                "Database results retrieved. We have customer-level revenue data and "
                "churn risk flags. Still need external benchmarks for growth analysis."
            ),
            criteria_scores={
                "Retrieve Q4 sales data": 0.9,
                "Calculate year-over-year growth rate": 0.3,
                "Identify top 3 growth opportunities": 0.4,
            },
        ),
        # --- Step 2: revision ---
        RevisionOutput(
            should_revise=True,
            reasoning=(
                "Refine criteria to include competitive"
                " benchmarking from web research"
            ),
            new_description=(
                "Analyze Q4 sales and identify growth"
                " opportunities with competitive context"
            ),
            new_criteria=[
                "Retrieve and analyze Q4 sales data",
                "Calculate YoY growth with industry comparison",
                "Identify top 3 growth opportunities with competitive context",
            ],
        ),
        # --- Step 2: plan (tool: web_search) ---
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="research_competitors",
                            description="Search for industry growth benchmarks",
                            tool_name="web_search",
                            parameters={"query": "SaaS industry Q4 2025 growth benchmarks"},
                        )
                    ],
                    reasoning="Need external data to contextualize our growth rate",
                    expected_outcome="Industry benchmarks for comparison",
                    confidence=0.7,
                ),
            ],
            selected_index=0,
            selection_reasoning="Competitive context is needed before identifying opportunities",
        ),
        # --- Step 3: eval (score=0.93, revision triggered, goal achieved) ---
        EvaluationOutput(
            score=0.93,
            confidence=0.92,
            reasoning=(
                "Strong position: Q4 revenue up 10.5% with top customers identified. "
                "Industry benchmarks show we're above median. "
                "Three clear growth opportunities: expand Acme Corp account, "
                "reduce churn in flagged accounts, enter adjacent market."
            ),
            criteria_scores={
                "Retrieve and analyze Q4 sales data": 1.0,
                "Calculate YoY growth with industry comparison": 0.9,
                "Identify top 3 growth opportunities with competitive context": 0.85,
            },
        ),
        # --- Step 3: revision (not needed) ---
        RevisionOutput(
            should_revise=False,
            reasoning="All criteria well-met, goal is achieved",
        ),
        # --- Step 3: plan (calculator for final numbers) ---
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="calculate_projections",
                            description="Calculate projected annual revenue",
                            tool_name="calculator",
                            parameters={"expression": "2.1M * 4 * 1.05"},
                        )
                    ],
                    reasoning="Final calculation to support growth projection",
                    expected_outcome="Annual revenue projection with growth rate",
                    confidence=0.9,
                ),
            ],
            selected_index=0,
            selection_reasoning="Quantitative projection finalizes the analysis",
        ),
    ]

    return MockStructuredChatModel(structured_responses=responses)


def _get_real_model():
    """Try to create a real LangChain model from available API keys."""
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


def main() -> None:
    real_model = _get_real_model()
    if real_model is not None:
        model = real_model
        mode = "real LLM"
    else:
        model = _build_mock_model()
        mode = "simulated (no API key — set ANTHROPIC_API_KEY or OPENAI_API_KEY for real LLM)"

    tools = [SearchTool(), CalculatorTool(), DatabaseTool()]

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
        .with_environment(
            perceive_fn=memory.perceive,
            transition_fn=memory.record,
        )
        .with_max_steps(4)
        .with_num_hypotheses(1)
        .build()
    )

    print("=== LLM Agent with Tools ===")
    print(f"Goal: {initial_state['goal'].description}")
    print(f"Available tools: {[t.name for t in tools]}")
    print(f"Mode: {mode}")
    print()

    result = app.invoke(initial_state)

    # -- Print step-by-step with tool results --
    eval_history = result.get("eval_history", [])
    action_history = result.get("action_history", [])
    feedback = result.get("action_feedback", [])

    for i, sig in enumerate(eval_history):
        step_num = i + 1
        print(f"--- Step {step_num} ---")

        # Show feedback from previous steps
        if i > 0 and feedback:
            prev_fb = feedback[:i]
            for fb in prev_fb[-2:]:  # Show last 2 feedback entries
                tool = fb.get("tool_name") or "direct"
                result_str = str(fb.get("result", ""))[:80]
                print(f"    [Tool feedback] {fb['action']} (via {tool}): {result_str}")

        print(f"  Eval: {sig.score:.2f} (confidence {sig.confidence:.2f})")

        if i < len(action_history):
            action = action_history[i]
            tool_info = f" [tool: {action.tool_name}]" if action.tool_name else ""
            print(f"  Plan: {action.name}{tool_info} — \"{action.description}\"")

        # Show tool result from this step's feedback
        if i < len(feedback):
            fb = feedback[i]
            if fb.get("result"):
                print(f"  Tool result: {str(fb['result'])[:100]}")

        print()

    # -- Final summary --
    stop_reason = result.get("stop_reason", "max_steps")
    final_score = eval_history[-1].score if eval_history else 0.0
    print(f"Result: {stop_reason} in {result['step']} steps (score {final_score:.2f})")
    print(f"Actions taken: {len(action_history)}")
    for i, action in enumerate(action_history):
        tool_info = f" (via {action.tool_name})" if action.tool_name else ""
        print(f"  [{i+1}] {action.name}{tool_info}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
