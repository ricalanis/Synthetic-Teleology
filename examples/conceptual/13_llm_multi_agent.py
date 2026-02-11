#!/usr/bin/env python3
"""Example 13: LLM-powered multi-agent coordination.

Demonstrates:
- build_multi_agent_graph() with AgentConfig for multi-agent orchestration
- Two LLM agents with different goals (growth & retention)
- LLM-powered negotiation (propose-critique-synthesize protocol)
- WorkingMemory per agent for environmental context
- Negotiation produces shared direction that influences subsequent rounds

Self-contained: runs with mock LLMs by default (no API key required).
Set ANTHROPIC_API_KEY or OPENAI_API_KEY for real LLM mode.

Run:
    PYTHONPATH=src python examples/conceptual/13_llm_multi_agent.py
"""

from __future__ import annotations

import json
import os

from synthetic_teleology.graph import WorkingMemory
from synthetic_teleology.graph.multi_agent import AgentConfig, build_multi_agent_graph


def _build_growth_mock():
    """Mock LLM for the growth agent (2 rounds x 1 step each).

    Each step: EvaluationOutput + PlanningOutput (no revision with scores > -0.3).
    """
    from synthetic_teleology.services.llm_evaluation import EvaluationOutput
    from synthetic_teleology.services.llm_planning import (
        ActionProposal,
        PlanHypothesis,
        PlanningOutput,
    )
    from synthetic_teleology.testing import MockStructuredChatModel

    return MockStructuredChatModel(structured_responses=[
        # --- Round 1 step 1: eval ---
        EvaluationOutput(
            score=0.35,
            confidence=0.65,
            reasoning="CAC at $42 below threshold, but growth rate 6.5% below 10% target.",
            criteria_scores={"CAC < $50": 0.7, "Growth > 10%": 0.1},
        ),
        # --- Round 1 step 1: plan ---
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="expand_marketing",
                            description="Expand to social media and content marketing channels",
                        )
                    ],
                    reasoning="Diversifying channels reduces CAC while increasing reach",
                    expected_outcome="15% increase in lead volume from new channels",
                    confidence=0.7,
                ),
            ],
            selected_index=0,
            selection_reasoning="Channel expansion is highest-leverage growth action",
        ),
        # --- Round 2 step 1: eval (after negotiation shared direction injected) ---
        EvaluationOutput(
            score=0.55,
            confidence=0.75,
            reasoning=(
                "Marketing expansion working. Growth rate up to 8%."
                " Aligned with shared direction."
            ),
            criteria_scores={"CAC < $50": 0.85, "Growth > 10%": 0.4},
        ),
        # --- Round 2 step 1: plan ---
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="launch_referral",
                            description="Launch referral program with 2x incentive",
                        )
                    ],
                    reasoning="Referrals have lowest CAC and complement retention strategy",
                    expected_outcome="Additional 5% growth from referrals",
                    confidence=0.8,
                ),
            ],
            selected_index=0,
            selection_reasoning="Referral program compounds growth cost-effectively",
        ),
    ])


def _build_retention_mock():
    """Mock LLM for the retention agent (2 rounds x 1 step each)."""
    from synthetic_teleology.services.llm_evaluation import EvaluationOutput
    from synthetic_teleology.services.llm_planning import (
        ActionProposal,
        PlanHypothesis,
        PlanningOutput,
    )
    from synthetic_teleology.testing import MockStructuredChatModel

    return MockStructuredChatModel(structured_responses=[
        # --- Round 1 step 1: eval ---
        EvaluationOutput(
            score=0.40,
            confidence=0.60,
            reasoning="Churn 6.2% above 5% target. NPS 38, below 50 target.",
            criteria_scores={"Churn < 5%": 0.3, "NPS > 50": 0.25},
        ),
        # --- Round 1 step 1: plan ---
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="improve_onboarding",
                            description="Redesign onboarding with guided setup and milestones",
                        )
                    ],
                    reasoning="First 30 days determine retention — onboarding is highest-impact",
                    expected_outcome="40% reduction in 30-day churn",
                    confidence=0.75,
                ),
            ],
            selected_index=0,
            selection_reasoning="Onboarding impacts both churn and NPS",
        ),
        # --- Round 2 step 1: eval ---
        EvaluationOutput(
            score=0.60,
            confidence=0.72,
            reasoning="Onboarding improvements showing results. Churn trending to 5.1%.",
            criteria_scores={"Churn < 5%": 0.6, "NPS > 50": 0.5},
        ),
        # --- Round 2 step 1: plan ---
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="loyalty_program",
                            description="Launch tiered loyalty program with usage-based rewards",
                        )
                    ],
                    reasoning="Loyalty increases switching costs and complements growth referrals",
                    expected_outcome="NPS boost to 55+, further churn reduction",
                    confidence=0.8,
                ),
            ],
            selected_index=0,
            selection_reasoning="Loyalty provides sustained retention improvement",
        ),
    ])


def _build_negotiation_mock():
    """Mock LLM for negotiation: propose(x2) + critique + synthesize.

    The LLMNegotiator uses raw model.invoke(prompt) (not with_structured_output),
    so responses are JSON strings returned as AIMessage content.
    """
    from synthetic_teleology.testing import MockStructuredChatModel

    return MockStructuredChatModel(structured_responses=[
        # Propose: growth agent
        json.dumps({
            "direction": "Pursue cost-efficient growth through organic channels and referrals",
            "reasoning": "Low CAC channels maximize ROI while building sustainable pipeline",
            "priority_dimensions": ["CAC optimization", "channel diversification"],
            "confidence": 0.75,
        }),
        # Propose: retention agent
        json.dumps({
            "direction": "Invest in onboarding and loyalty to reduce churn below 5%",
            "reasoning": "Retention is cheaper than acquisition; onboarding is the lever",
            "priority_dimensions": ["onboarding quality", "customer loyalty"],
            "confidence": 0.70,
        }),
        # Critique
        json.dumps({
            "agreements": [
                "Both agents focus on sustainable, cost-efficient strategies",
                "Both recognize the value of long-term customer relationships",
            ],
            "disagreements": [
                "Growth prioritizes new acquisition; retention prioritizes existing customers",
            ],
            "synthesis_hints": [
                "Referral programs serve both growth and retention",
                "Balance acquisition spend with retention investment",
            ],
        }),
        # Synthesize
        json.dumps({
            "shared_direction": (
                "Balance growth (referral programs, channel expansion) "
                "with retention (onboarding, loyalty) for sustainable unit economics"
            ),
            "revised_criteria": [
                "CAC < $50", "Growth > 10%", "Churn < 5%", "NPS > 50",
            ],
            "confidence": 0.82,
            "reasoning": (
                "Referral growth and loyalty retention are complementary — "
                "both reduce cost per retained customer"
            ),
        }),
    ])


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
        growth_model = retention_model = negotiation_model = real_model
        mode = "real LLM"
    else:
        growth_model = _build_growth_mock()
        retention_model = _build_retention_mock()
        negotiation_model = _build_negotiation_mock()
        mode = "simulated (no API key — set ANTHROPIC_API_KEY or OPENAI_API_KEY for real LLM)"

    print("=== Multi-Agent Coordination ===")
    print(f"Mode: {mode}")
    print()

    # Per-agent working memory with domain context
    growth_memory = WorkingMemory(initial_context=(
        "Growth metrics (current):\n"
        "- Monthly new customers: 28\n"
        "- CAC: $42\n"
        "- Monthly growth rate: 6.5%\n"
        "- Marketing budget: $100k (60% utilized)\n"
        "- Top channel: organic search (45% of leads)"
    ))
    retention_memory = WorkingMemory(initial_context=(
        "Retention metrics (current):\n"
        "- Monthly churn rate: 6.2%\n"
        "- NPS score: 38\n"
        "- 30-day activation rate: 52%\n"
        "- Support ticket volume: 340/month\n"
        "- Average customer lifetime: 14 months"
    ))

    # Configure two agents with different goals
    configs = [
        AgentConfig(
            agent_id="growth-agent",
            goal="Maximize revenue growth through new customer acquisition",
            model=growth_model,
            criteria=["Customer acquisition cost < $50", "Monthly growth rate > 10%"],
            perceive_fn=growth_memory.perceive,
            transition_fn=growth_memory.record,
            max_steps_per_round=1,
        ),
        AgentConfig(
            agent_id="retention-agent",
            goal="Improve customer retention and reduce churn",
            model=retention_model,
            criteria=["Churn rate < 5%", "NPS score > 50"],
            perceive_fn=retention_memory.perceive,
            transition_fn=retention_memory.record,
            max_steps_per_round=1,
        ),
    ]

    for c in configs:
        print(f"Agent: {c.agent_id} — \"{c.goal}\"")
    print()

    # Build multi-agent graph with LLM negotiation
    app = build_multi_agent_graph(
        agent_configs=configs,
        negotiation_model=negotiation_model,
    )

    # Run the coordination loop (1 negotiation round)
    result = app.invoke({
        "max_rounds": 1,
        "agent_results": {},
        "events": [],
        "reasoning_trace": [],
    })

    # --- Display agent results ---
    agent_results = result.get("agent_results", {})

    for agent_id, agent_result in agent_results.items():
        goal = agent_result.get("final_goal")
        signal = agent_result.get("eval_signal")
        steps = agent_result.get("steps", 0)
        stop = agent_result.get("stop_reason", "unknown")

        print(f"--- {agent_id} ---")
        if goal:
            desc = getattr(goal, "description", str(goal))
            print(f"  Goal: {desc[:80]}")
        if signal:
            print(f"  Final score: {signal.score:.2f} (confidence: {signal.confidence:.2f})")
        print(f"  Steps: {steps}, Stop: {stop}")
        print()

    # --- Negotiation outcome ---
    shared = result.get("shared_direction", "")
    neg_round = result.get("negotiation_round", 0)

    if shared:
        print("=== Negotiation Outcome ===")
        print(f"  Rounds: {neg_round}")
        print(f"  Shared direction: {shared}")
        print()

    # --- Event timeline ---
    events = result.get("events", [])
    print(f"=== Timeline ({len(events)} events) ===")
    for evt in events:
        etype = evt.get("type", "?")
        if etype == "agent_round_completed":
            print(f"  [{etype}] {evt['agent_id']}: score={evt.get('eval_score', 0):.2f}")
        elif etype == "llm_negotiation_completed":
            print(f"  [{etype}] round={evt['round']}, confidence={evt.get('confidence', 0):.2f}")
        else:
            print(f"  [{etype}]")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
