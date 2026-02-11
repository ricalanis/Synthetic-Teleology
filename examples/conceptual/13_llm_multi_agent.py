#!/usr/bin/env python3
"""Example 13: LLM-powered multi-agent coordination.

Demonstrates:
- Multiple LLM agents with different goals, each with its own feedback loop
- Per-agent WorkingMemory and MockStructuredChatModel
- GraphBuilder per agent (shows internal mechanics)
- Comparing agent results for coordination

Self-contained: runs with mock LLMs by default (no API key required).
Set ANTHROPIC_API_KEY or OPENAI_API_KEY for real LLM mode.

Run:
    PYTHONPATH=src python examples/conceptual/13_llm_multi_agent.py
"""

from __future__ import annotations

import os

from synthetic_teleology.graph import GraphBuilder, WorkingMemory


def _build_growth_mock():
    """Mock responses for the growth agent (2 steps).

    Step 1: score=0.35 (no revision) → expand_marketing
    Step 2: score=0.75 (revision)    → launch_referral → done
    """
    from synthetic_teleology.services.llm_evaluation import EvaluationOutput
    from synthetic_teleology.services.llm_planning import (
        ActionProposal,
        PlanHypothesis,
        PlanningOutput,
    )
    from synthetic_teleology.services.llm_revision import RevisionOutput
    from synthetic_teleology.testing import MockStructuredChatModel

    return MockStructuredChatModel(structured_responses=[
        # --- Step 1: eval ---
        EvaluationOutput(
            score=0.35,
            confidence=0.65,
            reasoning=(
                "Initial marketing analysis shows potential"
                " in untapped channels. CAC is $42, below threshold."
            ),
            criteria_scores={
                "Customer acquisition cost < $50": 0.7,
                "Monthly growth rate > 10%": 0.1,
            },
        ),
        # --- Step 1: plan ---
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="expand_marketing",
                            description=(
                                "Expand to social media and"
                                " content marketing channels"
                            ),
                        )
                    ],
                    reasoning="Diversifying channels can reduce CAC while increasing reach",
                    expected_outcome="15% increase in lead volume from new channels",
                    confidence=0.7,
                ),
            ],
            selected_index=0,
            selection_reasoning="Channel expansion is the highest-leverage growth action",
        ),
        # --- Step 2: eval (score=0.75, revision triggered) ---
        EvaluationOutput(
            score=0.75,
            confidence=0.82,
            reasoning=(
                "Marketing expansion yielding results."
                " Growth rate up to 8%, need to push past 10%."
            ),
            criteria_scores={
                "Customer acquisition cost < $50": 0.9,
                "Monthly growth rate > 10%": 0.6,
            },
        ),
        # --- Step 2: revision ---
        RevisionOutput(
            should_revise=True,
            reasoning="Add referral program criterion to complement marketing spend",
            new_description="Maximize revenue growth through acquisition and referral programs",
            new_criteria=["CAC < $50", "Monthly growth > 10%", "Referral program launched"],
        ),
        # --- Step 2: plan ---
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="launch_referral",
                            description=(
                                "Launch customer referral"
                                " program with 2x incentive"
                            ),
                        )
                    ],
                    reasoning="Referrals have lowest CAC and highest LTV",
                    expected_outcome="Additional 5% monthly growth from referrals",
                    confidence=0.8,
                ),
            ],
            selected_index=0,
            selection_reasoning="Referral program compounds growth cost-effectively",
        ),
    ])


def _build_retention_mock():
    """Mock responses for the retention agent (2 steps).

    Step 1: score=0.40 (no revision) → improve_onboarding
    Step 2: score=0.70 (revision)    → loyalty_program → done
    """
    from synthetic_teleology.services.llm_evaluation import EvaluationOutput
    from synthetic_teleology.services.llm_planning import (
        ActionProposal,
        PlanHypothesis,
        PlanningOutput,
    )
    from synthetic_teleology.services.llm_revision import RevisionOutput
    from synthetic_teleology.testing import MockStructuredChatModel

    return MockStructuredChatModel(structured_responses=[
        # --- Step 1: eval ---
        EvaluationOutput(
            score=0.40,
            confidence=0.60,
            reasoning=(
                "Churn rate at 6.2%, above 5% target."
                " NPS at 38, below 50 target."
                " Onboarding is weak spot."
            ),
            criteria_scores={
                "Churn rate < 5%": 0.3,
                "NPS score > 50": 0.25,
            },
        ),
        # --- Step 1: plan ---
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="improve_onboarding",
                            description=(
                                "Redesign onboarding flow with"
                                " guided setup and milestone"
                                " tracking"
                            ),
                        )
                    ],
                    reasoning=(
                        "First 30 days determine retention"
                        " — onboarding is the highest-impact fix"
                    ),
                    expected_outcome="Reduce 30-day churn by 40%",
                    confidence=0.75,
                ),
            ],
            selected_index=0,
            selection_reasoning="Onboarding directly impacts both churn and NPS",
        ),
        # --- Step 2: eval (score=0.70, revision triggered) ---
        EvaluationOutput(
            score=0.70,
            confidence=0.78,
            reasoning=(
                "Onboarding improvements showing results."
                " Churn trending down to 5.1%. NPS up to 44."
            ),
            criteria_scores={
                "Churn rate < 5%": 0.65,
                "NPS score > 50": 0.55,
            },
        ),
        # --- Step 2: revision ---
        RevisionOutput(
            should_revise=True,
            reasoning="Add loyalty program as retention complement to onboarding improvements",
            new_description="Improve retention through onboarding and loyalty initiatives",
            new_criteria=["Churn rate < 5%", "NPS > 50", "Loyalty program active"],
        ),
        # --- Step 2: plan ---
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="loyalty_program",
                            description=(
                                "Launch tiered loyalty program"
                                " with usage-based rewards"
                            ),
                        )
                    ],
                    reasoning="Loyalty programs increase switching costs and improve NPS",
                    expected_outcome="Further 15% reduction in churn, NPS boost to 55+",
                    confidence=0.8,
                ),
            ],
            selected_index=0,
            selection_reasoning="Loyalty program provides sustained retention improvement",
        ),
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


def _run_agent(name: str, model, goal: str, criteria: list[str], context: str, max_steps: int = 2):
    """Build and run a single agent, returning the result dict."""
    memory = WorkingMemory(initial_context=context)

    app, initial_state = (
        GraphBuilder(name)
        .with_model(model)
        .with_goal(goal, criteria=criteria)
        .with_environment(
            perceive_fn=memory.perceive,
            transition_fn=memory.record,
        )
        .with_max_steps(max_steps)
        .with_num_hypotheses(1)
        .build()
    )

    return app.invoke(initial_state)


def main() -> None:
    real_model = _get_real_model()
    if real_model is not None:
        growth_model = real_model
        retention_model = real_model
        mode = "real LLM"
    else:
        growth_model = _build_growth_mock()
        retention_model = _build_retention_mock()
        mode = "simulated (no API key — set ANTHROPIC_API_KEY or OPENAI_API_KEY for real LLM)"

    print("=== Multi-Agent Coordination (Simulated) ===")
    print(f"Mode: {mode}")
    print()

    agents = {
        "growth-agent": {
            "model": growth_model,
            "goal": "Maximize revenue growth through new customer acquisition",
            "criteria": ["Customer acquisition cost < $50", "Monthly growth rate > 10%"],
            "context": (
                "Growth metrics (current):\n"
                "- Monthly new customers: 28\n"
                "- CAC: $42\n"
                "- Monthly growth rate: 6.5%\n"
                "- Marketing budget: $100k (60% utilized)\n"
                "- Top channel: organic search (45% of leads)"
            ),
        },
        "retention-agent": {
            "model": retention_model,
            "goal": "Improve customer retention and reduce churn",
            "criteria": ["Churn rate < 5%", "NPS score > 50"],
            "context": (
                "Retention metrics (current):\n"
                "- Monthly churn rate: 6.2%\n"
                "- NPS score: 38\n"
                "- 30-day activation rate: 52%\n"
                "- Support ticket volume: 340/month\n"
                "- Average customer lifetime: 14 months"
            ),
        },
    }

    for name, cfg in agents.items():
        print(f"Agent: {name} — \"{cfg['goal']}\"")
    print()

    # -- Run each agent independently --
    results = {}
    for name, cfg in agents.items():
        results[name] = _run_agent(
            name=name,
            model=cfg["model"],
            goal=cfg["goal"],
            criteria=cfg["criteria"],
            context=cfg["context"],
            max_steps=2,
        )

    # -- Display per-agent results --
    for name, result in results.items():
        eval_history = result.get("eval_history", [])
        action_history = result.get("action_history", [])
        feedback = result.get("action_feedback", [])
        goal_history = result.get("goal_history", [])

        scores = [f"{e.score:.2f}" for e in eval_history]
        actions = [a.name for a in action_history]

        print(f"--- {name} ---")
        print(f"  Score progression: {' -> '.join(scores)}")
        print(f"  Actions: {actions}")

        if goal_history:
            print(f"  Goal revised: \"{goal_history[-1].description[:70]}\"")

        # Show feedback loop in action
        if len(feedback) > 1:
            last_fb = feedback[-1]
            print(f"  Last feedback: {last_fb['action']} (step {last_fb['step']})")

        print()

    # -- Coordination summary --
    print("=== Coordination Summary ===")
    growth_actions = [a.name for a in results["growth-agent"].get("action_history", [])]
    retention_actions = [a.name for a in results["retention-agent"].get("action_history", [])]
    growth_final = results["growth-agent"].get("eval_history", [])
    retention_final = results["retention-agent"].get("eval_history", [])

    g_score = growth_final[-1].score if growth_final else 0.0
    r_score = retention_final[-1].score if retention_final else 0.0

    print(f"  growth-agent: score {g_score:.2f}, actions: {growth_actions}")
    print(f"  retention-agent: score {r_score:.2f}, actions: {retention_actions}")
    print()
    print("  Combined direction: Balance acquisition growth with retention improvements")
    print("  (In production, use build_multi_agent_graph() with LLMNegotiator for")
    print("   automated negotiation between agents.)")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
