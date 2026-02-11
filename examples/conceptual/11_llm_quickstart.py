#!/usr/bin/env python3
"""Example 11: LLM-powered teleological agent quickstart.

Demonstrates the v1.2.0 feedback loop:
- WorkingMemory provides initial context for perception
- action_feedback flows tool/action results back to perception
- Enriched observations show eval trends + recent actions to LLM services
- Reasoning trace captures chain-of-thought across all steps

Self-contained: runs with a mock LLM by default (no API key required).
Set ANTHROPIC_API_KEY or OPENAI_API_KEY for real LLM mode.

Run:
    PYTHONPATH=src python examples/conceptual/11_llm_quickstart.py
"""

from __future__ import annotations

import os

from synthetic_teleology.graph import GraphBuilder, WorkingMemory


def _build_mock_model():
    """Build a MockStructuredChatModel with pre-configured realistic responses.

    The mock responses are interleaved in the exact order consumed by the
    teleological loop: evaluate → [revise if triggered] → plan, per step.

    Step 1: score=0.20 (no revision)  → eval + plan
    Step 2: score=0.45 (no revision)  → eval + plan
    Step 3: score=0.70 (revision)     → eval + revision + plan
    Step 4: score=0.92 (revision)     → eval + revision + plan → goal achieved
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
        # --- Step 1: eval (score=0.20, no revision triggered) ---
        EvaluationOutput(
            score=0.20,
            confidence=0.60,
            reasoning=(
                "Initial assessment: team productivity data shows clear bottlenecks "
                "(high meeting load, slow PR reviews, rising incidents). "
                "We're at the beginning — need data gathering first."
            ),
            criteria_scores={
                "Identify at least 3 bottlenecks": 0.3,
                "Propose actionable solutions for each": 0.0,
                "Estimate time-to-impact for each solution": 0.0,
            },
        ),
        # --- Step 1: plan ---
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="survey_team",
                            description="Survey team on bottlenecks",
                        )
                    ],
                    reasoning="Direct team input reveals root causes beyond what metrics show",
                    expected_outcome="Ranked list of bottlenecks from team perspective",
                    confidence=0.8,
                    risk="Low — surveys are non-disruptive",
                ),
            ],
            selected_index=0,
            selection_reasoning="Team survey is the fastest path to validated bottleneck data",
        ),
        # --- Step 2: eval (score=0.45, no revision triggered) ---
        EvaluationOutput(
            score=0.45,
            confidence=0.72,
            reasoning=(
                "Progress: survey data confirms 3 bottlenecks (meeting overhead 73%, "
                "tool fragmentation 58%, unclear priorities 45%). "
                "Bottleneck identification is nearly complete, but no solutions proposed yet."
            ),
            criteria_scores={
                "Identify at least 3 bottlenecks": 0.8,
                "Propose actionable solutions for each": 0.1,
                "Estimate time-to-impact for each solution": 0.0,
            },
        ),
        # --- Step 2: plan ---
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="propose_solutions",
                            description=(
                                "Propose 3 solutions addressing"
                                " each bottleneck"
                            ),
                        )
                    ],
                    reasoning="With bottlenecks identified, we can now propose targeted solutions",
                    expected_outcome="3 actionable solutions with rationale",
                    confidence=0.75,
                    risk="Medium — solutions need team buy-in",
                ),
            ],
            selected_index=0,
            selection_reasoning=(
                "Solution proposal is the logical next step"
                " after bottleneck identification"
            ),
        ),
        # --- Step 3: eval (score=0.70, revision triggered: abs(0.70) >= 0.5) ---
        EvaluationOutput(
            score=0.70,
            confidence=0.80,
            reasoning=(
                "Good progress: 3 solutions proposed (async standups, meeting-free Wednesdays, "
                "decision docs). All criteria partially met. Missing: time-to-impact estimates."
            ),
            criteria_scores={
                "Identify at least 3 bottlenecks": 1.0,
                "Propose actionable solutions for each": 0.7,
                "Estimate time-to-impact for each solution": 0.3,
            },
        ),
        # --- Step 3: revision (should_revise=True) ---
        RevisionOutput(
            should_revise=True,
            reasoning="Goal is on track but criteria should emphasize implementation timelines",
            new_description="Develop and timeline a plan to increase team productivity by 25%",
            new_criteria=[
                "Identify at least 3 bottlenecks",
                "Propose actionable solutions for each",
                "Provide week-by-week implementation timeline",
            ],
        ),
        # --- Step 3: plan ---
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="estimate_impact",
                            description=(
                                "Estimate time-to-impact"
                                " for each solution"
                            ),
                        )
                    ],
                    reasoning="Need quantitative impact estimates to finalize the plan",
                    expected_outcome="Per-solution time and productivity impact estimates",
                    confidence=0.85,
                    risk="Low — estimation uses survey data already collected",
                ),
            ],
            selected_index=0,
            selection_reasoning="Impact estimation closes the remaining gap in our criteria",
        ),
        # --- Step 4: eval (score=0.92, revision triggered, then goal achieved) ---
        EvaluationOutput(
            score=0.92,
            confidence=0.95,
            reasoning=(
                "All criteria met: 3 bottlenecks identified, solutions proposed with "
                "implementation timeline. Estimated 25% productivity gain from combined "
                "interventions (async standups ~5h/wk, meeting-free Wed ~3h, decision docs ~2h)."
            ),
            criteria_scores={
                "Identify at least 3 bottlenecks": 1.0,
                "Propose actionable solutions for each": 0.95,
                "Provide week-by-week implementation timeline": 0.85,
            },
        ),
        # --- Step 4: revision (triggered but no changes needed) ---
        RevisionOutput(
            should_revise=False,
            reasoning="Goal nearly achieved — no revision needed at this stage",
        ),
        # --- Step 4: plan (generated but loop will stop at reflect) ---
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="finalize_plan",
                            description=(
                                "Finalize implementation plan"
                                " with stakeholder sign-off"
                            ),
                        )
                    ],
                    reasoning="All data gathered, plan is complete",
                    expected_outcome="Final approved implementation plan",
                    confidence=0.95,
                    risk="Minimal — plan is well-supported by data",
                ),
            ],
            selected_index=0,
            selection_reasoning="Finalization is the natural conclusion",
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
    # -- Choose model: real LLM if available, mock otherwise --
    real_model = _get_real_model()
    if real_model is not None:
        model = real_model
        mode = "real LLM"
    else:
        model = _build_mock_model()
        mode = "simulated (no API key — set ANTHROPIC_API_KEY or OPENAI_API_KEY for real LLM)"

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
        .with_environment(
            perceive_fn=memory.perceive,
            transition_fn=memory.record,
        )
        .with_max_steps(5)
        .with_num_hypotheses(1)
        .build()
    )

    print("=== LLM Teleological Agent — Quickstart ===")
    print(f"Goal: {initial_state['goal'].description}")
    print(f"Mode: {mode}")
    print()

    # -- Run the teleological loop --
    result = app.invoke(initial_state)

    # -- Print step-by-step results from action_feedback + eval_history --
    eval_history = result.get("eval_history", [])
    action_history = result.get("action_history", [])
    feedback = result.get("action_feedback", [])

    for i, sig in enumerate(eval_history):
        step_num = i + 1
        print(f"--- Step {step_num} ---")

        # Show observation enrichment (from feedback)
        if i < len(feedback):
            if i == 0:
                print("  Observation: Team productivity data...")
            else:
                print("  Observation: Team productivity data...")
                prev_actions = [f["action"] for f in feedback[:i]]
                prev_scores = [f"{e.score:.2f}" for e in eval_history[:i]]
                print(f"    [Feedback] Recent actions: {', '.join(prev_actions)}")
                print(f"    [Feedback] Eval trend: {' -> '.join(prev_scores)}")

        print(f"  Eval: {sig.score:.2f} (confidence {sig.confidence:.2f})")

        if i < len(action_history):
            action = action_history[i]
            print(f"  Plan: {action.name} — \"{action.description}\"")

        print()

    # -- Final summary --
    stop_reason = result.get("stop_reason", "max_steps")
    final_score = eval_history[-1].score if eval_history else 0.0
    print(f"Result: {stop_reason} in {result['step']} steps (score {final_score:.2f})")

    trace = result.get("reasoning_trace", [])
    print(f"Reasoning trace: {len(trace)} entries")

    # -- Show goal revision history --
    goal_history = result.get("goal_history", [])
    if goal_history:
        print(f"Goal revisions: {len(goal_history)}")
        for g in goal_history:
            print(f"  -> \"{g.description[:80]}\"")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
