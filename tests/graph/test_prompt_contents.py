"""Tests for v1.5.0 Phase 5: Prompt content verification.

Verifies that LLM services pass the correct information to prompts.
"""

from __future__ import annotations

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import EvalSignal, StateSnapshot
from synthetic_teleology.services.llm_evaluation import EvaluationOutput, LLMEvaluator
from synthetic_teleology.services.llm_planning import LLMPlanner, PlanningOutput
from synthetic_teleology.services.llm_revision import LLMReviser, RevisionOutput
from synthetic_teleology.testing.mock_llm import PromptCapturingMock


class TestLLMEvaluatorPrompts:

    def test_prompt_contains_goal_description(self) -> None:
        mock = PromptCapturingMock(
            structured_responses=[
                EvaluationOutput(
                    score=0.5,
                    confidence=0.8,
                    reasoning="Looks good",
                    criteria_scores={},
                ),
            ],
        )
        evaluator = LLMEvaluator(model=mock)
        goal = Goal(
            name="revenue-goal",
            description="Increase revenue by 20%",
            success_criteria=["Revenue > $120k", "Positive trend"],
        )
        state = StateSnapshot(timestamp=0.0, observation="Revenue is $100k")

        evaluator.evaluate(goal, state)

        assert len(mock.captured_prompts) == 1
        prompt = mock.captured_prompts[0]
        assert "Increase revenue by 20%" in prompt
        assert "Revenue > $120k" in prompt
        assert "Revenue is $100k" in prompt


class TestLLMPlannerPrompts:

    def test_prompt_contains_goal_and_constraints(self) -> None:
        mock = PromptCapturingMock(
            structured_responses=[
                PlanningOutput(
                    hypotheses=[],
                    selected_index=0,
                    selection_reasoning="No plans",
                ),
            ],
        )
        planner = LLMPlanner(model=mock, num_hypotheses=2)
        goal = Goal(
            name="perf-goal",
            description="Optimize database queries",
            success_criteria=["P99 < 100ms"],
        )
        state = StateSnapshot(
            timestamp=0.0,
            observation="Current P99 is 500ms",
        )

        planner.plan(goal, state)

        assert len(mock.captured_prompts) == 1
        prompt = mock.captured_prompts[0]
        assert "Optimize database queries" in prompt
        assert "P99 < 100ms" in prompt
        assert "Current P99 is 500ms" in prompt


class TestLLMReviserPrompts:

    def test_prompt_contains_eval_score(self) -> None:
        mock = PromptCapturingMock(
            structured_responses=[
                RevisionOutput(
                    should_revise=False,
                    reasoning="No need",
                ),
            ],
        )
        reviser = LLMReviser(model=mock)
        goal = Goal(name="g1", description="Test goal")
        state = StateSnapshot(timestamp=0.0, observation="Bad state")
        signal = EvalSignal(
            score=-0.75,
            confidence=0.9,
            reasoning="Poor performance",
        )

        reviser.update(goal, state, signal)

        assert len(mock.captured_prompts) == 1
        prompt = mock.captured_prompts[0]
        assert "-0.75" in prompt
        assert "Test goal" in prompt


class TestEnrichedObservation:

    def test_enriched_observation_includes_feedback(self) -> None:
        from synthetic_teleology.graph.nodes import _build_enriched_observation

        state = {
            "step": 3,
            "action_feedback": [
                {
                    "action": "search",
                    "tool_name": "web_search",
                    "result": "Found 10 results",
                    "step": 2,
                },
            ],
            "eval_history": [],
            "goal_history": [],
        }
        result = _build_enriched_observation("Base observation", state)
        assert "search" in result
        assert "web_search" in result
        assert "Found 10 results" in result
