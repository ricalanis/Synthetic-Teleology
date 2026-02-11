"""Tests for LLMPlanner with mocked LLM."""

from __future__ import annotations

import pytest

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import PolicySpec, StateSnapshot
from synthetic_teleology.services.llm_planning import (
    ActionProposal,
    LLMPlanner,
    PlanHypothesis,
    PlanningOutput,
    _softmax,
)
from tests.helpers.mock_llm import MockStructuredChatModel


def _sample_planning_output() -> PlanningOutput:
    return PlanningOutput(
        hypotheses=[
            PlanHypothesis(
                actions=[
                    ActionProposal(name="search_market", description="Search market data"),
                    ActionProposal(name="analyze", description="Analyze results"),
                ],
                reasoning="Start with market research",
                expected_outcome="Understanding of market trends",
                confidence=0.8,
                risk="May be slow",
            ),
            PlanHypothesis(
                actions=[
                    ActionProposal(name="ask_expert", description="Consult domain expert"),
                ],
                reasoning="Expert consultation is faster",
                expected_outcome="Direct insight",
                confidence=0.6,
                risk="Expert may not be available",
            ),
            PlanHypothesis(
                actions=[
                    ActionProposal(name="noop", description="Wait for more data"),
                ],
                reasoning="Patience may be warranted",
                expected_outcome="More data arrives",
                confidence=0.3,
                risk="Delay",
            ),
        ],
        selected_index=0,
        selection_reasoning="Market research is most reliable",
    )


class TestLLMPlanner:

    def test_plan_returns_policy(self) -> None:
        model = MockStructuredChatModel(structured_responses=[_sample_planning_output()])
        planner = LLMPlanner(model=model, num_hypotheses=3)
        goal = Goal(name="g", description="Analyze market trends")
        state = StateSnapshot(timestamp=0.0, observation="No data yet")

        policy = planner.plan(goal, state)

        assert isinstance(policy, PolicySpec)
        assert policy.size > 0

    def test_plan_creates_probabilistic_policy(self) -> None:
        model = MockStructuredChatModel(structured_responses=[_sample_planning_output()])
        planner = LLMPlanner(model=model, num_hypotheses=3)
        goal = Goal(name="g", description="test")
        state = StateSnapshot(timestamp=0.0, observation="test")

        policy = planner.plan(goal, state)

        assert len(policy.actions) == 3
        assert len(policy.probabilities) == 3
        assert sum(policy.probabilities) == pytest.approx(1.0)

    def test_plan_highest_confidence_gets_highest_prob(self) -> None:
        model = MockStructuredChatModel(structured_responses=[_sample_planning_output()])
        planner = LLMPlanner(model=model, num_hypotheses=3, temperature=1.0)
        goal = Goal(name="g", description="test")
        state = StateSnapshot(timestamp=0.0, observation="test")

        policy = planner.plan(goal, state)

        assert policy.probabilities[0] > policy.probabilities[1]
        assert policy.probabilities[1] > policy.probabilities[2]

    def test_plan_stores_metadata(self) -> None:
        model = MockStructuredChatModel(structured_responses=[_sample_planning_output()])
        planner = LLMPlanner(model=model, num_hypotheses=3)
        goal = Goal(name="g", description="test")
        state = StateSnapshot(timestamp=0.0, observation="test")

        policy = planner.plan(goal, state)

        assert policy.metadata.get("planner") == "LLMPlanner"
        assert "hypotheses" in policy.metadata
        assert len(policy.metadata["hypotheses"]) == 3

    def test_plan_single_hypothesis(self) -> None:
        output = PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[ActionProposal(name="only_action", description="Single plan")],
                    reasoning="Only option",
                    expected_outcome="Result",
                    confidence=0.9,
                ),
            ],
            selected_index=0,
            selection_reasoning="Only one option",
        )
        model = MockStructuredChatModel(structured_responses=[output])
        planner = LLMPlanner(model=model, num_hypotheses=1)
        goal = Goal(name="g", description="test")
        state = StateSnapshot(timestamp=0.0, observation="test")

        policy = planner.plan(goal, state)
        assert policy.size == 1

    def test_plan_with_tools_description(self) -> None:
        model = MockStructuredChatModel(structured_responses=[_sample_planning_output()])

        class MockTool:
            name = "calculator"
            description = "Performs arithmetic"

        planner = LLMPlanner(model=model, tools=[MockTool()])
        desc = planner._get_tools_description()
        assert "calculator" in desc
        assert "arithmetic" in desc


class TestSoftmax:

    def test_basic_softmax(self) -> None:
        result = _softmax([1.0, 1.0, 1.0])
        assert len(result) == 3
        assert all(abs(p - 1 / 3) < 0.01 for p in result)

    def test_softmax_sums_to_one(self) -> None:
        result = _softmax([0.8, 0.6, 0.3])
        assert sum(result) == pytest.approx(1.0)

    def test_softmax_ordering(self) -> None:
        result = _softmax([0.8, 0.5, 0.2])
        assert result[0] > result[1] > result[2]

    def test_softmax_empty(self) -> None:
        assert _softmax([]) == []

    def test_softmax_temperature(self) -> None:
        sharp = _softmax([0.8, 0.2], temperature=0.1)
        flat = _softmax([0.8, 0.2], temperature=10.0)
        assert abs(sharp[0] - sharp[1]) > abs(flat[0] - flat[1])
