"""Tests for LLMReviser with mocked LLM."""

from __future__ import annotations

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import EvalSignal, ObjectiveVector, StateSnapshot
from synthetic_teleology.services.llm_revision import LLMReviser, RevisionOutput
from tests.helpers.mock_llm import MockStructuredChatModel


class TestLLMReviser:

    def test_no_revision_returns_none(self) -> None:
        model = MockStructuredChatModel(structured_responses=[
            RevisionOutput(should_revise=False, reasoning="Goal is on track")
        ])
        reviser = LLMReviser(model=model)
        goal = Goal(name="test-goal", description="Increase revenue by 20%")
        state = StateSnapshot(timestamp=0.0, observation="Revenue at $118k")
        signal = EvalSignal(score=0.6, confidence=0.8, reasoning="Good progress")

        result = reviser.update(goal, state, signal)
        assert result is None

    def test_revision_returns_new_goal(self) -> None:
        model = MockStructuredChatModel(structured_responses=[
            RevisionOutput(
                should_revise=True,
                reasoning="Goal is too ambitious given market conditions",
                new_description="Increase revenue by 10%",
                new_criteria=["Revenue > $110k"],
            )
        ])
        reviser = LLMReviser(model=model)
        goal = Goal(
            name="test-goal",
            description="Increase revenue by 20%",
            success_criteria=["Revenue > $120k"],
        )
        state = StateSnapshot(timestamp=0.0, observation="Market downturn detected")
        signal = EvalSignal(score=-0.3, confidence=0.7, reasoning="Regression")

        new_goal = reviser.update(goal, state, signal)

        assert new_goal is not None
        assert new_goal.description == "Increase revenue by 10%"
        assert new_goal.success_criteria == ["Revenue > $110k"]
        assert new_goal.version == goal.version + 1

    def test_revision_preserves_name(self) -> None:
        model = MockStructuredChatModel(structured_responses=[
            RevisionOutput(
                should_revise=True,
                reasoning="Need to adjust",
                new_description="Adjusted goal",
            )
        ])
        reviser = LLMReviser(model=model)
        goal = Goal(name="my-goal", description="Original")
        state = StateSnapshot(timestamp=0.0, observation="state")
        signal = EvalSignal(score=-0.5, reasoning="bad")

        new_goal = reviser.update(goal, state, signal)
        assert new_goal is not None
        assert new_goal.name == "my-goal"

    def test_revision_with_numeric_values(self) -> None:
        model = MockStructuredChatModel(structured_responses=[
            RevisionOutput(
                should_revise=True,
                reasoning="Adjusting targets",
                new_values=[3.0, 3.0],
            )
        ])
        reviser = LLMReviser(model=model)
        objective = ObjectiveVector(
            values=(5.0, 5.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        goal = Goal(name="numeric-goal", description="reach target", objective=objective)
        state = StateSnapshot(timestamp=0.0, observation="stuck at 2.5")
        signal = EvalSignal(score=-0.2, reasoning="not converging")

        new_goal = reviser.update(goal, state, signal)
        assert new_goal is not None
        assert new_goal.objective is not None
        assert new_goal.objective.values == (3.0, 3.0)

    def test_revision_without_criteria_change(self) -> None:
        model = MockStructuredChatModel(structured_responses=[
            RevisionOutput(
                should_revise=True,
                reasoning="Description change only",
                new_description="Updated description",
                new_criteria=None,
            )
        ])
        reviser = LLMReviser(model=model)
        goal = Goal(
            name="g",
            description="Original",
            success_criteria=["criterion1", "criterion2"],
        )
        state = StateSnapshot(timestamp=0.0, observation="state")
        signal = EvalSignal(score=-0.1, reasoning="slight regression")

        new_goal = reviser.update(goal, state, signal)
        assert new_goal is not None
        assert new_goal.description == "Updated description"
        assert new_goal.success_criteria == ["criterion1", "criterion2"]
