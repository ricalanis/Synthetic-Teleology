"""Tests for LLMEvaluator with mocked LLM."""

from __future__ import annotations

import pytest

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import EvalSignal, StateSnapshot
from synthetic_teleology.services.llm_evaluation import EvaluationOutput, LLMEvaluator
from tests.helpers.mock_llm import MockStructuredChatModel


def _make_model(*outputs: EvaluationOutput) -> MockStructuredChatModel:
    return MockStructuredChatModel(structured_responses=list(outputs))


class TestLLMEvaluator:

    def test_evaluate_returns_eval_signal(self) -> None:
        model = _make_model(
            EvaluationOutput(
                score=0.7,
                confidence=0.9,
                reasoning="Revenue grew 14% towards 20% target",
                criteria_scores={"revenue_growth": 0.7},
            )
        )
        evaluator = LLMEvaluator(model=model)
        goal = Goal(
            name="revenue",
            description="Increase revenue by 20%",
            success_criteria=["Revenue > $120k"],
        )
        state = StateSnapshot(timestamp=0.0, observation="Current revenue: $114k")

        signal = evaluator.evaluate(goal, state)

        assert isinstance(signal, EvalSignal)
        assert signal.score == pytest.approx(0.7)
        assert signal.confidence == pytest.approx(0.9)
        assert "Revenue grew" in signal.reasoning

    def test_evaluate_clamps_score(self) -> None:
        model = _make_model(
            EvaluationOutput(score=1.0, confidence=1.0, reasoning="Perfect")
        )
        evaluator = LLMEvaluator(model=model)
        goal = Goal(name="g", description="test goal")
        state = StateSnapshot(timestamp=0.0, observation="done")

        signal = evaluator.evaluate(goal, state)
        assert -1.0 <= signal.score <= 1.0
        assert 0.0 <= signal.confidence <= 1.0

    def test_evaluate_with_criteria(self) -> None:
        model = _make_model(
            EvaluationOutput(
                score=0.5,
                confidence=0.8,
                reasoning="Partial progress",
                criteria_scores={"speed": 0.6, "quality": 0.4},
            )
        )
        evaluator = LLMEvaluator(model=model, criteria=["extra criterion"])
        goal = Goal(name="g", description="test", success_criteria=["speed", "quality"])
        state = StateSnapshot(timestamp=0.0, observation="partial progress")

        signal = evaluator.evaluate(goal, state)
        assert signal.criteria_scores == {"speed": 0.6, "quality": 0.4}

    def test_validate_with_description(self) -> None:
        model = _make_model(
            EvaluationOutput(score=0.0, confidence=1.0, reasoning="ok")
        )
        evaluator = LLMEvaluator(model=model)
        goal = Goal(name="g", description="has description")
        state = StateSnapshot(timestamp=0.0)
        assert evaluator.validate(goal, state) is True

    def test_validate_with_only_name(self) -> None:
        model = _make_model(
            EvaluationOutput(score=0.0, confidence=1.0, reasoning="ok")
        )
        evaluator = LLMEvaluator(model=model)
        goal = Goal(name="named-goal")
        state = StateSnapshot(timestamp=0.0)
        assert evaluator.validate(goal, state) is True

    def test_evaluate_numeric_state(self) -> None:
        model = _make_model(
            EvaluationOutput(
                score=0.3,
                confidence=0.7,
                reasoning="Some progress in numeric space",
            )
        )
        evaluator = LLMEvaluator(model=model)
        goal = Goal(name="numeric", description="reach target")
        state = StateSnapshot(timestamp=0.0, values=(2.5, 3.0))

        signal = evaluator.evaluate(goal, state)
        assert signal.score == pytest.approx(0.3)
