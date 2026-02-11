"""Tests for LLM service error handling, timeouts, and fail-safe behavior."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.runnables import RunnableSerializable
from pydantic import ConfigDict

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import EvalSignal, StateSnapshot
from synthetic_teleology.services.llm_constraints import LLMConstraintChecker
from synthetic_teleology.services.llm_evaluation import LLMEvaluator
from synthetic_teleology.services.llm_planning import LLMPlanner
from synthetic_teleology.services.llm_revision import LLMReviser
from tests.helpers.mock_llm import MockStructuredChatModel


def _make_goal() -> Goal:
    return Goal(name="test", description="Test goal", success_criteria=["Pass"])


def _make_snapshot() -> StateSnapshot:
    return StateSnapshot(timestamp=0.0, observation="Test observation")


class _ErrorRunnable(RunnableSerializable):
    """A runnable that always raises an error."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        raise RuntimeError("LLM service unavailable")


class _SlowRunnable(RunnableSerializable):
    """A runnable that sleeps longer than any reasonable timeout."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        time.sleep(5)
        return None


def _make_error_model() -> MockStructuredChatModel:
    """Create a mock model whose chain will be replaced with an error runnable."""
    return MockStructuredChatModel(structured_responses=[])


class TestEvaluatorErrorMetadata:

    def test_evaluator_error_metadata(self) -> None:
        """Error signal has llm_error=True in metadata and confidence=0.0."""
        model = _make_error_model()
        evaluator = LLMEvaluator(model=model)
        # Replace the chain with one that always errors
        evaluator._chain = _ErrorRunnable()

        signal = evaluator.evaluate(_make_goal(), _make_snapshot())

        assert signal.score == 0.0
        assert signal.confidence == 0.0
        assert signal.metadata["llm_error"] is True
        assert signal.metadata["error_type"] == "RuntimeError"
        assert "unavailable" in signal.metadata["error"]


class TestPlannerErrorNoopFallback:

    def test_planner_error_noop_fallback(self) -> None:
        """Error returns non-empty policy with noop action."""
        model = _make_error_model()
        planner = LLMPlanner(model=model)
        planner._chain = _ErrorRunnable()

        policy = planner.plan(_make_goal(), _make_snapshot())

        assert policy.size > 0
        assert policy.actions[0].name == "noop_fallback"
        assert policy.metadata["llm_error"] is True


class TestConstraintCheckerFailsClosed:

    def test_constraint_checker_fails_closed(self) -> None:
        """Error returns (False, 'error...') — fail-closed, not fail-open."""
        model = _make_error_model()
        checker = LLMConstraintChecker(model=model, constraints=["Be safe"])
        checker._chain = _ErrorRunnable()

        passed, message = checker.check(_make_goal(), _make_snapshot())

        assert passed is False
        assert "constraint check failed" in message.lower()


class TestEvaluatorTimeout:

    def test_evaluator_timeout(self) -> None:
        """Timeout triggers error handling path."""
        model = _make_error_model()
        evaluator = LLMEvaluator(model=model, timeout=0.001)
        evaluator._chain = _SlowRunnable()

        signal = evaluator.evaluate(_make_goal(), _make_snapshot())

        assert signal.confidence == 0.0
        assert signal.metadata["llm_error"] is True


class TestReviserErrorReturnsNone:

    def test_reviser_error_returns_none(self) -> None:
        """Confirm None return on error (safe fallback)."""
        model = _make_error_model()
        reviser = LLMReviser(model=model)
        reviser._chain = _ErrorRunnable()

        result = reviser.update(
            _make_goal(),
            _make_snapshot(),
            EvalSignal(score=-0.5, confidence=0.8),
        )

        assert result is None
