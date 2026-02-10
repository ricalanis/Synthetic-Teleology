"""Tests for evaluation services."""

from __future__ import annotations

import time

import pytest

from synthetic_teleology.domain.enums import Direction, StateSource
from synthetic_teleology.domain.values import (
    EvalSignal,
    ObjectiveVector,
    StateSnapshot,
)
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.services.evaluation import (
    CompositeEvaluator,
    NumericEvaluator,
    ReflectiveEvaluator,
)


class TestNumericEvaluator:
    """Tests for NumericEvaluator with known inputs/outputs."""

    def test_perfect_match_gives_score_one(self) -> None:
        evaluator = NumericEvaluator(max_distance=10.0)
        obj = ObjectiveVector(
            values=(5.0, 5.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(5.0, 5.0))

        signal = evaluator.evaluate(goal, state)
        assert signal.score == pytest.approx(1.0)

    def test_far_state_gives_negative_score(self) -> None:
        evaluator = NumericEvaluator(max_distance=5.0)
        obj = ObjectiveVector(
            values=(0.0,),
            directions=(Direction.APPROACH,),
        )
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(10.0,))

        signal = evaluator.evaluate(goal, state)
        assert signal.score == pytest.approx(-1.0)

    def test_half_distance_approach(self) -> None:
        evaluator = NumericEvaluator(max_distance=10.0)
        obj = ObjectiveVector(
            values=(0.0,),
            directions=(Direction.APPROACH,),
        )
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(5.0,))

        signal = evaluator.evaluate(goal, state)
        # 1 - 2*(5/10) = 0.0
        assert signal.score == pytest.approx(0.0)

    def test_maximize_direction(self) -> None:
        evaluator = NumericEvaluator(max_distance=10.0)
        obj = ObjectiveVector(
            values=(5.0,),
            directions=(Direction.MAXIMIZE,),
        )
        goal = Goal(name="g", objective=obj)
        # State exceeds goal -> positive
        state = StateSnapshot(timestamp=time.time(), values=(8.0,))
        signal = evaluator.evaluate(goal, state)
        assert signal.score > 0.0

    def test_minimize_direction(self) -> None:
        evaluator = NumericEvaluator(max_distance=10.0)
        obj = ObjectiveVector(
            values=(5.0,),
            directions=(Direction.MINIMIZE,),
        )
        goal = Goal(name="g", objective=obj)
        # State below goal -> positive for minimize
        state = StateSnapshot(timestamp=time.time(), values=(2.0,))
        signal = evaluator.evaluate(goal, state)
        assert signal.score > 0.0

    def test_validate_true_when_matching(self) -> None:
        evaluator = NumericEvaluator()
        obj = ObjectiveVector(
            values=(1.0, 2.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(1.0, 2.0))
        assert evaluator.validate(goal, state) is True

    def test_validate_false_no_objective(self) -> None:
        evaluator = NumericEvaluator()
        goal = Goal(name="g", objective=None)
        state = StateSnapshot(timestamp=time.time(), values=(1.0,))
        assert evaluator.validate(goal, state) is False

    def test_validate_false_dimension_mismatch(self) -> None:
        evaluator = NumericEvaluator()
        obj = ObjectiveVector(values=(1.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(1.0, 2.0))
        assert evaluator.validate(goal, state) is False

    def test_evaluate_raises_on_invalid(self) -> None:
        evaluator = NumericEvaluator()
        goal = Goal(name="g", objective=None)
        state = StateSnapshot(timestamp=time.time(), values=(1.0,))
        with pytest.raises(ValueError, match="Cannot evaluate"):
            evaluator.evaluate(goal, state)

    def test_dimension_scores_returned(self) -> None:
        evaluator = NumericEvaluator()
        obj = ObjectiveVector(
            values=(0.0, 0.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(1.0, 2.0))

        signal = evaluator.evaluate(goal, state)
        assert len(signal.dimension_scores) == 2

    def test_max_distance_validation(self) -> None:
        with pytest.raises(ValueError, match="max_distance must be positive"):
            NumericEvaluator(max_distance=-1.0)


class TestCompositeEvaluator:
    """Tests for CompositeEvaluator aggregation."""

    def test_single_evaluator(self) -> None:
        inner = NumericEvaluator(max_distance=10.0)
        composite = CompositeEvaluator(evaluators=[(inner, 1.0)])
        obj = ObjectiveVector(
            values=(5.0,),
            directions=(Direction.APPROACH,),
        )
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(5.0,))

        signal = composite.evaluate(goal, state)
        assert signal.score == pytest.approx(1.0)

    def test_weighted_average(self) -> None:
        ev1 = NumericEvaluator(max_distance=10.0)
        ev2 = NumericEvaluator(max_distance=5.0)
        composite = CompositeEvaluator(evaluators=[(ev1, 1.0), (ev2, 1.0)])
        obj = ObjectiveVector(values=(0.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(5.0,))

        signal = composite.evaluate(goal, state)
        # ev1: 1-2*(5/10)=0.0, ev2: 1-2*(5/5)=-1.0 -> avg = -0.5
        assert signal.score == pytest.approx(-0.5)

    def test_skip_invalid_evaluator(self) -> None:
        ev_valid = NumericEvaluator(max_distance=10.0)
        ev_invalid = NumericEvaluator(max_distance=10.0)

        composite = CompositeEvaluator(evaluators=[(ev_valid, 1.0), (ev_invalid, 1.0)])

        obj = ObjectiveVector(values=(5.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(5.0,))

        signal = composite.evaluate(goal, state)
        assert signal.confidence > 0

    def test_no_valid_evaluator_returns_neutral(self) -> None:
        ev = NumericEvaluator(max_distance=10.0)
        composite = CompositeEvaluator(evaluators=[(ev, 1.0)])

        goal = Goal(name="g", objective=None)
        state = StateSnapshot(timestamp=time.time(), values=(5.0,))

        signal = composite.evaluate(goal, state)
        assert signal.score == 0.0
        assert signal.confidence == 0.0

    def test_empty_evaluators_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CompositeEvaluator(evaluators=[])


class TestReflectiveEvaluator:
    """Tests for ReflectiveEvaluator confidence adjustment."""

    def test_wraps_inner_evaluator(self) -> None:
        inner = NumericEvaluator(max_distance=10.0)
        reflective = ReflectiveEvaluator(inner)
        assert reflective.inner is inner

    def test_single_eval_returns_raw_confidence(self) -> None:
        inner = NumericEvaluator(max_distance=10.0)
        reflective = ReflectiveEvaluator(inner)
        obj = ObjectiveVector(values=(5.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(5.0,))

        signal = reflective.evaluate(goal, state)
        assert signal.confidence == pytest.approx(1.0)

    def test_history_accumulates(self) -> None:
        inner = NumericEvaluator(max_distance=10.0)
        reflective = ReflectiveEvaluator(inner, history_size=10)
        obj = ObjectiveVector(values=(0.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)

        for val in [0.0, 1.0, 2.0, 3.0, 4.0]:
            state = StateSnapshot(timestamp=time.time(), values=(val,))
            reflective.evaluate(goal, state)

        assert len(reflective.history) == 5

    def test_drift_reduces_confidence(self) -> None:
        inner = NumericEvaluator(max_distance=10.0)
        reflective = ReflectiveEvaluator(
            inner, history_size=10, drift_threshold=0.1
        )
        # Goal at 0.0 so oscillating between 0 and 10 gives scores ~1.0 and ~-1.0
        obj = ObjectiveVector(values=(0.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)

        # Create oscillating evaluations to trigger drift
        confidences = []
        for val in [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]:
            state = StateSnapshot(timestamp=time.time(), values=(val,))
            signal = reflective.evaluate(goal, state)
            confidences.append(signal.confidence)

        # Later evaluations should have reduced confidence due to high score variance
        assert confidences[-1] < 1.0
