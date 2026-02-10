"""Tests for goal revision services."""

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
from synthetic_teleology.services.goal_revision import (
    GradientUpdater,
    GoalUpdaterChain,
    ThresholdUpdater,
)


class TestThresholdUpdater:
    """Test ThresholdUpdater triggers/doesn't trigger based on score magnitude."""

    def test_no_revision_below_threshold(self) -> None:
        updater = ThresholdUpdater(threshold=0.5, learning_rate=0.1)
        obj = ObjectiveVector(
            values=(5.0, 5.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(4.5, 4.5))
        signal = EvalSignal(score=0.3, confidence=1.0)  # |0.3| < 0.5

        result = updater.update(goal, state, signal)
        assert result is None

    def test_revision_above_threshold(self) -> None:
        updater = ThresholdUpdater(threshold=0.5, learning_rate=0.1)
        obj = ObjectiveVector(
            values=(5.0, 5.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(0.0, 0.0))
        signal = EvalSignal(score=-0.8, confidence=1.0)  # |-0.8| > 0.5

        result = updater.update(goal, state, signal)
        assert result is not None
        assert result.objective is not None
        # New values should move toward state from goal
        for new_v, old_v, state_v in zip(
            result.objective.values, obj.values, state.values
        ):
            # new = old + lr * (state - old) = 5.0 + 0.1 * (0.0 - 5.0) = 4.5
            assert new_v == pytest.approx(old_v + 0.1 * (state_v - old_v))

    def test_no_revision_when_no_objective(self) -> None:
        updater = ThresholdUpdater(threshold=0.5)
        goal = Goal(name="g", objective=None)
        state = StateSnapshot(timestamp=time.time(), values=(1.0,))
        signal = EvalSignal(score=-0.9)
        assert updater.update(goal, state, signal) is None

    def test_no_revision_dimension_mismatch(self) -> None:
        updater = ThresholdUpdater(threshold=0.5)
        obj = ObjectiveVector(values=(5.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(1.0, 2.0))
        signal = EvalSignal(score=-0.9)
        assert updater.update(goal, state, signal) is None

    def test_invalid_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            ThresholdUpdater(threshold=0.0)

    def test_invalid_learning_rate(self) -> None:
        with pytest.raises(ValueError, match="learning_rate must be in"):
            ThresholdUpdater(learning_rate=0.0)

    def test_original_goal_marked_revised(self) -> None:
        updater = ThresholdUpdater(threshold=0.3, learning_rate=0.1)
        obj = ObjectiveVector(values=(5.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(0.0,))
        signal = EvalSignal(score=-0.8)

        result = updater.update(goal, state, signal)
        assert result is not None
        from synthetic_teleology.domain.enums import GoalStatus
        assert goal.status == GoalStatus.REVISED


class TestGradientUpdater:
    """Test GradientUpdater direction-aware updates."""

    def test_no_revision_without_dimension_scores(self) -> None:
        updater = GradientUpdater(learning_rate=0.05)
        obj = ObjectiveVector(values=(5.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(3.0,))
        signal = EvalSignal(score=-0.5)  # no dimension_scores

        result = updater.update(goal, state, signal)
        assert result is None

    def test_revision_with_gradient(self) -> None:
        updater = GradientUpdater(learning_rate=0.1, min_gradient_norm=0.0)
        obj = ObjectiveVector(
            values=(5.0, 5.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(3.0, 7.0))
        signal = EvalSignal(
            score=-0.5,
            dimension_scores=(0.5, 0.8),
        )

        result = updater.update(goal, state, signal)
        assert result is not None
        assert result.objective is not None

    def test_no_revision_small_gradient(self) -> None:
        updater = GradientUpdater(learning_rate=0.1, min_gradient_norm=100.0)
        obj = ObjectiveVector(
            values=(5.0,),
            directions=(Direction.APPROACH,),
        )
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(4.99,))
        signal = EvalSignal(score=0.99, dimension_scores=(0.99,))

        result = updater.update(goal, state, signal)
        assert result is None

    def test_invalid_learning_rate(self) -> None:
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            GradientUpdater(learning_rate=-0.1)


class TestGoalUpdaterChain:
    """Test GoalUpdaterChain first-non-None-wins logic."""

    def test_first_updater_wins(self) -> None:
        # First updater with low threshold -> triggers
        updater1 = ThresholdUpdater(threshold=0.1, learning_rate=0.1)
        updater2 = ThresholdUpdater(threshold=0.9, learning_rate=0.5)
        chain = GoalUpdaterChain([updater1, updater2])

        obj = ObjectiveVector(values=(5.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(0.0,))
        signal = EvalSignal(score=-0.5)

        result = chain.update(goal, state, signal)
        assert result is not None
        # Should use updater1's learning_rate of 0.1
        expected = 5.0 + 0.1 * (0.0 - 5.0)
        assert result.objective is not None
        assert result.objective.values[0] == pytest.approx(expected)

    def test_fallthrough_to_second(self) -> None:
        updater1 = ThresholdUpdater(threshold=0.99, learning_rate=0.1)  # won't trigger
        updater2 = ThresholdUpdater(threshold=0.1, learning_rate=0.5)

        chain = GoalUpdaterChain([updater1, updater2])

        obj = ObjectiveVector(values=(5.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(0.0,))
        signal = EvalSignal(score=-0.5)

        result = chain.update(goal, state, signal)
        assert result is not None

    def test_no_updater_returns_none(self) -> None:
        updater1 = ThresholdUpdater(threshold=0.99)
        chain = GoalUpdaterChain([updater1])

        obj = ObjectiveVector(values=(5.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(4.9,))
        signal = EvalSignal(score=0.1)

        result = chain.update(goal, state, signal)
        assert result is None

    def test_empty_chain_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            GoalUpdaterChain([])

    def test_updaters_property(self) -> None:
        u1 = ThresholdUpdater(threshold=0.5)
        u2 = ThresholdUpdater(threshold=0.9)
        chain = GoalUpdaterChain([u1, u2])
        assert len(chain.updaters) == 2
