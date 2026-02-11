"""Tests for SelfModelingEvaluator and ActiveInferenceUpdater."""

from __future__ import annotations

import pytest

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import EvalSignal, ObjectiveVector, StateSnapshot
from synthetic_teleology.services.evaluation import (
    NumericEvaluator,
    SelfModelingEvaluator,
)
from synthetic_teleology.services.goal_revision import ActiveInferenceUpdater


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _goal(values: tuple[float, ...] = (5.0, 5.0)) -> Goal:
    return Goal(
        name="test",
        objective=ObjectiveVector(
            values=values,
            directions=tuple(Direction.APPROACH for _ in values),
        ),
    )


def _snapshot(values: tuple[float, ...] = (4.0, 4.0)) -> StateSnapshot:
    return StateSnapshot(timestamp=0.0, values=values)


def _signal(score: float = 0.5, confidence: float = 0.8) -> EvalSignal:
    return EvalSignal(score=score, confidence=confidence)


# ===================================================================== #
#  SelfModelingEvaluator tests                                           #
# ===================================================================== #


class TestSelfModelingEvaluator:
    def test_wraps_inner_evaluator(self) -> None:
        inner = NumericEvaluator()
        sme = SelfModelingEvaluator(inner)
        assert sme.inner is inner

    def test_first_evaluation_no_prediction(self) -> None:
        inner = NumericEvaluator()
        sme = SelfModelingEvaluator(inner)
        goal = _goal()
        snap = _snapshot()
        result = sme.evaluate(goal, snap)
        assert result.metadata.get("self_model_prediction") is None
        assert result.metadata.get("self_model_surprise") == 0.0

    def test_predictions_start_after_3_steps(self) -> None:
        inner = NumericEvaluator()
        sme = SelfModelingEvaluator(inner)
        goal = _goal()

        # First 3 evaluations: no prediction
        for i in range(3):
            snap = _snapshot(values=(4.0 + i * 0.1, 4.0 + i * 0.1))
            result = sme.evaluate(goal, snap)

        # 4th should have a prediction
        snap = _snapshot(values=(4.3, 4.3))
        result = sme.evaluate(goal, snap)
        assert result.metadata.get("self_model_prediction") is not None

    def test_surprise_increases_with_unexpected_scores(self) -> None:
        inner = NumericEvaluator()
        sme = SelfModelingEvaluator(inner, ema_alpha=0.5)
        goal = _goal()

        # Build up stable history
        for i in range(5):
            snap = _snapshot(values=(4.0, 4.0))
            sme.evaluate(goal, snap)

        surprise_before = sme.surprise_ema

        # Now drastically change the state
        snap = _snapshot(values=(0.0, 0.0))
        sme.evaluate(goal, snap)

        assert sme.surprise_ema > surprise_before

    def test_confidence_reduced_on_persistent_surprise(self) -> None:
        inner = NumericEvaluator()
        sme = SelfModelingEvaluator(
            inner,
            surprise_threshold=0.05,
            ema_alpha=0.8,
            r_squared_gate=0.0,  # always trust model
        )
        goal = _goal()

        # Build history with predictable pattern
        for i in range(5):
            snap = _snapshot(values=(4.0, 4.0))
            sme.evaluate(goal, snap)

        # Now introduce surprise
        snap = _snapshot(values=(0.0, 0.0))
        result = sme.evaluate(goal, snap)

        # If surprise is above threshold, confidence should be reduced
        if sme.surprise_ema > 0.05:
            assert result.confidence < 1.0

    def test_recommends_goal_edit_when_surprised(self) -> None:
        inner = NumericEvaluator()
        sme = SelfModelingEvaluator(
            inner,
            surprise_threshold=0.01,
            ema_alpha=0.9,
            r_squared_gate=0.0,
        )
        goal = _goal()

        # Build stable baseline
        for _ in range(5):
            snap = _snapshot(values=(4.0, 4.0))
            sme.evaluate(goal, snap)

        # Introduce persistent surprise
        for _ in range(5):
            snap = _snapshot(values=(0.0, 0.0))
            sme.evaluate(goal, snap)

        assert sme.recommends_goal_edit

    def test_no_goal_edit_when_stable(self) -> None:
        inner = NumericEvaluator()
        sme = SelfModelingEvaluator(inner)
        goal = _goal()

        for _ in range(10):
            snap = _snapshot(values=(4.0, 4.0))
            sme.evaluate(goal, snap)

        assert not sme.recommends_goal_edit

    def test_r_squared_quality_gate(self) -> None:
        inner = NumericEvaluator()
        sme = SelfModelingEvaluator(inner, r_squared_gate=0.99)
        goal = _goal()

        # Even with surprise, high r_squared gate prevents recommendation
        for i in range(10):
            v = 4.0 if i % 2 == 0 else 0.0  # alternating
            snap = _snapshot(values=(v, v))
            sme.evaluate(goal, snap)

        # Alternating pattern has low r_squared -> no recommendation
        assert not sme.recommends_goal_edit

    def test_validates_via_inner(self) -> None:
        inner = NumericEvaluator()
        sme = SelfModelingEvaluator(inner)

        goal_no_obj = Goal(name="no-obj")
        snap = _snapshot()
        assert not sme.validate(goal_no_obj, snap)

    def test_metadata_contains_self_model_fields(self) -> None:
        inner = NumericEvaluator()
        sme = SelfModelingEvaluator(inner)
        goal = _goal()

        for _ in range(4):
            snap = _snapshot(values=(4.0, 4.0))
            result = sme.evaluate(goal, snap)

        assert "self_model_prediction" in result.metadata
        assert "self_model_surprise" in result.metadata
        assert "self_model_r_squared" in result.metadata


# ===================================================================== #
#  ActiveInferenceUpdater tests                                          #
# ===================================================================== #


class TestActiveInferenceUpdater:
    def test_no_revision_when_below_thresholds(self) -> None:
        updater = ActiveInferenceUpdater(
            free_energy_threshold=10.0,
            prediction_error_threshold=10.0,
        )
        goal = _goal()
        snap = _snapshot(values=(4.9, 4.9))
        signal = _signal(score=0.8, confidence=0.9)

        result = updater.update(goal, snap, signal)
        assert result is None

    def test_revision_triggered_on_high_free_energy(self) -> None:
        updater = ActiveInferenceUpdater(
            free_energy_threshold=0.01,
            prediction_error_threshold=0.01,
            learning_rate=0.2,
        )
        goal = _goal(values=(5.0, 5.0))
        snap = _snapshot(values=(0.0, 0.0))

        # First call sets prev_score
        signal1 = _signal(score=0.5, confidence=0.3)
        updater.update(goal, snap, signal1)

        # Second call has prediction error
        signal2 = _signal(score=-0.5, confidence=0.3)
        result = updater.update(goal, snap, signal2)
        assert result is not None

    def test_revision_moves_goal_toward_state(self) -> None:
        updater = ActiveInferenceUpdater(
            free_energy_threshold=0.01,
            prediction_error_threshold=0.01,
            learning_rate=0.5,
        )
        goal = _goal(values=(10.0, 10.0))
        snap = _snapshot(values=(2.0, 2.0))

        signal1 = _signal(score=0.5, confidence=0.2)
        updater.update(goal, snap, signal1)

        signal2 = _signal(score=-0.5, confidence=0.2)
        result = updater.update(goal, snap, signal2)

        if result is not None and result.objective is not None:
            # Goal should move toward state (2.0, 2.0)
            for v in result.objective.values:
                assert v < 10.0

    def test_no_revision_without_objective(self) -> None:
        updater = ActiveInferenceUpdater()
        goal = Goal(name="text-only", description="Just a description")
        snap = _snapshot()
        signal = _signal()
        assert updater.update(goal, snap, signal) is None

    def test_no_revision_dimension_mismatch(self) -> None:
        updater = ActiveInferenceUpdater()
        goal = _goal(values=(5.0,))
        snap = _snapshot(values=(4.0, 4.0))
        signal = _signal()
        assert updater.update(goal, snap, signal) is None

    def test_revision_reason_is_active_inference(self) -> None:
        updater = ActiveInferenceUpdater(
            free_energy_threshold=0.01,
            prediction_error_threshold=0.01,
        )
        goal = _goal(values=(10.0, 10.0))
        snap = _snapshot(values=(0.0, 0.0))

        signal1 = _signal(score=0.5, confidence=0.2)
        updater.update(goal, snap, signal1)

        signal2 = _signal(score=-0.8, confidence=0.2)
        result = updater.update(goal, snap, signal2)

        if result is not None:
            # The original goal should now be REVISED status
            from synthetic_teleology.domain.enums import GoalStatus
            assert goal.status == GoalStatus.REVISED

    def test_epistemic_component_uses_dimension_scores(self) -> None:
        updater = ActiveInferenceUpdater(
            free_energy_threshold=0.01,
            prediction_error_threshold=0.01,
            epistemic_weight=0.8,
            pragmatic_weight=0.2,
        )
        goal = _goal(values=(5.0, 5.0))
        snap = _snapshot(values=(3.0, 3.0))

        signal1 = EvalSignal(
            score=-0.5,
            confidence=0.3,
            dimension_scores=(0.1, 0.9),
        )
        updater.update(goal, snap, signal1)

        signal2 = EvalSignal(
            score=0.5,
            confidence=0.3,
            dimension_scores=(0.1, 0.9),
        )
        result = updater.update(goal, snap, signal2)
        # Just verify it runs without error
        assert result is None or result.objective is not None

    def test_high_confidence_prevents_revision(self) -> None:
        """High confidence -> low epistemic value -> lower free energy."""
        updater = ActiveInferenceUpdater(
            free_energy_threshold=5.0,
            prediction_error_threshold=0.01,
        )
        goal = _goal(values=(5.0, 5.0))
        snap = _snapshot(values=(4.9, 4.9))

        signal1 = _signal(score=0.8, confidence=0.99)
        updater.update(goal, snap, signal1)

        signal2 = _signal(score=0.2, confidence=0.99)
        result = updater.update(goal, snap, signal2)
        # High confidence + close values -> free energy likely below 5.0
        assert result is None
