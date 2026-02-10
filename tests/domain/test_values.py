"""Tests for domain value objects."""

from __future__ import annotations

import pytest
import numpy as np

from synthetic_teleology.domain.enums import Direction, StateSource, ConstraintType
from synthetic_teleology.domain.values import (
    ObjectiveVector,
    EvalSignal,
    ConstraintSpec,
    ActionSpec,
    PolicySpec,
    StateSnapshot,
    GoalRevision,
)


# ===================================================================== #
#  ObjectiveVector                                                        #
# ===================================================================== #


class TestObjectiveVector:
    """Tests for ObjectiveVector construction, validation, distance, equality."""

    def test_construction(self) -> None:
        ov = ObjectiveVector(
            values=(1.0, 2.0),
            directions=(Direction.MAXIMIZE, Direction.MINIMIZE),
        )
        assert ov.dimension == 2
        assert ov.values == (1.0, 2.0)
        assert ov.weights is None

    def test_construction_with_weights(self) -> None:
        ov = ObjectiveVector(
            values=(1.0, 2.0, 3.0),
            directions=(Direction.APPROACH, Direction.MAINTAIN, Direction.MAXIMIZE),
            weights=(0.5, 0.3, 0.2),
        )
        assert ov.dimension == 3
        assert ov.weights == (0.5, 0.3, 0.2)

    def test_validation_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="values length"):
            ObjectiveVector(
                values=(1.0, 2.0),
                directions=(Direction.APPROACH,),
            )

    def test_validation_weights_mismatch(self) -> None:
        with pytest.raises(ValueError, match="weights length"):
            ObjectiveVector(
                values=(1.0, 2.0),
                directions=(Direction.APPROACH, Direction.APPROACH),
                weights=(1.0,),
            )

    def test_distance_to_self_is_zero(self) -> None:
        ov = ObjectiveVector(
            values=(3.0, 4.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        assert ov.distance_to(ov) == pytest.approx(0.0)

    def test_distance_to_other(self) -> None:
        ov1 = ObjectiveVector(
            values=(0.0, 0.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        ov2 = ObjectiveVector(
            values=(3.0, 4.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        assert ov1.distance_to(ov2) == pytest.approx(5.0)

    def test_distance_with_weights(self) -> None:
        ov1 = ObjectiveVector(
            values=(0.0, 0.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
            weights=(4.0, 1.0),
        )
        ov2 = ObjectiveVector(
            values=(1.0, 2.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
            weights=(4.0, 1.0),
        )
        # sqrt(4*(1-0)^2 + 1*(2-0)^2) = sqrt(4+4) = sqrt(8)
        expected = np.sqrt(8.0)
        assert ov1.distance_to(ov2) == pytest.approx(expected)

    def test_distance_dimension_mismatch(self) -> None:
        ov1 = ObjectiveVector(values=(1.0,), directions=(Direction.APPROACH,))
        ov2 = ObjectiveVector(
            values=(1.0, 2.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        with pytest.raises(ValueError, match="Dimension mismatch"):
            ov1.distance_to(ov2)

    def test_with_values(self) -> None:
        ov = ObjectiveVector(
            values=(1.0, 2.0),
            directions=(Direction.MAXIMIZE, Direction.MINIMIZE),
            weights=(0.5, 0.5),
        )
        ov2 = ov.with_values((3.0, 4.0))
        assert ov2.values == (3.0, 4.0)
        assert ov2.directions == ov.directions
        assert ov2.weights == ov.weights

    def test_immutability(self) -> None:
        ov = ObjectiveVector(
            values=(1.0, 2.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        with pytest.raises(AttributeError):
            ov.values = (3.0, 4.0)  # type: ignore[misc]

    def test_equality(self) -> None:
        ov1 = ObjectiveVector(
            values=(1.0, 2.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        ov2 = ObjectiveVector(
            values=(1.0, 2.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        assert ov1 == ov2

    def test_inequality(self) -> None:
        ov1 = ObjectiveVector(
            values=(1.0, 2.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        ov2 = ObjectiveVector(
            values=(1.0, 3.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        assert ov1 != ov2


# ===================================================================== #
#  EvalSignal                                                             #
# ===================================================================== #


class TestEvalSignal:
    """Tests for EvalSignal construction, validation, properties."""

    def test_valid_construction(self) -> None:
        sig = EvalSignal(score=0.5, confidence=0.9)
        assert sig.score == 0.5
        assert sig.confidence == 0.9
        assert sig.is_satisfactory is True

    def test_negative_signal(self) -> None:
        sig = EvalSignal(score=-0.3, confidence=0.8)
        assert sig.is_satisfactory is False
        assert sig.magnitude == pytest.approx(0.3)

    def test_score_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="score must be in"):
            EvalSignal(score=1.5)

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            EvalSignal(score=0.0, confidence=1.5)

    def test_boundary_values(self) -> None:
        EvalSignal(score=-1.0, confidence=0.0)
        EvalSignal(score=1.0, confidence=1.0)

    def test_immutability(self) -> None:
        sig = EvalSignal(score=0.5)
        with pytest.raises(AttributeError):
            sig.score = 0.6  # type: ignore[misc]

    def test_dimension_scores(self) -> None:
        sig = EvalSignal(score=0.5, dimension_scores=(0.3, 0.7))
        assert sig.dimension_scores == (0.3, 0.7)


# ===================================================================== #
#  ConstraintSpec                                                         #
# ===================================================================== #


class TestConstraintSpec:
    """Tests for ConstraintSpec construction."""

    def test_construction(self) -> None:
        spec = ConstraintSpec(
            name="test-constraint",
            constraint_type=ConstraintType.HARD,
            parameters={"max_cost": 100},
        )
        assert spec.name == "test-constraint"
        assert spec.constraint_type == ConstraintType.HARD
        assert spec.parameters["max_cost"] == 100

    def test_immutability(self) -> None:
        spec = ConstraintSpec(name="x", constraint_type=ConstraintType.SOFT)
        with pytest.raises(AttributeError):
            spec.name = "y"  # type: ignore[misc]


# ===================================================================== #
#  ActionSpec                                                             #
# ===================================================================== #


class TestActionSpec:
    """Tests for ActionSpec construction."""

    def test_construction(self) -> None:
        action = ActionSpec(name="move", parameters={"delta": (1.0, 0.0)}, cost=0.5)
        assert action.name == "move"
        assert action.cost == 0.5
        assert action.parameters["delta"] == (1.0, 0.0)

    def test_default_id_generated(self) -> None:
        a1 = ActionSpec(name="a")
        a2 = ActionSpec(name="b")
        assert a1.action_id != a2.action_id

    def test_immutability(self) -> None:
        action = ActionSpec(name="x")
        with pytest.raises(AttributeError):
            action.name = "y"  # type: ignore[misc]


# ===================================================================== #
#  PolicySpec                                                             #
# ===================================================================== #


class TestPolicySpec:
    """Tests for PolicySpec construction, stochastic/deterministic."""

    def test_deterministic_policy(self) -> None:
        a1 = ActionSpec(name="a")
        a2 = ActionSpec(name="b")
        policy = PolicySpec(actions=(a1, a2))
        assert policy.size == 2
        assert policy.is_stochastic is False

    def test_stochastic_policy(self) -> None:
        a1 = ActionSpec(name="a")
        a2 = ActionSpec(name="b")
        policy = PolicySpec(actions=(a1, a2), probabilities=(0.3, 0.7))
        assert policy.is_stochastic is True

    def test_probabilities_must_sum_to_one(self) -> None:
        a1 = ActionSpec(name="a")
        a2 = ActionSpec(name="b")
        with pytest.raises(ValueError, match="probabilities must sum to 1"):
            PolicySpec(actions=(a1, a2), probabilities=(0.3, 0.3))

    def test_probabilities_length_mismatch(self) -> None:
        a1 = ActionSpec(name="a")
        with pytest.raises(ValueError, match="probabilities length"):
            PolicySpec(actions=(a1,), probabilities=(0.5, 0.5))

    def test_empty_policy(self) -> None:
        policy = PolicySpec()
        assert policy.size == 0
        assert policy.is_stochastic is False


# ===================================================================== #
#  StateSnapshot                                                          #
# ===================================================================== #


class TestStateSnapshot:
    """Tests for StateSnapshot construction and helpers."""

    def test_construction(self) -> None:
        snap = StateSnapshot(timestamp=1.0, values=(1.0, 2.0, 3.0))
        assert snap.dimension == 3
        assert snap.values == (1.0, 2.0, 3.0)
        assert snap.source == StateSource.ENVIRONMENT

    def test_as_array(self) -> None:
        snap = StateSnapshot(timestamp=1.0, values=(1.0, 2.0))
        arr = snap.as_array()
        np.testing.assert_array_equal(arr, [1.0, 2.0])

    def test_immutability(self) -> None:
        snap = StateSnapshot(timestamp=1.0, values=(1.0,))
        with pytest.raises(AttributeError):
            snap.timestamp = 2.0  # type: ignore[misc]


# ===================================================================== #
#  GoalRevision                                                           #
# ===================================================================== #


class TestGoalRevision:
    """Tests for GoalRevision value object."""

    def test_construction(self) -> None:
        rev = GoalRevision(
            timestamp=1.0,
            previous_goal_id="g1",
            new_goal_id="g2",
            reason="threshold",
        )
        assert rev.previous_goal_id == "g1"
        assert rev.new_goal_id == "g2"
        assert rev.reason == "threshold"

    def test_default_id(self) -> None:
        r1 = GoalRevision()
        r2 = GoalRevision()
        assert r1.revision_id != r2.revision_id

    def test_immutability(self) -> None:
        rev = GoalRevision()
        with pytest.raises(AttributeError):
            rev.reason = "changed"  # type: ignore[misc]
