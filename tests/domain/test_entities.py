"""Tests for domain entities (Goal, Constraint)."""

from __future__ import annotations

import pytest

from synthetic_teleology.domain.entities import Constraint, Goal
from synthetic_teleology.domain.enums import (
    ConstraintType,
    GoalStatus,
)
from synthetic_teleology.domain.values import (
    ConstraintSpec,
    EvalSignal,
    ObjectiveVector,
)

# ===================================================================== #
#  Goal lifecycle                                                         #
# ===================================================================== #


class TestGoal:
    """Test Goal entity lifecycle: create, revise, achieve, abandon, suspend."""

    def test_creation_defaults(self) -> None:
        goal = Goal(name="test")
        assert goal.status == GoalStatus.ACTIVE
        assert goal.is_active is True
        assert goal.is_terminal is False
        assert goal.version == 1

    def test_revise_produces_new_goal(self, sample_objective: ObjectiveVector) -> None:
        goal = Goal(name="g1", objective=sample_objective)
        new_obj = sample_objective.with_values((6.0, 6.0))
        new_goal, revision = goal.revise(new_obj, reason="test-revision")

        # Goal is frozen — original is unchanged
        assert goal.status == GoalStatus.ACTIVE
        assert goal.is_terminal is False
        assert new_goal.status == GoalStatus.ACTIVE
        assert new_goal.version == goal.version + 1
        assert new_goal.objective == new_obj
        assert revision.previous_goal_id == goal.goal_id
        assert revision.new_goal_id == new_goal.goal_id
        assert revision.reason == "test-revision"

    def test_revise_with_eval_signal(self, sample_objective: ObjectiveVector) -> None:
        goal = Goal(name="g1", objective=sample_objective)
        signal = EvalSignal(score=-0.5, confidence=0.8)
        new_obj = sample_objective.with_values((4.0, 4.0))
        new_goal, revision = goal.revise(new_obj, eval_signal=signal)

        assert revision.eval_signal is signal

    def test_achieve(self) -> None:
        original = Goal(name="g1")
        achieved = original.achieve()
        assert achieved.status == GoalStatus.ACHIEVED
        assert achieved.is_terminal is True
        assert achieved.is_active is False
        # Original is unchanged (frozen)
        assert original.status == GoalStatus.ACTIVE

    def test_abandon(self) -> None:
        original = Goal(name="g1")
        abandoned = original.abandon()
        assert abandoned.status == GoalStatus.ABANDONED
        assert abandoned.is_terminal is True
        assert original.status == GoalStatus.ACTIVE

    def test_suspend_and_reactivate(self) -> None:
        original = Goal(name="g1")
        suspended = original.suspend()
        assert suspended.status == GoalStatus.SUSPENDED
        assert suspended.is_active is False
        assert suspended.is_terminal is False

        reactivated = suspended.reactivate()
        assert reactivated.status == GoalStatus.ACTIVE
        assert reactivated.is_active is True

    def test_reactivate_non_suspended_raises(self) -> None:
        goal = Goal(name="g1")
        with pytest.raises(ValueError, match="Only suspended goals"):
            goal.reactivate()

    def test_metadata_preserved_on_revise(self, sample_objective: ObjectiveVector) -> None:
        goal = Goal(name="g1", objective=sample_objective, metadata={"key": "val"})
        new_obj = sample_objective.with_values((7.0, 7.0))
        new_goal, _ = goal.revise(new_obj)
        assert new_goal.metadata == {"key": "val"}

    def test_name_preserved_on_revise(self, sample_objective: ObjectiveVector) -> None:
        goal = Goal(name="my-goal", description="desc", objective=sample_objective)
        new_obj = sample_objective.with_values((7.0, 7.0))
        new_goal, _ = goal.revise(new_obj)
        assert new_goal.name == "my-goal"
        assert new_goal.description == "desc"


# ===================================================================== #
#  Constraint entity                                                      #
# ===================================================================== #


class TestConstraint:
    """Test Constraint entity: activation toggle, properties."""

    def test_creation(self) -> None:
        spec = ConstraintSpec(name="s", constraint_type=ConstraintType.HARD)
        c = Constraint(
            name="safety",
            constraint_type=ConstraintType.HARD,
            spec=spec,
            weight=2.0,
        )
        assert c.name == "safety"
        assert c.constraint_type == ConstraintType.HARD
        assert c.weight == 2.0
        assert c.is_active is True

    def test_deactivate_and_activate(self) -> None:
        spec = ConstraintSpec(name="s", constraint_type=ConstraintType.SOFT)
        c = Constraint(name="soft", constraint_type=ConstraintType.SOFT, spec=spec)
        assert c.is_active is True

        c.deactivate()
        assert c.is_active is False

        c.activate()
        assert c.is_active is True

    def test_unique_ids(self) -> None:
        spec = ConstraintSpec(name="s", constraint_type=ConstraintType.HARD)
        c1 = Constraint(name="a", constraint_type=ConstraintType.HARD, spec=spec)
        c2 = Constraint(name="b", constraint_type=ConstraintType.HARD, spec=spec)
        assert c1.constraint_id != c2.constraint_id

    def test_repr(self) -> None:
        spec = ConstraintSpec(name="s", constraint_type=ConstraintType.HARD)
        c = Constraint(name="safety", constraint_type=ConstraintType.HARD, spec=spec)
        r = repr(c)
        assert "safety" in r
        assert "hard" in r
