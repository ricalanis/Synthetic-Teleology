"""Tests for v1.5.0 Phase 2: Goal immutability.

Covers:
- Goal is frozen (assignment raises FrozenInstanceError)
- achieve()/abandon()/suspend()/reactivate() return new Goal
- revise() does NOT mutate original goal
- reflect_node returns updated goal in state dict
"""

from __future__ import annotations

import dataclasses

import pytest

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import GoalStatus
from synthetic_teleology.domain.values import EvalSignal
from synthetic_teleology.graph.nodes import reflect_node


class TestGoalFrozen:

    def test_frozen_assignment_raises(self) -> None:
        goal = Goal(name="test")
        with pytest.raises(dataclasses.FrozenInstanceError):
            goal.status = GoalStatus.ACHIEVED  # type: ignore[misc]

    def test_frozen_parent_id_raises(self) -> None:
        goal = Goal(name="test")
        with pytest.raises(dataclasses.FrozenInstanceError):
            goal.parent_id = "parent"  # type: ignore[misc]


class TestGoalLifecycleReturnsNew:

    def test_achieve_returns_new_goal(self) -> None:
        original = Goal(name="g1")
        achieved = original.achieve()
        assert achieved.status == GoalStatus.ACHIEVED
        assert original.status == GoalStatus.ACTIVE
        assert achieved is not original

    def test_abandon_returns_new_goal(self) -> None:
        original = Goal(name="g1")
        abandoned = original.abandon()
        assert abandoned.status == GoalStatus.ABANDONED
        assert original.status == GoalStatus.ACTIVE

    def test_suspend_returns_new_goal(self) -> None:
        original = Goal(name="g1")
        suspended = original.suspend()
        assert suspended.status == GoalStatus.SUSPENDED
        assert original.status == GoalStatus.ACTIVE

    def test_reactivate_returns_new_goal(self) -> None:
        suspended = Goal(name="g1", status=GoalStatus.SUSPENDED)
        reactivated = suspended.reactivate()
        assert reactivated.status == GoalStatus.ACTIVE
        assert suspended.status == GoalStatus.SUSPENDED

    def test_revise_does_not_mutate_original(self) -> None:
        goal = Goal(name="g1", description="original desc")
        new_goal, revision = goal.revise(new_description="revised desc", reason="test")
        assert goal.status == GoalStatus.ACTIVE
        assert goal.description == "original desc"
        assert new_goal.description == "revised desc"
        assert new_goal.status == GoalStatus.ACTIVE


class TestReflectNodeReturnsGoal:

    def test_reflect_returns_achieved_goal(self) -> None:
        """When score >= threshold, reflect_node returns the achieved goal."""
        goal = Goal(name="test")
        state = {
            "step": 5,
            "goal": goal,
            "eval_signal": EvalSignal(score=0.95, confidence=0.9),
            "max_steps": 100,
            "goal_achieved_threshold": 0.9,
            "filtered_policy": None,
        }
        result = reflect_node(state)
        assert result["stop_reason"] == "goal_achieved"
        assert result["goal"].status == GoalStatus.ACHIEVED

    def test_reflect_returns_unchanged_goal_on_continue(self) -> None:
        """When no stop condition, reflect_node returns the same goal."""
        goal = Goal(name="test")
        state = {
            "step": 5,
            "goal": goal,
            "eval_signal": EvalSignal(score=0.5, confidence=0.9),
            "max_steps": 100,
            "goal_achieved_threshold": 0.9,
            "filtered_policy": None,
        }
        result = reflect_node(state)
        assert result["goal"].status == GoalStatus.ACTIVE
        assert result.get("stop_reason") is None
