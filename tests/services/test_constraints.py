"""Tests for constraint engine services."""

from __future__ import annotations

import time

import pytest

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import (
    ActionSpec,
    ObjectiveVector,
    PolicySpec,
    StateSnapshot,
)
from synthetic_teleology.services.constraint_engine import (
    BudgetChecker,
    ConstraintPipeline,
    PolicyFilter,
    SafetyChecker,
)


@pytest.fixture
def goal_2d() -> Goal:
    obj = ObjectiveVector(
        values=(5.0, 5.0),
        directions=(Direction.APPROACH, Direction.APPROACH),
    )
    return Goal(name="g", objective=obj)


@pytest.fixture
def safe_state() -> StateSnapshot:
    return StateSnapshot(timestamp=time.time(), values=(5.0, 5.0))


@pytest.fixture
def unsafe_state() -> StateSnapshot:
    return StateSnapshot(timestamp=time.time(), values=(-1.0, 5.0))


class TestSafetyChecker:
    """Test SafetyChecker with state bounds."""

    def test_passes_within_bounds(self, goal_2d: Goal, safe_state: StateSnapshot) -> None:
        checker = SafetyChecker(
            lower_bounds=[0.0, 0.0],
            upper_bounds=[10.0, 10.0],
        )
        passed, msg = checker.check(goal_2d, safe_state)
        assert passed is True
        assert msg == ""

    def test_fails_below_lower_bound(self, goal_2d: Goal, unsafe_state: StateSnapshot) -> None:
        checker = SafetyChecker(
            lower_bounds=[0.0, 0.0],
            upper_bounds=[10.0, 10.0],
        )
        passed, msg = checker.check(goal_2d, unsafe_state)
        assert passed is False
        assert "lower_bound" in msg

    def test_fails_above_upper_bound(self, goal_2d: Goal) -> None:
        state = StateSnapshot(timestamp=time.time(), values=(5.0, 15.0))
        checker = SafetyChecker(
            lower_bounds=[0.0, 0.0],
            upper_bounds=[10.0, 10.0],
        )
        passed, msg = checker.check(goal_2d, state)
        assert passed is False
        assert "upper_bound" in msg

    def test_checks_action_effects(self, goal_2d: Goal, safe_state: StateSnapshot) -> None:
        checker = SafetyChecker(
            lower_bounds=[0.0, 0.0],
            upper_bounds=[10.0, 10.0],
        )
        # Action that would push state out of bounds
        action = ActionSpec(name="bad", parameters={"effect": (6.0, 0.0)})
        passed, msg = checker.check(goal_2d, safe_state, action)
        assert passed is False
        assert "projected" in msg

    def test_no_bounds_always_passes(self, goal_2d: Goal, safe_state: StateSnapshot) -> None:
        checker = SafetyChecker()
        passed, msg = checker.check(goal_2d, safe_state)
        assert passed is True


class TestBudgetChecker:
    """Test BudgetChecker with cost budget."""

    def test_passes_within_budget(self, goal_2d: Goal, safe_state: StateSnapshot) -> None:
        checker = BudgetChecker(total_budget=100.0)
        action = ActionSpec(name="a", cost=10.0)
        passed, msg = checker.check(goal_2d, safe_state, action)
        assert passed is True

    def test_fails_over_budget(self, goal_2d: Goal, safe_state: StateSnapshot) -> None:
        checker = BudgetChecker(total_budget=5.0)
        action = ActionSpec(name="a", cost=10.0)
        passed, msg = checker.check(goal_2d, safe_state, action)
        assert passed is False
        assert "exceed budget" in msg

    def test_tracks_spending(self, goal_2d: Goal, safe_state: StateSnapshot) -> None:
        checker = BudgetChecker(total_budget=20.0)
        checker.record_cost(15.0)
        assert checker.total_spent == 15.0
        assert checker.budget_remaining == 5.0

        action = ActionSpec(name="a", cost=10.0)
        passed, _ = checker.check(goal_2d, safe_state, action)
        assert passed is False

    def test_reset(self) -> None:
        checker = BudgetChecker(total_budget=100.0)
        checker.record_cost(50.0)
        checker.reset()
        assert checker.total_spent == 0.0
        assert checker.budget_remaining == 100.0

    def test_no_action_just_checks_exhaustion(
        self, goal_2d: Goal, safe_state: StateSnapshot
    ) -> None:
        checker = BudgetChecker(total_budget=10.0)
        checker.record_cost(10.0)
        passed, msg = checker.check(goal_2d, safe_state)
        assert passed is False
        assert "exhausted" in msg

    def test_negative_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            BudgetChecker(total_budget=-1.0)


class TestConstraintPipeline:
    """Test ConstraintPipeline: chain of responsibility."""

    def test_all_pass(self, goal_2d: Goal, safe_state: StateSnapshot) -> None:
        checker1 = SafetyChecker(lower_bounds=[0.0, 0.0], upper_bounds=[10.0, 10.0])
        checker2 = BudgetChecker(total_budget=100.0)
        pipeline = ConstraintPipeline(checkers=[checker1, checker2])

        passed, violations = pipeline.check_all(goal_2d, safe_state)
        assert passed is True
        assert violations == []

    def test_collects_violations(self, goal_2d: Goal, unsafe_state: StateSnapshot) -> None:
        checker1 = SafetyChecker(lower_bounds=[0.0, 0.0], upper_bounds=[10.0, 10.0])
        budget = BudgetChecker(total_budget=0.0)
        budget.record_cost(1.0)
        pipeline = ConstraintPipeline(checkers=[checker1, budget])

        passed, violations = pipeline.check_all(goal_2d, unsafe_state)
        assert passed is False
        assert len(violations) == 2

    def test_fail_fast(self, goal_2d: Goal, unsafe_state: StateSnapshot) -> None:
        checker1 = SafetyChecker(lower_bounds=[0.0, 0.0], upper_bounds=[10.0, 10.0])
        budget = BudgetChecker(total_budget=0.0)
        budget.record_cost(1.0)
        pipeline = ConstraintPipeline(
            checkers=[checker1, budget], fail_fast=True
        )

        passed, violations = pipeline.check_all(goal_2d, unsafe_state)
        assert passed is False
        assert len(violations) == 1  # stopped at first

    def test_check_convenience_method(self, goal_2d: Goal, safe_state: StateSnapshot) -> None:
        pipeline = ConstraintPipeline(checkers=[])
        passed, msg = pipeline.check(goal_2d, safe_state)
        assert passed is True
        assert msg == ""


class TestPolicyFilter:
    """Test PolicyFilter filters unsafe actions from PolicySpec."""

    def test_filters_unsafe_actions(self, goal_2d: Goal, safe_state: StateSnapshot) -> None:
        checker = SafetyChecker(
            lower_bounds=[0.0, 0.0], upper_bounds=[10.0, 10.0]
        )
        pipeline = ConstraintPipeline(checkers=[checker])
        pf = PolicyFilter(pipeline)

        safe_action = ActionSpec(name="safe", parameters={"effect": (1.0, 0.0)})
        unsafe_action = ActionSpec(name="unsafe", parameters={"effect": (6.0, 0.0)})
        policy = PolicySpec(actions=(safe_action, unsafe_action))

        filtered = pf.filter(policy, goal_2d, safe_state)
        assert filtered.size == 1
        assert filtered.actions[0].name == "safe"

    def test_all_filtered_returns_empty(self, goal_2d: Goal, safe_state: StateSnapshot) -> None:
        checker = SafetyChecker(
            lower_bounds=[0.0, 0.0], upper_bounds=[10.0, 10.0]
        )
        pipeline = ConstraintPipeline(checkers=[checker])
        pf = PolicyFilter(pipeline)

        unsafe = ActionSpec(name="unsafe", parameters={"effect": (20.0, 20.0)})
        policy = PolicySpec(actions=(unsafe,))

        filtered = pf.filter(policy, goal_2d, safe_state)
        assert filtered.size == 0

    def test_stochastic_renormalization(self, goal_2d: Goal, safe_state: StateSnapshot) -> None:
        pipeline = ConstraintPipeline(checkers=[])  # all pass
        pf = PolicyFilter(pipeline)

        a1 = ActionSpec(name="a1", parameters={})
        a2 = ActionSpec(name="a2", parameters={})
        policy = PolicySpec(actions=(a1, a2), probabilities=(0.3, 0.7))

        filtered = pf.filter(policy, goal_2d, safe_state)
        assert filtered.is_stochastic is True
        assert filtered.probabilities is not None
        assert sum(filtered.probabilities) == pytest.approx(1.0)

    def test_no_actions_policy(self, goal_2d: Goal, safe_state: StateSnapshot) -> None:
        pipeline = ConstraintPipeline(checkers=[])
        pf = PolicyFilter(pipeline)
        policy = PolicySpec()
        filtered = pf.filter(policy, goal_2d, safe_state)
        assert filtered.size == 0
