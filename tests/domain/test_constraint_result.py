"""Tests for ConstraintResult, check_detailed, and ActionSpec extensions."""

from __future__ import annotations

import pytest

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import ActionSpec, ConstraintResult, StateSnapshot
from synthetic_teleology.services.constraint_engine import (
    ConstraintPipeline,
    SafetyChecker,
)


def _make_goal() -> Goal:
    return Goal(name="test")


def _make_snapshot(values: tuple[float, ...] = (1.0, 2.0)) -> StateSnapshot:
    return StateSnapshot(timestamp=0.0, values=values)


class TestConstraintResultFrozen:

    def test_constraint_result_frozen(self) -> None:
        """ConstraintResult is immutable (frozen dataclass)."""
        result = ConstraintResult(passed=True, message="ok", severity=0.0)
        with pytest.raises(AttributeError):
            result.passed = False  # type: ignore[misc]

    def test_constraint_result_defaults(self) -> None:
        result = ConstraintResult(passed=False)
        assert result.message == ""
        assert result.severity == 0.0
        assert result.checker_name == ""
        assert result.suggested_mitigation == ""
        assert dict(result.metadata) == {}


class TestCheckDetailedWrapsCheck:

    def test_check_detailed_wraps_check(self) -> None:
        """check_detailed() default implementation wraps check()."""
        checker = SafetyChecker(
            lower_bounds=[0.0, 0.0],
            upper_bounds=[10.0, 10.0],
        )
        result = checker.check_detailed(_make_goal(), _make_snapshot())
        assert result.passed is True
        assert result.checker_name == "SafetyChecker"

    def test_check_detailed_reports_violation(self) -> None:
        """check_detailed reports violations from check()."""
        checker = SafetyChecker(
            upper_bounds=[0.5, 0.5],
            name="BoundsChecker",
        )
        result = checker.check_detailed(_make_goal(), _make_snapshot((1.0, 2.0)))
        assert result.passed is False
        assert "BoundsChecker" in result.message


class TestPipelineCheckAllDetailed:

    def test_pipeline_check_all_detailed(self) -> None:
        """Pipeline returns list of ConstraintResult objects."""
        checker1 = SafetyChecker(upper_bounds=[10.0, 10.0], name="Safe")
        checker2 = SafetyChecker(upper_bounds=[0.5, 0.5], name="Strict")
        pipeline = ConstraintPipeline(checkers=[checker1, checker2])

        results = pipeline.check_all_detailed(_make_goal(), _make_snapshot())

        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is False


class TestActionSpecExtensions:

    def test_action_spec_with_effect_field(self) -> None:
        """ActionSpec accepts effect and preconditions fields."""
        action = ActionSpec(
            name="move",
            effect=(1.0, 0.0),
            preconditions={"energy": 10},
        )
        assert action.effect == (1.0, 0.0)
        assert action.preconditions["energy"] == 10

    def test_action_spec_backward_compatible(self) -> None:
        """ActionSpec without new fields still works (defaults)."""
        action = ActionSpec(name="old_action")
        assert action.effect is None
        assert dict(action.preconditions) == {}
