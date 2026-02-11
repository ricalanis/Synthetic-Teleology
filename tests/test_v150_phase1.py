"""Tests for v1.5.0 Phase 1 quick wins.

Covers:
- GoalTree propagate_revision removes old goals from _all_goals
- _softmax with temperature=0 raises ValueError
- LLMPlanner rejects non-positive temperature
- LLMConstraintChecker falls back when individual violations are empty
"""

from __future__ import annotations

import pytest

from synthetic_teleology.domain.aggregates import GoalTree
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import ObjectiveVector
from synthetic_teleology.services.llm_planning import _softmax


class TestGoalTreeMemoryLeak:

    def test_propagate_revision_removes_old_child(self) -> None:
        """After propagate_revision, the old child goal_id is not in _all_goals."""
        root = Goal(
            name="root",
            objective=ObjectiveVector(
                values=(1.0, 2.0),
                directions=(Direction.APPROACH, Direction.APPROACH),
            ),
        )
        child = Goal(
            name="child",
            objective=ObjectiveVector(
                values=(0.5, 1.0),
                directions=(Direction.APPROACH, Direction.APPROACH),
            ),
        )
        tree = GoalTree(root)
        tree.add_subgoal(root.goal_id, child)

        old_child_id = child.goal_id
        assert old_child_id in tree._all_goals

        # propagate_revision is called with the parent whose children
        # should be revised — the root itself (its children list)
        revisions = tree.propagate_revision(root)
        assert len(revisions) == 1

        # Old child should be removed from _all_goals
        assert old_child_id not in tree._all_goals

    def test_propagate_revision_adds_new_child(self) -> None:
        root = Goal(
            name="root",
            objective=ObjectiveVector(
                values=(1.0,), directions=(Direction.APPROACH,),
            ),
        )
        child = Goal(
            name="child",
            objective=ObjectiveVector(
                values=(0.5,), directions=(Direction.APPROACH,),
            ),
        )
        tree = GoalTree(root)
        tree.add_subgoal(root.goal_id, child)

        revisions = tree.propagate_revision(root)
        # New child should be in _all_goals
        new_child_id = revisions[0].new_goal_id
        assert new_child_id in tree._all_goals


class TestSoftmaxValidation:

    def test_softmax_zero_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be positive"):
            _softmax([0.5, 0.3], temperature=0.0)

    def test_softmax_negative_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be positive"):
            _softmax([0.5, 0.3], temperature=-1.0)

    def test_softmax_empty_values(self) -> None:
        assert _softmax([]) == []

    def test_softmax_normal(self) -> None:
        result = _softmax([1.0, 1.0], temperature=1.0)
        assert len(result) == 2
        assert abs(result[0] - 0.5) < 1e-6
        assert abs(result[1] - 0.5) < 1e-6


class TestLLMPlannerTemperatureValidation:

    def test_reject_zero_temperature(self) -> None:
        from tests.helpers.mock_llm import MockStructuredChatModel

        mock = MockStructuredChatModel(responses=["test"])
        with pytest.raises(ValueError, match="temperature must be positive"):
            from synthetic_teleology.services.llm_planning import LLMPlanner
            LLMPlanner(model=mock, temperature=0.0)


class TestLLMConstraintCheckerEmptyViolations:

    def test_fallback_on_empty_individual_violations(self) -> None:
        """When overall_safe=False but no individual assessments fail,
        the checker should still produce a meaningful violation message."""
        from unittest.mock import MagicMock

        from synthetic_teleology.services.llm_constraints import (
            ConstraintAssessment,
            ConstraintCheckOutput,
            LLMConstraintChecker,
        )

        # Build a mock model that returns overall_safe=False but all
        # individual assessments satisfied
        mock_result = ConstraintCheckOutput(
            assessments=[
                ConstraintAssessment(
                    constraint_name="budget",
                    is_satisfied=True,
                    severity=0.0,
                    reasoning="Within budget",
                ),
            ],
            overall_safe=False,
            overall_reasoning="Something else is wrong",
        )

        mock_model = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result
        mock_model.with_structured_output.return_value = mock_chain

        checker = LLMConstraintChecker(
            model=mock_model, constraints=["Stay within budget"]
        )
        # Replace internal chain with mock
        checker._chain = mock_chain

        from synthetic_teleology.domain.entities import Goal
        from synthetic_teleology.domain.values import StateSnapshot

        goal = Goal(description="test")
        state = StateSnapshot(timestamp=0.0, observation="test")

        ok, msg = checker.check(goal, state)
        assert not ok
        assert "Overall assessment" in msg
        assert "Something else is wrong" in msg
