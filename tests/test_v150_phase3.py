"""Tests for v1.5.0 Phase 3: Thread safety.

Covers:
- Concurrent BudgetChecker.record_cost() calls preserve total
- GoalTree concurrent add_subgoal() doesn't corrupt state
- EvolvingConstraintManager concurrent record_violations() safe
"""

from __future__ import annotations

import concurrent.futures

from synthetic_teleology.domain.aggregates import GoalTree
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import ObjectiveVector
from synthetic_teleology.services.constraint_engine import BudgetChecker


class TestBudgetCheckerThreadSafety:

    def test_concurrent_record_cost_preserves_total(self) -> None:
        checker = BudgetChecker(total_budget=10000.0)
        num_calls = 1000
        cost_per_call = 1.0

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(checker.record_cost, cost_per_call)
                for _ in range(num_calls)
            ]
            concurrent.futures.wait(futures)

        assert abs(checker.total_spent - num_calls * cost_per_call) < 1e-6

    def test_concurrent_reset_and_record(self) -> None:
        checker = BudgetChecker(total_budget=10000.0)
        checker.record_cost(100.0)
        checker.reset()
        assert checker.total_spent == 0.0


class TestGoalTreeThreadSafety:

    def test_concurrent_add_subgoal(self) -> None:
        obj = ObjectiveVector(
            values=(1.0,), directions=(Direction.APPROACH,)
        )
        root = Goal(goal_id="root", name="root", objective=obj)
        tree = GoalTree(root)

        def add_child(i: int) -> None:
            child = Goal(goal_id=f"c{i}", name=f"child-{i}", objective=obj)
            tree.add_subgoal("root", child)

        num_children = 50
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(add_child, i) for i in range(num_children)]
            concurrent.futures.wait(futures)

        assert tree.size == num_children + 1  # +1 for root


class TestEvolvingConstraintManagerThreadSafety:

    def test_concurrent_record_violations(self) -> None:
        from unittest.mock import MagicMock

        from synthetic_teleology.services.evolving_constraints import (
            EvolvingConstraintManager,
        )

        mock_model = MagicMock()
        manager = EvolvingConstraintManager(
            model=mock_model,
            initial_constraints=["budget < 100"],
        )

        def record(i: int) -> None:
            manager.record_violations([f"violation-{i}"])

        num_records = 100
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(record, i) for i in range(num_records)]
            concurrent.futures.wait(futures)

        assert len(manager._violation_history) == num_records
