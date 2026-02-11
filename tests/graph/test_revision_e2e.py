"""Tests for v1.5.0 Phase 5: End-to-end revision path.

Verifies that a graph where the evaluator returns a bad score
triggers the revision path and updates goal_history.
"""

from __future__ import annotations

import time
from typing import Any

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import (
    ActionSpec,
    EvalSignal,
    PolicySpec,
    StateSnapshot,
)
from synthetic_teleology.graph.graph import build_teleological_graph
from synthetic_teleology.services.constraint_engine import ConstraintPipeline, PolicyFilter
from synthetic_teleology.services.evaluation import BaseEvaluator
from synthetic_teleology.services.goal_revision import BaseGoalUpdater
from synthetic_teleology.services.planning import BasePlanner


class _BadScoreEvaluator(BaseEvaluator):
    """Returns a bad score to trigger revision."""

    def __init__(self) -> None:
        self._call_count = 0

    def validate(self, goal: Goal, state: StateSnapshot) -> bool:
        return True

    def evaluate(self, goal: Goal, state: StateSnapshot) -> EvalSignal:
        self._call_count += 1
        # First eval: bad score to trigger revision
        # Second eval: good score to stop
        if self._call_count <= 1:
            return EvalSignal(score=-0.5, confidence=0.8, reasoning="Poor")
        return EvalSignal(score=0.95, confidence=0.9, reasoning="Achieved")


class _TestReviser(BaseGoalUpdater):
    """Always revises the goal."""

    def update(
        self, goal: Goal, state: StateSnapshot,
        signal: EvalSignal, constraints: Any = None,
    ) -> Goal | None:
        new_goal, _ = goal.revise(
            new_description="Revised: " + (goal.description or ""),
            reason="test revision",
            eval_signal=signal,
        )
        return new_goal


class _NoopPlanner(BasePlanner):
    def plan(self, goal: Goal, state: StateSnapshot) -> PolicySpec:
        return PolicySpec(
            actions=(ActionSpec(name="noop", parameters={"effect": (0.0,)}),)
        )


class TestRevisionE2E:

    def test_revision_path_populates_goal_history(self) -> None:
        """Graph executes revision path and populates goal_history."""
        evaluator = _BadScoreEvaluator()
        updater = _TestReviser()
        planner = _NoopPlanner()
        pipeline = ConstraintPipeline(checkers=[])
        policy_filter = PolicyFilter(pipeline)

        app = build_teleological_graph(
            evaluator=evaluator,
            goal_updater=updater,
            planner=planner,
            constraint_pipeline=pipeline,
            policy_filter=policy_filter,
        )

        goal = Goal(name="test", description="Original goal")

        def perceive() -> StateSnapshot:
            return StateSnapshot(timestamp=time.time(), values=(1.0,), observation="test")

        initial_state = {
            "step": 0,
            "max_steps": 5,
            "goal_achieved_threshold": 0.9,
            "goal": goal,
            "perceive_fn": perceive,
            "act_fn": None,
            "transition_fn": None,
            "events": [],
            "goal_history": [],
            "eval_history": [],
            "action_history": [],
            "reasoning_trace": [],
            "action_feedback": [],
        }

        result = app.invoke(initial_state)

        # Goal should have been revised
        assert len(result["goal_history"]) >= 1
        # The final goal should have "Revised" in description
        assert "Revised" in result["goal"].description
        # Reasoning trace should have entries
        assert len(result["reasoning_trace"]) >= 1
