"""Tests for SyncAgenticLoop."""

from __future__ import annotations

import time

import pytest

from synthetic_teleology.domain.enums import Direction, GoalStatus
from synthetic_teleology.domain.values import (
    ActionSpec,
    EvalSignal,
    ObjectiveVector,
    PolicySpec,
    StateSnapshot,
)
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.goal_revision import ThresholdUpdater
from synthetic_teleology.services.planning import GreedyPlanner
from synthetic_teleology.services.constraint_engine import (
    ConstraintPipeline,
    PolicyFilter,
)
from synthetic_teleology.services.loop import RunResult, StopReason, SyncAgenticLoop


def _build_loop(
    initial_state: tuple[float, ...] = (0.0, 0.0),
    goal_values: tuple[float, ...] = (5.0, 5.0),
    max_steps: int = 10,
    step_size: float = 1.0,
) -> tuple[SyncAgenticLoop, Goal, list]:
    """Build a minimal loop with a numeric environment simulation."""
    dims = len(initial_state)
    state_holder = {"values": list(initial_state)}

    def perceive_fn() -> StateSnapshot:
        return StateSnapshot(
            timestamp=time.time(),
            values=tuple(state_holder["values"]),
        )

    def act_fn(policy: PolicySpec, state: StateSnapshot) -> ActionSpec | None:
        if policy.size > 0:
            return policy.actions[0]
        return None

    def transition_fn(action: ActionSpec | None) -> None:
        if action is not None:
            delta = action.parameters.get("delta", action.parameters.get("effect"))
            if delta is not None:
                for i in range(len(state_holder["values"])):
                    if i < len(delta):
                        state_holder["values"][i] += delta[i]

    # Build action space
    actions = []
    for d in range(dims):
        effect_pos = tuple(step_size if i == d else 0.0 for i in range(dims))
        actions.append(ActionSpec(name=f"pos_{d}", parameters={"effect": effect_pos, "delta": effect_pos}))
        effect_neg = tuple(-step_size if i == d else 0.0 for i in range(dims))
        actions.append(ActionSpec(name=f"neg_{d}", parameters={"effect": effect_neg, "delta": effect_neg}))
    actions.append(ActionSpec(name="noop", parameters={"effect": tuple(0.0 for _ in range(dims)), "delta": tuple(0.0 for _ in range(dims))}))

    evaluator = NumericEvaluator(max_distance=10.0)
    updater = ThresholdUpdater(threshold=0.8, learning_rate=0.1)
    planner = GreedyPlanner(action_space=actions)
    pipeline = ConstraintPipeline(checkers=[])

    obj = ObjectiveVector(
        values=goal_values,
        directions=tuple(Direction.APPROACH for _ in goal_values),
    )
    goal = Goal(name="test-goal", objective=obj)

    loop = SyncAgenticLoop(
        evaluator=evaluator,
        goal_updater=updater,
        planner=planner,
        constraint_pipeline=pipeline,
        max_steps=max_steps,
        goal_achieved_threshold=0.9,
        stop_on_empty_policy=True,
        perceive_fn=perceive_fn,
        act_fn=act_fn,
        transition_fn=transition_fn,
    )

    return loop, goal, state_holder["values"]


class TestSyncAgenticLoop:
    """Test SyncAgenticLoop runs to completion, stops on goal achieved, handles errors."""

    def test_runs_to_max_steps(self) -> None:
        loop, goal, _ = _build_loop(max_steps=5)
        result = loop.run(goal)
        assert result.steps_completed == 5
        assert result.stopped_reason == StopReason.MAX_STEPS

    def test_stops_on_goal_achieved(self) -> None:
        # Start close to goal so it gets achieved quickly
        loop, goal, _ = _build_loop(
            initial_state=(4.5, 4.5),
            goal_values=(5.0, 5.0),
            max_steps=50,
            step_size=0.5,
        )
        result = loop.run(goal)
        assert result.stopped_reason == StopReason.GOAL_ACHIEVED
        assert result.steps_completed < 50

    def test_returns_run_result(self) -> None:
        loop, goal, _ = _build_loop(max_steps=3)
        result = loop.run(goal)
        assert isinstance(result, RunResult)
        assert result.final_goal is not None
        assert result.final_state is not None
        assert result.elapsed_seconds > 0
        assert len(result.events) > 0

    def test_events_emitted(self) -> None:
        loop, goal, _ = _build_loop(max_steps=3)
        result = loop.run(goal)
        assert len(result.events) >= 3

    def test_handles_perceive_error(self) -> None:
        def bad_perceive() -> StateSnapshot:
            raise RuntimeError("sensor failure")

        evaluator = NumericEvaluator()
        updater = ThresholdUpdater()
        actions = [ActionSpec(name="a", parameters={"effect": (0.0,)})]
        planner = GreedyPlanner(action_space=actions)
        pipeline = ConstraintPipeline(checkers=[])

        loop = SyncAgenticLoop(
            evaluator=evaluator,
            goal_updater=updater,
            planner=planner,
            constraint_pipeline=pipeline,
            max_steps=5,
            perceive_fn=bad_perceive,
        )

        obj = ObjectiveVector(values=(1.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)
        result = loop.run(goal)
        assert result.stopped_reason == StopReason.ERROR

    def test_no_perceive_fn_raises(self) -> None:
        evaluator = NumericEvaluator()
        updater = ThresholdUpdater()
        actions = [ActionSpec(name="a", parameters={"effect": (0.0,)})]
        planner = GreedyPlanner(action_space=actions)
        pipeline = ConstraintPipeline(checkers=[])

        loop = SyncAgenticLoop(
            evaluator=evaluator,
            goal_updater=updater,
            planner=planner,
            constraint_pipeline=pipeline,
            max_steps=5,
        )

        obj = ObjectiveVector(values=(1.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)
        result = loop.run(goal)
        # Should catch the RuntimeError and return ERROR
        assert result.stopped_reason == StopReason.ERROR

    def test_metadata_contains_eval_info(self) -> None:
        loop, goal, _ = _build_loop(max_steps=3)
        result = loop.run(goal)
        assert "last_eval_score" in result.metadata
        assert "last_eval_confidence" in result.metadata
