"""Tests for GraphBuilder fluent API."""

from __future__ import annotations

import pytest

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import ObjectiveVector
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph.builder import GraphBuilder
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.goal_revision import ThresholdUpdater


class TestGraphBuilder:

    def test_build_minimal(self) -> None:
        env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
        app, state = (
            GraphBuilder("test-agent")
            .with_objective((5.0, 5.0))
            .with_environment(
                perceive_fn=lambda: env.observe(),
                transition_fn=lambda a: env.step(a) if a else None,
            )
            .build()
        )
        assert app is not None
        assert state["goal"].name == "test-agent-goal"
        assert state["step"] == 0
        assert state["max_steps"] == 100

    def test_build_with_custom_components(self) -> None:
        env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
        app, state = (
            GraphBuilder("custom-agent")
            .with_objective((3.0, 7.0))
            .with_evaluator(NumericEvaluator(max_distance=15.0))
            .with_goal_updater(ThresholdUpdater(threshold=0.6))
            .with_max_steps(50)
            .with_goal_achieved_threshold(0.85)
            .with_environment(perceive_fn=lambda: env.observe())
            .build()
        )
        assert state["max_steps"] == 50
        assert state["goal_achieved_threshold"] == 0.85

    def test_build_raises_without_goal(self) -> None:
        env = NumericEnvironment(dimensions=2)
        with pytest.raises(ValueError, match="requires a goal"):
            GraphBuilder("no-goal").with_environment(perceive_fn=lambda: env.observe()).build()

    def test_build_raises_without_perceive_fn(self) -> None:
        with pytest.raises(RuntimeError, match="requires a perceive_fn"):
            GraphBuilder("no-env").with_objective((1.0,)).build()

    def test_invoke_produces_result(self) -> None:
        env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
        app, state = (
            GraphBuilder("invoke-agent")
            .with_objective((2.0, 2.0))
            .with_max_steps(5)
            .with_environment(
                perceive_fn=lambda: env.observe(),
                act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
                transition_fn=lambda a: env.step(a) if a else None,
            )
            .build()
        )
        result = app.invoke(state)
        assert result["step"] == 5
        assert len(result["events"]) > 0

    def test_with_goal_entity(self) -> None:
        env = NumericEnvironment(dimensions=2)
        goal = Goal(name="explicit-goal", objective=ObjectiveVector(
            values=(1.0, 1.0), directions=(Direction.APPROACH, Direction.APPROACH)
        ))
        app, state = (
            GraphBuilder("entity-agent")
            .with_goal(goal)
            .with_environment(perceive_fn=lambda: env.observe())
            .build()
        )
        assert state["goal"].name == "explicit-goal"

    def test_with_checkpointer(self) -> None:
        from langgraph.checkpoint.memory import MemorySaver
        env = NumericEnvironment(dimensions=2)
        app, state = (
            GraphBuilder("cp-agent")
            .with_objective((1.0,))
            .with_environment(perceive_fn=lambda: env.observe())
            .with_checkpointer(MemorySaver())
            .build()
        )
        assert app is not None

    def test_with_metadata(self) -> None:
        env = NumericEnvironment(dimensions=2)
        app, state = (
            GraphBuilder("meta-agent")
            .with_objective((1.0,))
            .with_environment(perceive_fn=lambda: env.observe())
            .with_metadata(experiment="test-run")
            .build()
        )
        assert state["metadata"]["experiment"] == "test-run"

    def test_repr(self) -> None:
        builder = GraphBuilder("repr-agent").with_objective((1.0, 2.0))
        r = repr(builder)
        assert "repr-agent" in r

    def test_fluent_chaining(self) -> None:
        """All methods return self for chaining."""
        env = NumericEnvironment(dimensions=2)
        builder = GraphBuilder("chain-agent")
        result = (
            builder
            .with_objective((1.0, 2.0))
            .with_evaluator(NumericEvaluator())
            .with_goal_updater(ThresholdUpdater())
            .with_max_steps(10)
            .with_goal_achieved_threshold(0.8)
            .with_action_step_size(0.3)
            .with_environment(perceive_fn=lambda: env.observe())
        )
        assert result is builder
