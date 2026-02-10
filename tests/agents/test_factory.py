"""Tests for AgentFactory and AgentBuilder."""

from __future__ import annotations

import time

import pytest

from synthetic_teleology.agents.factory import AgentBuilder, AgentFactory
from synthetic_teleology.agents.teleological import TeleologicalAgent
from synthetic_teleology.domain.aggregates import ConstraintSet
from synthetic_teleology.domain.entities import Constraint, Goal
from synthetic_teleology.domain.enums import AgentState, ConstraintType, Direction
from synthetic_teleology.domain.values import (
    ActionSpec,
    ConstraintSpec,
    ObjectiveVector,
)
from synthetic_teleology.infrastructure.event_bus import EventBus
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.goal_revision import ThresholdUpdater
from synthetic_teleology.services.planning import GreedyPlanner


class TestAgentFactory:
    """Test AgentFactory static factory methods."""

    def test_create_simple_agent(self) -> None:
        agent = AgentFactory.create_simple_agent(
            agent_id="simple-1",
            target_values=(5.0, 5.0),
        )
        assert isinstance(agent, TeleologicalAgent)
        assert agent.id == "simple-1"
        assert agent.state == AgentState.IDLE
        assert agent.current_goal is not None
        assert agent.current_goal.objective is not None
        assert agent.current_goal.objective.values == (5.0, 5.0)

    def test_create_simple_agent_with_custom_directions(self) -> None:
        agent = AgentFactory.create_simple_agent(
            agent_id="simple-2",
            target_values=(10.0,),
            directions=(Direction.MAXIMIZE,),
        )
        assert agent.current_goal.objective.directions == (Direction.MAXIMIZE,)

    def test_create_simple_agent_with_event_bus(self) -> None:
        bus = EventBus()
        agent = AgentFactory.create_simple_agent(
            agent_id="simple-3",
            target_values=(1.0, 2.0),
            event_bus=bus,
        )
        assert agent.event_bus is bus

    def test_create_teleological_agent(self) -> None:
        bus = EventBus()
        obj = ObjectiveVector(values=(5.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)
        evaluator = NumericEvaluator()
        updater = ThresholdUpdater()
        actions = [ActionSpec(name="a", parameters={"effect": (1.0,)})]
        planner = GreedyPlanner(action_space=actions)

        agent = AgentFactory.create_teleological_agent(
            agent_id="tele-1",
            goal=goal,
            event_bus=bus,
            evaluator=evaluator,
            updater=updater,
            planner=planner,
        )
        assert isinstance(agent, TeleologicalAgent)
        assert agent.id == "tele-1"
        assert agent.evaluator is evaluator
        assert agent.updater is updater
        assert agent.planner is planner

    def test_create_constrained_agent(self) -> None:
        bus = EventBus()
        obj = ObjectiveVector(values=(5.0, 5.0), directions=(Direction.APPROACH, Direction.APPROACH))
        goal = Goal(name="g", objective=obj)
        evaluator = NumericEvaluator()
        updater = ThresholdUpdater()
        actions = [ActionSpec(name="a", parameters={"effect": (1.0, 0.0)})]
        planner = GreedyPlanner(action_space=actions)

        spec = ConstraintSpec(
            name="safety",
            constraint_type=ConstraintType.HARD,
            parameters={},
        )
        constraint = Constraint(
            name="safety",
            constraint_type=ConstraintType.HARD,
            spec=spec,
        )

        agent = AgentFactory.create_constrained_agent(
            agent_id="const-1",
            goal=goal,
            event_bus=bus,
            evaluator=evaluator,
            updater=updater,
            planner=planner,
            constraints=[constraint],
        )
        assert isinstance(agent, TeleologicalAgent)
        assert len(agent.constraints) == 1


class TestAgentBuilder:
    """Test AgentBuilder fluent construction pattern."""

    def test_build_with_goal(self) -> None:
        obj = ObjectiveVector(values=(5.0, 5.0), directions=(Direction.APPROACH, Direction.APPROACH))
        goal = Goal(name="g", objective=obj)

        agent = (
            AgentBuilder("builder-1")
            .with_goal(goal)
            .build()
        )
        assert isinstance(agent, TeleologicalAgent)
        assert agent.id == "builder-1"
        assert agent.current_goal.name == "g"

    def test_build_with_objective(self) -> None:
        agent = (
            AgentBuilder("builder-2")
            .with_objective(values=(3.0, 4.0))
            .build()
        )
        assert agent.current_goal.objective is not None
        assert agent.current_goal.objective.values == (3.0, 4.0)

    def test_build_with_custom_evaluator(self) -> None:
        evaluator = NumericEvaluator(max_distance=20.0)
        agent = (
            AgentBuilder("builder-3")
            .with_objective(values=(5.0,))
            .with_evaluator(evaluator)
            .build()
        )
        assert agent.evaluator is evaluator

    def test_build_with_custom_updater(self) -> None:
        updater = ThresholdUpdater(threshold=0.3, learning_rate=0.2)
        agent = (
            AgentBuilder("builder-4")
            .with_objective(values=(5.0,))
            .with_goal_updater(updater)
            .build()
        )
        assert agent.updater is updater

    def test_build_with_custom_planner(self) -> None:
        actions = [ActionSpec(name="a", parameters={"effect": (1.0,)})]
        planner = GreedyPlanner(action_space=actions)
        agent = (
            AgentBuilder("builder-5")
            .with_objective(values=(5.0,))
            .with_planner(planner)
            .build()
        )
        assert agent.planner is planner

    def test_build_with_event_bus(self) -> None:
        bus = EventBus()
        agent = (
            AgentBuilder("builder-6")
            .with_objective(values=(5.0,))
            .with_event_bus(bus)
            .build()
        )
        assert agent.event_bus is bus

    def test_build_with_constraints(self) -> None:
        spec = ConstraintSpec(
            name="c1",
            constraint_type=ConstraintType.SOFT,
            parameters={},
        )
        c = Constraint(name="c1", constraint_type=ConstraintType.SOFT, spec=spec)

        agent = (
            AgentBuilder("builder-7")
            .with_objective(values=(5.0,))
            .with_constraints(c)
            .build()
        )
        assert len(agent.constraints) == 1

    def test_build_without_goal_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a goal"):
            AgentBuilder("builder-8").build()

    def test_defaults_filled_in(self) -> None:
        agent = (
            AgentBuilder("builder-9")
            .with_objective(values=(1.0, 2.0))
            .build()
        )
        # Should have default evaluator, updater, planner
        assert agent.evaluator is not None
        assert agent.updater is not None
        assert agent.planner is not None

    def test_chaining_returns_self(self) -> None:
        builder = AgentBuilder("builder-10")
        result = builder.with_objective(values=(1.0,))
        assert result is builder

    def test_repr(self) -> None:
        builder = AgentBuilder("builder-11").with_objective(values=(1.0,))
        r = repr(builder)
        assert "AgentBuilder" in r
        assert "builder-11" in r

    def test_with_action_step_size(self) -> None:
        agent = (
            AgentBuilder("builder-12")
            .with_objective(values=(5.0, 5.0))
            .with_action_step_size(1.0)
            .build()
        )
        assert agent is not None
        # Planner should have been built with step_size=1.0
        assert len(agent.planner.action_space) == 5  # 2*2 + 1 = 5
