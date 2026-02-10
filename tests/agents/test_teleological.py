"""Tests for TeleologicalAgent."""

from __future__ import annotations

import time

import pytest

from synthetic_teleology.agents.teleological import TeleologicalAgent
from synthetic_teleology.domain.aggregates import ConstraintSet
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import AgentState, Direction
from synthetic_teleology.domain.values import (
    ActionSpec,
    EvalSignal,
    ObjectiveVector,
    PolicySpec,
    StateSnapshot,
)
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.infrastructure.event_bus import EventBus
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.goal_revision import ThresholdUpdater
from synthetic_teleology.services.planning import GreedyPlanner


def _make_agent(
    target: tuple[float, ...] = (5.0, 5.0),
    threshold: float = 0.5,
    step_size: float = 0.5,
) -> tuple[TeleologicalAgent, EventBus]:
    """Build a minimal TeleologicalAgent for testing."""
    bus = EventBus()
    dims = len(target)
    obj = ObjectiveVector(
        values=target,
        directions=tuple(Direction.APPROACH for _ in target),
    )
    goal = Goal(name="test-goal", objective=obj)
    evaluator = NumericEvaluator(max_distance=10.0)
    updater = ThresholdUpdater(threshold=threshold, learning_rate=0.1)
    actions: list[ActionSpec] = []
    for d in range(dims):
        eff_pos = tuple(step_size if i == d else 0.0 for i in range(dims))
        actions.append(ActionSpec(name=f"pos_{d}", parameters={"effect": eff_pos, "delta": eff_pos}))
        eff_neg = tuple(-step_size if i == d else 0.0 for i in range(dims))
        actions.append(ActionSpec(name=f"neg_{d}", parameters={"effect": eff_neg, "delta": eff_neg}))
    actions.append(ActionSpec(name="noop", parameters={"effect": tuple(0.0 for _ in range(dims)), "delta": tuple(0.0 for _ in range(dims))}))
    planner = GreedyPlanner(action_space=actions)

    agent = TeleologicalAgent(
        agent_id="agent-test",
        initial_goal=goal,
        event_bus=bus,
        evaluator=evaluator,
        updater=updater,
        planner=planner,
    )
    return agent, bus


class TestTeleologicalAgent:
    """Test TeleologicalAgent state transitions, lifecycle, and delegation."""

    def test_initial_state_is_idle(self) -> None:
        agent, _ = _make_agent()
        assert agent.state == AgentState.IDLE
        assert agent.step_count == 0

    def test_agent_id(self) -> None:
        agent, _ = _make_agent()
        assert agent.id == "agent-test"

    def test_current_goal(self) -> None:
        agent, _ = _make_agent()
        assert agent.current_goal is not None
        assert agent.current_goal.name == "test-goal"

    def test_strategy_accessors(self) -> None:
        agent, _ = _make_agent()
        assert isinstance(agent.evaluator, NumericEvaluator)
        assert isinstance(agent.updater, ThresholdUpdater)
        assert isinstance(agent.planner, GreedyPlanner)
        assert isinstance(agent.constraints, ConstraintSet)

    def test_perceive_transitions_to_evaluating(self) -> None:
        agent, _ = _make_agent()
        env = NumericEnvironment(dimensions=2, initial_state=(3.0, 4.0))
        state = agent.perceive(env)
        assert agent.state == AgentState.EVALUATING
        assert state is not None
        assert len(state.values) == 2

    def test_evaluate_returns_eval_signal(self) -> None:
        agent, _ = _make_agent()
        env = NumericEnvironment(dimensions=2, initial_state=(3.0, 4.0))
        state = agent.perceive(env)
        goal = agent.current_goal
        signal = agent.evaluate(goal, state)
        assert isinstance(signal, EvalSignal)
        assert -1.0 <= signal.score <= 1.0
        assert agent.last_eval is signal

    def test_revise_goal_transitions(self) -> None:
        agent, _ = _make_agent()
        env = NumericEnvironment(dimensions=2, initial_state=(3.0, 4.0))
        state = agent.perceive(env)
        goal = agent.current_goal
        signal = agent.evaluate(goal, state)
        # Revise goal transitions through REVISING -> PLANNING
        result = agent.revise_goal(goal, state, signal, agent.constraints)
        assert agent.state == AgentState.PLANNING

    def test_plan_returns_policy(self) -> None:
        agent, _ = _make_agent()
        env = NumericEnvironment(dimensions=2, initial_state=(3.0, 4.0))
        state = agent.perceive(env)
        goal = agent.current_goal
        signal = agent.evaluate(goal, state)
        agent.revise_goal(goal, state, signal, agent.constraints)
        policy = agent.plan(goal, state)
        assert isinstance(policy, PolicySpec)
        assert policy.size >= 1
        assert agent.last_policy is policy

    def test_select_action_from_deterministic_policy(self) -> None:
        agent, _ = _make_agent()
        env = NumericEnvironment(dimensions=2, initial_state=(3.0, 4.0))
        state = agent.perceive(env)
        goal = agent.current_goal
        signal = agent.evaluate(goal, state)
        agent.revise_goal(goal, state, signal, agent.constraints)
        policy = agent.plan(goal, state)
        action = agent.select_action(policy)
        assert isinstance(action, ActionSpec)
        assert agent.state == AgentState.ACTING

    def test_select_action_empty_policy_returns_noop(self) -> None:
        agent, _ = _make_agent()
        env = NumericEnvironment(dimensions=2, initial_state=(3.0, 4.0))
        state = agent.perceive(env)
        goal = agent.current_goal
        signal = agent.evaluate(goal, state)
        agent.revise_goal(goal, state, signal, agent.constraints)
        _ = agent.plan(goal, state)
        empty_policy = PolicySpec()
        action = agent.select_action(empty_policy)
        assert action.name == "noop"

    def test_run_cycle_full_loop(self) -> None:
        agent, bus = _make_agent()
        env = NumericEnvironment(dimensions=2, initial_state=(3.0, 4.0))
        events_received: list = []
        bus.subscribe_all(lambda e: events_received.append(e))

        action = agent.run_cycle(env)
        assert isinstance(action, ActionSpec)
        assert agent.state == AgentState.IDLE
        assert agent.step_count == 1
        assert len(events_received) > 0

    def test_run_cycle_increments_step_count(self) -> None:
        agent, _ = _make_agent()
        env = NumericEnvironment(dimensions=2, initial_state=(3.0, 4.0))

        agent.run_cycle(env)
        assert agent.step_count == 1
        agent.run_cycle(env)
        assert agent.step_count == 2

    def test_last_snapshot_updated_after_perceive(self) -> None:
        agent, _ = _make_agent()
        assert agent.last_snapshot is None
        env = NumericEnvironment(dimensions=2, initial_state=(1.0, 2.0))
        agent.perceive(env)
        assert agent.last_snapshot is not None

    def test_events_emitted_during_cycle(self) -> None:
        agent, bus = _make_agent()
        env = NumericEnvironment(dimensions=2, initial_state=(3.0, 4.0))
        collected: list = []
        bus.subscribe_all(lambda e: collected.append(e))

        agent.run_cycle(env)
        # Should have at least: StateChanged, EvaluationCompleted, PlanGenerated, ActionExecuted, LoopStepCompleted
        assert len(collected) >= 4

    def test_repr(self) -> None:
        agent, _ = _make_agent()
        r = repr(agent)
        assert "TeleologicalAgent" in r
        assert "agent-test" in r

    def test_invalid_state_transition_raises(self) -> None:
        agent, _ = _make_agent()
        # Try jumping from IDLE to ACTING (invalid)
        with pytest.raises(ValueError, match="Invalid state transition"):
            agent._transition_to(AgentState.ACTING)
