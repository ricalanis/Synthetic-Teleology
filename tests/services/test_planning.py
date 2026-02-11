"""Tests for planning services."""

from __future__ import annotations

import time

import pytest

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import (
    ActionSpec,
    ObjectiveVector,
    StateSnapshot,
)
from synthetic_teleology.services.planning import (
    GreedyPlanner,
    StochasticPlanner,
)


class TestGreedyPlanner:
    """Test GreedyPlanner selects the best action."""

    @pytest.fixture
    def action_space(self) -> list[ActionSpec]:
        return [
            ActionSpec(name="toward", parameters={"effect": (1.0,)}),
            ActionSpec(name="away", parameters={"effect": (-1.0,)}),
            ActionSpec(name="noop", parameters={"effect": (0.0,)}),
        ]

    def test_selects_action_that_minimizes_distance(
        self, action_space: list[ActionSpec]
    ) -> None:
        planner = GreedyPlanner(action_space=action_space)
        obj = ObjectiveVector(values=(5.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(3.0,))

        policy = planner.plan(goal, state)
        assert policy.size == 1
        # State is at 3.0, goal is at 5.0. "toward" (+1.0) -> 4.0, closer
        assert policy.actions[0].name == "toward"

    def test_empty_policy_when_no_objective(self, action_space: list[ActionSpec]) -> None:
        planner = GreedyPlanner(action_space=action_space)
        goal = Goal(name="g", objective=None)
        state = StateSnapshot(timestamp=time.time(), values=(3.0,))

        policy = planner.plan(goal, state)
        assert policy.size == 0

    def test_empty_action_space_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty action space"):
            GreedyPlanner(action_space=[])

    def test_action_space_property(self, action_space: list[ActionSpec]) -> None:
        planner = GreedyPlanner(action_space=action_space)
        assert len(planner.action_space) == 3


class TestStochasticPlanner:
    """Test StochasticPlanner returns valid stochastic policy."""

    @pytest.fixture
    def action_space(self) -> list[ActionSpec]:
        return [
            ActionSpec(name="a1", parameters={"effect": (1.0,)}),
            ActionSpec(name="a2", parameters={"effect": (-1.0,)}),
            ActionSpec(name="a3", parameters={"effect": (0.0,)}),
        ]

    def test_returns_stochastic_policy(self, action_space: list[ActionSpec]) -> None:
        planner = StochasticPlanner(
            action_space=action_space, temperature=1.0, seed=42
        )
        obj = ObjectiveVector(values=(5.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(3.0,))

        policy = planner.plan(goal, state)
        assert policy.is_stochastic is True
        assert policy.size == 3
        assert policy.probabilities is not None
        assert sum(policy.probabilities) == pytest.approx(1.0)

    def test_probabilities_favor_closer_actions(
        self, action_space: list[ActionSpec]
    ) -> None:
        planner = StochasticPlanner(
            action_space=action_space, temperature=0.1, seed=42
        )
        obj = ObjectiveVector(values=(5.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(3.0,))

        policy = planner.plan(goal, state)
        assert policy.probabilities is not None
        # "a1" moves toward goal -> should have highest probability
        # Index 0 = "a1" with effect (1.0,) -> state becomes 4.0, dist=1.0
        assert policy.probabilities[0] > policy.probabilities[1]

    def test_empty_policy_no_objective(self, action_space: list[ActionSpec]) -> None:
        planner = StochasticPlanner(action_space=action_space, seed=42)
        goal = Goal(name="g", objective=None)
        state = StateSnapshot(timestamp=time.time(), values=(3.0,))

        policy = planner.plan(goal, state)
        assert policy.size == 0

    def test_sample_size(self, action_space: list[ActionSpec]) -> None:
        planner = StochasticPlanner(
            action_space=action_space, sample_size=2, seed=42
        )
        obj = ObjectiveVector(values=(5.0,), directions=(Direction.APPROACH,))
        goal = Goal(name="g", objective=obj)
        state = StateSnapshot(timestamp=time.time(), values=(3.0,))

        policy = planner.plan(goal, state)
        assert policy.size == 2

    def test_invalid_temperature(self, action_space: list[ActionSpec]) -> None:
        with pytest.raises(ValueError, match="temperature must be positive"):
            StochasticPlanner(action_space=action_space, temperature=0.0)

    def test_empty_action_space_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty action space"):
            StochasticPlanner(action_space=[])
