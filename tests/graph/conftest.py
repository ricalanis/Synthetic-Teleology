"""Shared fixtures for graph tests."""

from __future__ import annotations

import pytest

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import (
    ActionSpec,
    ObjectiveVector,
)
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.services.constraint_engine import ConstraintPipeline, PolicyFilter
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.goal_revision import ThresholdUpdater
from synthetic_teleology.services.planning import GreedyPlanner


@pytest.fixture
def action_space() -> list[ActionSpec]:
    """2-D action space with 5 actions."""
    actions: list[ActionSpec] = []
    for d in range(2):
        for sign, label in [(0.5, "pos"), (-0.5, "neg")]:
            effect = tuple(sign if i == d else 0.0 for i in range(2))
            actions.append(
                ActionSpec(
                    name=f"step_dim{d}_{label}",
                    parameters={"effect": effect, "delta": effect},
                )
            )
    actions.append(
        ActionSpec(
            name="noop",
            parameters={"effect": (0.0, 0.0), "delta": (0.0, 0.0)},
        )
    )
    return actions


@pytest.fixture
def env() -> NumericEnvironment:
    return NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))


@pytest.fixture
def goal() -> Goal:
    objective = ObjectiveVector(
        values=(5.0, 5.0),
        directions=(Direction.APPROACH, Direction.APPROACH),
    )
    return Goal(name="test-goal", objective=objective)


@pytest.fixture
def evaluator() -> NumericEvaluator:
    return NumericEvaluator(max_distance=10.0)


@pytest.fixture
def updater() -> ThresholdUpdater:
    return ThresholdUpdater(threshold=0.8)


@pytest.fixture
def planner(action_space: list[ActionSpec]) -> GreedyPlanner:
    return GreedyPlanner(action_space=action_space)


@pytest.fixture
def pipeline() -> ConstraintPipeline:
    return ConstraintPipeline(checkers=[])


@pytest.fixture
def policy_filter(pipeline: ConstraintPipeline) -> PolicyFilter:
    return PolicyFilter(pipeline)


@pytest.fixture
def initial_state(
    goal: Goal,
    evaluator: NumericEvaluator,
    updater: ThresholdUpdater,
    planner: GreedyPlanner,
    pipeline: ConstraintPipeline,
    policy_filter: PolicyFilter,
    env: NumericEnvironment,
) -> dict:
    """A fully populated initial state dict for graph invocation."""
    return {
        "step": 0,
        "max_steps": 20,
        "goal_achieved_threshold": 0.9,
        "goal": goal,
        "evaluator": evaluator,
        "goal_updater": updater,
        "planner": planner,
        "constraint_pipeline": pipeline,
        "policy_filter": policy_filter,
        "perceive_fn": lambda: env.observe(),
        "act_fn": lambda p, s: p.actions[0] if p.size > 0 else None,
        "transition_fn": lambda a: env.step(a) if a else None,
        "events": [],
        "goal_history": [],
        "eval_history": [],
        "action_history": [],
        "metadata": {},
    }
