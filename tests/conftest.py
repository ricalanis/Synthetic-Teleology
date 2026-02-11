"""Shared fixtures for the Synthetic Teleology test suite."""

from __future__ import annotations

import time

import pytest

from synthetic_teleology.domain.aggregates import ConstraintSet
from synthetic_teleology.domain.entities import Constraint, Goal
from synthetic_teleology.domain.enums import (
    ConstraintType,
    Direction,
    GoalStatus,
    StateSource,
)
from synthetic_teleology.domain.values import (
    ActionSpec,
    ConstraintSpec,
    EvalSignal,
    ObjectiveVector,
    StateSnapshot,
)
from synthetic_teleology.infrastructure.event_bus import EventBus
from synthetic_teleology.services.constraint_engine import (
    ConstraintPipeline,
)
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.goal_revision import ThresholdUpdater
from synthetic_teleology.services.planning import GreedyPlanner

# ---------------------------------------------------------------------------
# Value-object fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_objective() -> ObjectiveVector:
    """2-D objective approaching (5.0, 5.0)."""
    return ObjectiveVector(
        values=(5.0, 5.0),
        directions=(Direction.APPROACH, Direction.APPROACH),
        weights=(1.0, 1.0),
    )


@pytest.fixture
def sample_state() -> StateSnapshot:
    """2-D state at (3.0, 4.0)."""
    return StateSnapshot(
        timestamp=time.time(),
        values=(3.0, 4.0),
        source=StateSource.ENVIRONMENT,
    )


@pytest.fixture
def sample_eval_signal() -> EvalSignal:
    """A moderately positive evaluation signal."""
    return EvalSignal(
        score=0.4,
        dimension_scores=(0.3, 0.5),
        confidence=0.9,
        explanation="test signal",
    )


@pytest.fixture
def perfect_eval_signal() -> EvalSignal:
    """A perfect evaluation signal."""
    return EvalSignal(score=1.0, confidence=1.0, explanation="perfect")


@pytest.fixture
def negative_eval_signal() -> EvalSignal:
    """A strongly negative evaluation signal."""
    return EvalSignal(
        score=-0.8,
        dimension_scores=(-0.7, -0.9),
        confidence=0.95,
        explanation="negative signal",
    )


# ---------------------------------------------------------------------------
# Entity fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_goal(sample_objective: ObjectiveVector) -> Goal:
    """An active 2-D goal."""
    return Goal(
        goal_id="goal-test",
        name="test-goal",
        description="A test goal",
        objective=sample_objective,
        status=GoalStatus.ACTIVE,
    )


@pytest.fixture
def sample_constraint_spec() -> ConstraintSpec:
    """A hard constraint spec."""
    return ConstraintSpec(
        name="safety-bounds",
        constraint_type=ConstraintType.HARD,
        description="State must remain within safe bounds",
        parameters={"lower_bounds": [0.0, 0.0], "upper_bounds": [10.0, 10.0]},
    )


@pytest.fixture
def sample_constraint(sample_constraint_spec: ConstraintSpec) -> Constraint:
    """A hard constraint entity."""
    return Constraint(
        name="safety-bounds",
        constraint_type=ConstraintType.HARD,
        spec=sample_constraint_spec,
        weight=1.0,
    )


# ---------------------------------------------------------------------------
# Aggregate fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def constraint_set(sample_constraint: Constraint) -> ConstraintSet:
    """A constraint set with one safety constraint."""
    cs = ConstraintSet()
    cs.add(sample_constraint)
    return cs


@pytest.fixture
def event_bus() -> EventBus:
    """A fresh event bus."""
    return EventBus()


# ---------------------------------------------------------------------------
# Service component fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def action_space_2d() -> list[ActionSpec]:
    """A small 2-D action space with 5 actions."""
    actions = []
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
def simple_agent_components(
    action_space_2d: list[ActionSpec],
) -> dict:
    """A dict of simple agent components: evaluator, updater, planner, pipeline."""
    evaluator = NumericEvaluator(max_distance=10.0)
    updater = ThresholdUpdater(threshold=0.5, learning_rate=0.1)
    planner = GreedyPlanner(action_space=action_space_2d)
    pipeline = ConstraintPipeline(checkers=[])
    return {
        "evaluator": evaluator,
        "updater": updater,
        "planner": planner,
        "pipeline": pipeline,
    }
