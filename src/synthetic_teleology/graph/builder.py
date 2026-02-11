"""Fluent builder for assembling a compiled teleological graph.

``GraphBuilder`` mirrors the ``AgentBuilder`` API but produces a compiled
LangGraph ``StateGraph`` instead of a ``TeleologicalAgent``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import ActionSpec, ObjectiveVector
from synthetic_teleology.graph.graph import build_teleological_graph
from synthetic_teleology.services.constraint_engine import (
    ConstraintPipeline,
    PolicyFilter,
)
from synthetic_teleology.services.evaluation import BaseEvaluator, NumericEvaluator
from synthetic_teleology.services.goal_revision import BaseGoalUpdater, ThresholdUpdater
from synthetic_teleology.services.planning import BasePlanner, GreedyPlanner


def _build_default_action_space(dimensions: int, step_size: float = 0.5) -> list[ActionSpec]:
    """Generate a default action space for an N-dimensional continuous space."""
    actions: list[ActionSpec] = []
    for d in range(dimensions):
        effect_pos = tuple(step_size if i == d else 0.0 for i in range(dimensions))
        actions.append(
            ActionSpec(
                name=f"step_dim{d}_pos",
                parameters={"effect": effect_pos, "delta": effect_pos},
            )
        )
        effect_neg = tuple(-step_size if i == d else 0.0 for i in range(dimensions))
        actions.append(
            ActionSpec(
                name=f"step_dim{d}_neg",
                parameters={"effect": effect_neg, "delta": effect_neg},
            )
        )
    noop_effect = tuple(0.0 for _ in range(dimensions))
    actions.append(
        ActionSpec(name="noop", parameters={"effect": noop_effect, "delta": noop_effect})
    )
    return actions


class GraphBuilder:
    """Fluent builder for assembling a compiled teleological LangGraph.

    Example
    -------
    ::

        app, initial = (
            GraphBuilder("agent-1")
            .with_objective((5.0, 5.0))
            .with_evaluator(NumericEvaluator())
            .with_planner(planner)
            .with_environment(perceive_fn=obs, transition_fn=step)
            .with_checkpointer(MemorySaver())
            .build()
        )
        result = app.invoke(initial)
    """

    def __init__(self, agent_id: str) -> None:
        self._agent_id = agent_id
        self._goal: Goal | None = None
        self._evaluator: BaseEvaluator | None = None
        self._updater: BaseGoalUpdater | None = None
        self._planner: BasePlanner | None = None
        self._constraint_checkers: list[Any] = []
        self._checkpointer: Any | None = None
        self._action_step_size: float = 0.5
        self._max_steps: int = 100
        self._goal_achieved_threshold: float = 0.9
        self._perceive_fn: Callable | None = None
        self._act_fn: Callable | None = None
        self._transition_fn: Callable | None = None
        self._metadata: dict[str, Any] = {}

    # -- fluent setters ---

    def with_goal(self, goal: Goal) -> GraphBuilder:
        """Set the initial goal."""
        self._goal = goal
        return self

    def with_objective(
        self,
        values: tuple[float, ...],
        directions: tuple[Direction, ...] | None = None,
        weights: tuple[float, ...] | None = None,
        goal_name: str = "",
    ) -> GraphBuilder:
        """Create a goal from raw objective parameters."""
        dirs = directions or tuple(Direction.APPROACH for _ in values)
        objective = ObjectiveVector(values=values, directions=dirs, weights=weights)
        self._goal = Goal(
            name=goal_name or f"{self._agent_id}-goal",
            objective=objective,
        )
        return self

    def with_evaluator(self, evaluator: BaseEvaluator) -> GraphBuilder:
        """Set the evaluation strategy."""
        self._evaluator = evaluator
        return self

    def with_goal_updater(self, updater: BaseGoalUpdater) -> GraphBuilder:
        """Set the goal-revision strategy."""
        self._updater = updater
        return self

    def with_planner(self, planner: BasePlanner) -> GraphBuilder:
        """Set the planning strategy."""
        self._planner = planner
        return self

    def with_constraint_checkers(self, *checkers: Any) -> GraphBuilder:
        """Add constraint checkers."""
        self._constraint_checkers.extend(checkers)
        return self

    def with_checkpointer(self, checkpointer: Any) -> GraphBuilder:
        """Set a LangGraph checkpointer for persistence."""
        self._checkpointer = checkpointer
        return self

    def with_action_step_size(self, step_size: float) -> GraphBuilder:
        """Set step size for auto-generated action space."""
        self._action_step_size = step_size
        return self

    def with_max_steps(self, max_steps: int) -> GraphBuilder:
        """Set maximum loop iterations."""
        self._max_steps = max_steps
        return self

    def with_goal_achieved_threshold(self, threshold: float) -> GraphBuilder:
        """Set the eval score threshold for goal achievement."""
        self._goal_achieved_threshold = threshold
        return self

    def with_environment(
        self,
        perceive_fn: Callable | None = None,
        act_fn: Callable | None = None,
        transition_fn: Callable | None = None,
    ) -> GraphBuilder:
        """Set environment interaction functions."""
        if perceive_fn is not None:
            self._perceive_fn = perceive_fn
        if act_fn is not None:
            self._act_fn = act_fn
        if transition_fn is not None:
            self._transition_fn = transition_fn
        return self

    def with_metadata(self, **kwargs: Any) -> GraphBuilder:
        """Set additional metadata."""
        self._metadata.update(kwargs)
        return self

    # -- build ---

    def build(self) -> tuple[Any, dict[str, Any]]:
        """Validate and build the compiled graph + initial state.

        Returns
        -------
        tuple[CompiledStateGraph, dict]
            The compiled graph and the initial state dict for ``app.invoke(initial_state)``.

        Raises
        ------
        ValueError
            If no goal has been specified.
        RuntimeError
            If no perceive_fn has been specified.
        """
        if self._goal is None:
            raise ValueError(
                "GraphBuilder requires a goal. "
                "Call .with_goal(goal) or .with_objective(...) before .build()."
            )
        if self._perceive_fn is None:
            raise RuntimeError(
                "GraphBuilder requires a perceive_fn. "
                "Call .with_environment(perceive_fn=...) before .build()."
            )

        evaluator = self._evaluator or NumericEvaluator()
        updater = self._updater or ThresholdUpdater()

        if self._planner is not None:
            planner = self._planner
        else:
            ndims = (
                self._goal.objective.dimension
                if self._goal.objective is not None
                else 2
            )
            action_space = _build_default_action_space(ndims, self._action_step_size)
            planner = GreedyPlanner(action_space=action_space)

        pipeline = ConstraintPipeline(checkers=self._constraint_checkers)
        policy_filter = PolicyFilter(pipeline)

        app = build_teleological_graph(checkpointer=self._checkpointer)

        initial_state: dict[str, Any] = {
            "step": 0,
            "max_steps": self._max_steps,
            "goal_achieved_threshold": self._goal_achieved_threshold,
            "goal": self._goal,
            "evaluator": evaluator,
            "goal_updater": updater,
            "planner": planner,
            "constraint_pipeline": pipeline,
            "policy_filter": policy_filter,
            "perceive_fn": self._perceive_fn,
            "act_fn": self._act_fn,
            "transition_fn": self._transition_fn,
            "events": [],
            "goal_history": [],
            "eval_history": [],
            "action_history": [],
            "metadata": self._metadata,
        }

        return app, initial_state

    def __repr__(self) -> str:
        parts = [f"agent_id={self._agent_id!r}"]
        if self._goal is not None:
            parts.append(f"goal={self._goal.goal_id!r}")
        if self._evaluator is not None:
            parts.append(f"evaluator={type(self._evaluator).__name__}")
        if self._planner is not None:
            parts.append(f"planner={type(self._planner).__name__}")
        return f"GraphBuilder({', '.join(parts)})"
