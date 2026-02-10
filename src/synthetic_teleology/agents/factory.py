"""Agent creation patterns for the Synthetic Teleology framework.

Provides two complementary creation patterns:

* :class:`AgentFactory` -- Static factory methods for common agent
  configurations (simple, teleological, multi-agent).
* :class:`AgentBuilder` -- Fluent builder for fine-grained agent assembly.

Example
-------
Factory style::

    agent = AgentFactory.create_teleological_agent(
        agent_id="agent-1",
        goal=goal,
        event_bus=bus,
        evaluator=NumericEvaluator(),
        updater=ThresholdUpdater(threshold=0.3),
        planner=planner,
    )

Builder style::

    agent = (
        AgentBuilder("agent-1")
        .with_goal(goal)
        .with_evaluator(NumericEvaluator())
        .with_goal_updater(ThresholdUpdater(threshold=0.3))
        .with_planner(planner)
        .with_constraints(safety_constraint, budget_constraint)
        .with_event_bus(bus)
        .build()
    )
"""

from __future__ import annotations

from typing import Sequence

from synthetic_teleology.agents.base import BaseAgent
from synthetic_teleology.agents.teleological import TeleologicalAgent
from synthetic_teleology.domain.aggregates import ConstraintSet
from synthetic_teleology.domain.entities import Constraint, Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import ActionSpec, ObjectiveVector
from synthetic_teleology.infrastructure.event_bus import EventBus
from synthetic_teleology.services.evaluation import BaseEvaluator, NumericEvaluator
from synthetic_teleology.services.goal_revision import BaseGoalUpdater, ThresholdUpdater
from synthetic_teleology.services.planning import BasePlanner, GreedyPlanner


def _build_default_action_space(dimensions: int, step_size: float = 0.5) -> list[ActionSpec]:
    """Generate a default action space for an N-dimensional continuous space.

    Creates ``2 * dimensions + 1`` actions: positive and negative steps
    along each axis, plus a no-op.

    Parameters
    ----------
    dimensions:
        Number of state dimensions.
    step_size:
        Magnitude of each step action.

    Returns
    -------
    list[ActionSpec]
        A set of primitive actions for navigating the space.
    """
    actions: list[ActionSpec] = []
    for d in range(dimensions):
        # Positive step along dimension d
        effect_pos = tuple(step_size if i == d else 0.0 for i in range(dimensions))
        actions.append(
            ActionSpec(
                name=f"step_dim{d}_pos",
                parameters={"effect": effect_pos, "delta": effect_pos},
            )
        )
        # Negative step along dimension d
        effect_neg = tuple(-step_size if i == d else 0.0 for i in range(dimensions))
        actions.append(
            ActionSpec(
                name=f"step_dim{d}_neg",
                parameters={"effect": effect_neg, "delta": effect_neg},
            )
        )
    # No-op
    noop_effect = tuple(0.0 for _ in range(dimensions))
    actions.append(
        ActionSpec(
            name="noop",
            parameters={"effect": noop_effect, "delta": noop_effect},
        )
    )
    return actions


# ---------------------------------------------------------------------------
# AgentFactory (static factory methods)
# ---------------------------------------------------------------------------

class AgentFactory:
    """Static factory methods for creating pre-configured agents.

    Each factory method returns a fully wired :class:`TeleologicalAgent`
    (or subclass) with sensible defaults for a particular use-case.
    """

    @staticmethod
    def create_simple_agent(
        agent_id: str,
        target_values: tuple[float, ...],
        event_bus: EventBus | None = None,
        *,
        directions: tuple[Direction, ...] | None = None,
        threshold: float = 0.5,
        step_size: float = 0.5,
    ) -> TeleologicalAgent:
        """Create a minimal teleological agent with numeric evaluation.

        Uses :class:`NumericEvaluator`, :class:`ThresholdUpdater`, and
        :class:`GreedyPlanner` with an auto-generated action space.

        Parameters
        ----------
        agent_id:
            Unique identifier for the agent.
        target_values:
            Target values for the objective vector (one per dimension).
        event_bus:
            Event bus instance.  If ``None``, a new one is created.
        directions:
            Optimization direction per dimension.  Defaults to ``APPROACH``
            for every dimension.
        threshold:
            Evaluation score magnitude above which goal revision triggers.
            Must be in ``(0, 1]``.  Defaults to ``0.5``.
        step_size:
            Step magnitude for the auto-generated action space.

        Returns
        -------
        TeleologicalAgent
            A fully configured agent ready for use.
        """
        bus = event_bus or EventBus()
        dims = directions or tuple(Direction.APPROACH for _ in target_values)
        ndims = len(target_values)

        objective = ObjectiveVector(values=target_values, directions=dims)
        goal = Goal(name=f"{agent_id}-goal", objective=objective)
        evaluator = NumericEvaluator()
        updater = ThresholdUpdater(threshold=threshold)
        action_space = _build_default_action_space(ndims, step_size=step_size)
        planner = GreedyPlanner(action_space=action_space)

        return TeleologicalAgent(
            agent_id=agent_id,
            initial_goal=goal,
            event_bus=bus,
            evaluator=evaluator,
            updater=updater,
            planner=planner,
        )

    @staticmethod
    def create_teleological_agent(
        agent_id: str,
        goal: Goal,
        event_bus: EventBus,
        evaluator: BaseEvaluator,
        updater: BaseGoalUpdater,
        planner: BasePlanner,
        constraints: ConstraintSet | None = None,
    ) -> TeleologicalAgent:
        """Create a teleological agent with fully specified strategies.

        Parameters
        ----------
        agent_id:
            Unique identifier for the agent.
        goal:
            Initial goal entity.
        event_bus:
            Event bus for domain events.
        evaluator:
            Evaluation strategy.
        updater:
            Goal-revision strategy.
        planner:
            Planning strategy.
        constraints:
            Optional constraint set.

        Returns
        -------
        TeleologicalAgent
            A fully configured agent.
        """
        return TeleologicalAgent(
            agent_id=agent_id,
            initial_goal=goal,
            event_bus=event_bus,
            evaluator=evaluator,
            updater=updater,
            planner=planner,
            constraints=constraints,
        )

    @staticmethod
    def create_constrained_agent(
        agent_id: str,
        goal: Goal,
        event_bus: EventBus,
        evaluator: BaseEvaluator,
        updater: BaseGoalUpdater,
        planner: BasePlanner,
        constraints: Sequence[Constraint],
    ) -> TeleologicalAgent:
        """Create a teleological agent with an explicit list of constraints.

        Convenience wrapper that builds a :class:`ConstraintSet` from the
        provided constraint sequence.

        Parameters
        ----------
        agent_id:
            Unique identifier.
        goal:
            Initial goal.
        event_bus:
            Event bus.
        evaluator:
            Evaluation strategy.
        updater:
            Goal-revision strategy.
        planner:
            Planning strategy.
        constraints:
            Sequence of :class:`Constraint` entities to enforce.

        Returns
        -------
        TeleologicalAgent
            A fully configured agent with constraints.
        """
        constraint_set = ConstraintSet()
        for constraint in constraints:
            constraint_set.add(constraint)

        return TeleologicalAgent(
            agent_id=agent_id,
            initial_goal=goal,
            event_bus=event_bus,
            evaluator=evaluator,
            updater=updater,
            planner=planner,
            constraints=constraint_set,
        )


# ---------------------------------------------------------------------------
# AgentBuilder (fluent builder pattern)
# ---------------------------------------------------------------------------

class AgentBuilder:
    """Fluent builder for assembling a :class:`TeleologicalAgent`.

    Collects configuration piece-by-piece and validates completeness at
    build time.

    Example
    -------
    ::

        agent = (
            AgentBuilder("agent-1")
            .with_goal(goal)
            .with_evaluator(NumericEvaluator())
            .with_goal_updater(ThresholdUpdater(threshold=0.5))
            .with_planner(planner)
            .with_constraints(constraint_a, constraint_b)
            .with_event_bus(bus)
            .build()
        )
    """

    def __init__(self, agent_id: str) -> None:
        self._agent_id = agent_id
        self._goal: Goal | None = None
        self._evaluator: BaseEvaluator | None = None
        self._updater: BaseGoalUpdater | None = None
        self._planner: BasePlanner | None = None
        self._constraint_set = ConstraintSet()
        self._event_bus: EventBus | None = None
        self._action_step_size: float = 0.5

    # -- fluent setters -----------------------------------------------------

    def with_goal(self, goal: Goal) -> AgentBuilder:
        """Set the initial goal.

        Parameters
        ----------
        goal:
            The :class:`Goal` entity to pursue.

        Returns
        -------
        AgentBuilder
            ``self`` for chaining.
        """
        self._goal = goal
        return self

    def with_objective(
        self,
        values: tuple[float, ...],
        directions: tuple[Direction, ...] | None = None,
        weights: tuple[float, ...] | None = None,
        goal_name: str = "",
    ) -> AgentBuilder:
        """Convenience: create a goal from raw objective parameters.

        Parameters
        ----------
        values:
            Target values for the objective vector.
        directions:
            Optimization direction per dimension.  Defaults to ``APPROACH``.
        weights:
            Optional per-dimension weights.
        goal_name:
            Optional name for the created goal.

        Returns
        -------
        AgentBuilder
            ``self`` for chaining.
        """
        dirs = directions or tuple(Direction.APPROACH for _ in values)
        objective = ObjectiveVector(values=values, directions=dirs, weights=weights)
        self._goal = Goal(
            name=goal_name or f"{self._agent_id}-goal",
            objective=objective,
        )
        return self

    def with_evaluator(self, evaluator: BaseEvaluator) -> AgentBuilder:
        """Set the evaluation strategy.

        Parameters
        ----------
        evaluator:
            A :class:`BaseEvaluator` implementation.

        Returns
        -------
        AgentBuilder
            ``self`` for chaining.
        """
        self._evaluator = evaluator
        return self

    def with_goal_updater(self, updater: BaseGoalUpdater) -> AgentBuilder:
        """Set the goal-revision strategy.

        Parameters
        ----------
        updater:
            A :class:`BaseGoalUpdater` implementation.

        Returns
        -------
        AgentBuilder
            ``self`` for chaining.
        """
        self._updater = updater
        return self

    def with_planner(self, planner: BasePlanner) -> AgentBuilder:
        """Set the planning strategy.

        Parameters
        ----------
        planner:
            A :class:`BasePlanner` implementation.

        Returns
        -------
        AgentBuilder
            ``self`` for chaining.
        """
        self._planner = planner
        return self

    def with_constraints(self, *constraints: Constraint) -> AgentBuilder:
        """Add one or more constraints.

        Parameters
        ----------
        *constraints:
            :class:`Constraint` entities to add to the constraint set.

        Returns
        -------
        AgentBuilder
            ``self`` for chaining.
        """
        for c in constraints:
            self._constraint_set.add(c)
        return self

    def with_constraint_set(self, constraint_set: ConstraintSet) -> AgentBuilder:
        """Replace the entire constraint set.

        Parameters
        ----------
        constraint_set:
            A pre-built :class:`ConstraintSet`.

        Returns
        -------
        AgentBuilder
            ``self`` for chaining.
        """
        self._constraint_set = constraint_set
        return self

    def with_event_bus(self, event_bus: EventBus) -> AgentBuilder:
        """Set the event bus.

        Parameters
        ----------
        event_bus:
            An :class:`EventBus` instance.

        Returns
        -------
        AgentBuilder
            ``self`` for chaining.
        """
        self._event_bus = event_bus
        return self

    def with_action_step_size(self, step_size: float) -> AgentBuilder:
        """Set the step size for the auto-generated default action space.

        Only used if no planner is explicitly set.

        Parameters
        ----------
        step_size:
            Magnitude of each axis-aligned step.

        Returns
        -------
        AgentBuilder
            ``self`` for chaining.
        """
        self._action_step_size = step_size
        return self

    # -- build --------------------------------------------------------------

    def build(self) -> TeleologicalAgent:
        """Validate configuration and construct the agent.

        Missing required components are filled with sensible defaults:

        * ``event_bus`` -> new :class:`EventBus`
        * ``evaluator`` -> :class:`NumericEvaluator`
        * ``updater`` -> :class:`ThresholdUpdater` with default threshold
        * ``planner`` -> :class:`GreedyPlanner` with auto-generated action space

        Raises
        ------
        ValueError
            If no goal has been specified (the only truly required parameter).

        Returns
        -------
        TeleologicalAgent
            The fully assembled agent.
        """
        if self._goal is None:
            raise ValueError(
                "AgentBuilder requires a goal.  "
                "Call .with_goal(goal) or .with_objective(...) before .build()."
            )

        bus = self._event_bus or EventBus()
        evaluator = self._evaluator or NumericEvaluator()
        updater = self._updater or ThresholdUpdater()

        if self._planner is not None:
            planner = self._planner
        else:
            # Build a default GreedyPlanner with an auto-generated action space
            ndims = (
                self._goal.objective.dimension
                if self._goal.objective is not None
                else 2
            )
            action_space = _build_default_action_space(
                ndims, step_size=self._action_step_size
            )
            planner = GreedyPlanner(action_space=action_space)

        return TeleologicalAgent(
            agent_id=self._agent_id,
            initial_goal=self._goal,
            event_bus=bus,
            evaluator=evaluator,
            updater=updater,
            planner=planner,
            constraints=self._constraint_set,
        )

    # -- introspection ------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"agent_id={self._agent_id!r}"]
        if self._goal is not None:
            parts.append(f"goal={self._goal.goal_id!r}")
        if self._evaluator is not None:
            parts.append(f"evaluator={type(self._evaluator).__name__}")
        if self._updater is not None:
            parts.append(f"updater={type(self._updater).__name__}")
        if self._planner is not None:
            parts.append(f"planner={type(self._planner).__name__}")
        parts.append(f"constraints={len(self._constraint_set)}")
        return f"AgentBuilder({', '.join(parts)})"
