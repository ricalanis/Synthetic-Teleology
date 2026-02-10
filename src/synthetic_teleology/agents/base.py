"""Base agent abstraction for the Synthetic Teleology framework.

Defines ``BaseAgent``, the abstract base class for all teleological agents.
Implements a finite-state-machine lifecycle with validated transitions and
provides the abstract contract (perceive -> evaluate -> revise -> plan -> act)
that concrete agents must fulfil.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

from synthetic_teleology.domain.aggregates import AgentIdentity, ConstraintSet
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import AgentState
from synthetic_teleology.domain.values import (
    ActionSpec,
    EvalSignal,
    GoalRevision,
    PolicySpec,
    StateSnapshot,
)
from synthetic_teleology.infrastructure.event_bus import EventBus

if TYPE_CHECKING:
    from synthetic_teleology.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid state transitions (finite-state machine)
# ---------------------------------------------------------------------------

_VALID_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.IDLE: {AgentState.PERCEIVING},
    AgentState.PERCEIVING: {AgentState.EVALUATING},
    AgentState.EVALUATING: {AgentState.REVISING, AgentState.PLANNING},
    AgentState.REVISING: {AgentState.PLANNING},
    AgentState.PLANNING: {AgentState.ACTING},
    AgentState.ACTING: {AgentState.REFLECTING, AgentState.IDLE},
    AgentState.REFLECTING: {AgentState.IDLE},
}


# ---------------------------------------------------------------------------
# BaseAgent ABC
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Abstract base class for all teleological agents.

    Provides:

    * **State machine** -- validated transitions through the perceive-evaluate-
      revise-plan-act-reflect lifecycle via :meth:`_transition_to`.
    * **Identity & history** -- delegated to :class:`AgentIdentity` aggregate
      which tracks the current goal and full revision log.
    * **Event publishing** -- all agents hold a reference to an :class:`EventBus`
      for emitting domain events.
    * **Lifecycle hooks** -- :meth:`on_goal_revised`, :meth:`on_constraint_violated`,
      and :meth:`on_reflection_triggered` can be overridden for side-effects
      without touching the core loop.

    Subclasses **must** implement the five abstract methods that constitute
    the teleological loop.
    """

    def __init__(
        self,
        agent_id: str,
        initial_goal: Goal,
        event_bus: EventBus,
    ) -> None:
        self._identity = AgentIdentity(agent_id, initial_goal)
        self._state = AgentState.IDLE
        self._event_bus = event_bus
        self._step_count: int = 0

    # -- public read-only properties ----------------------------------------

    @property
    def id(self) -> str:
        """Unique identifier for this agent."""
        return self._identity.agent_id

    @property
    def state(self) -> AgentState:
        """Current lifecycle state."""
        return self._state

    @property
    def current_goal(self) -> Goal:
        """The goal currently being pursued."""
        return self._identity.current_goal

    @property
    def goal_history(self) -> Sequence[GoalRevision]:
        """Chronological list of all goal revisions."""
        return self._identity.revision_log

    @property
    def step_count(self) -> int:
        """Number of completed perceive-act cycles."""
        return self._step_count

    @property
    def event_bus(self) -> EventBus:
        """The event bus this agent publishes to."""
        return self._event_bus

    # -- abstract teleological loop methods ---------------------------------

    @abstractmethod
    def perceive(self, env: BaseEnvironment) -> StateSnapshot:
        """Observe the environment and return a state snapshot.

        Implementations should transition to ``PERCEIVING`` then ``EVALUATING``
        (or delegate that to the caller / loop).
        """
        ...

    @abstractmethod
    def evaluate(self, goal: Goal, state: StateSnapshot) -> EvalSignal:
        """Compute Delta(G_t, S_t) -- how well the current state satisfies the goal.

        Returns an :class:`EvalSignal` with a scalar score in ``[-1, 1]``.
        """
        ...

    @abstractmethod
    def revise_goal(
        self,
        goal: Goal,
        state: StateSnapshot,
        delta: EvalSignal,
        constraints: ConstraintSet,
    ) -> Goal | None:
        """Optionally revise the goal based on evaluation and constraints.

        Returns the new :class:`Goal` if a revision occurred, or ``None``
        if the current goal is kept.
        """
        ...

    @abstractmethod
    def plan(self, goal: Goal, state: StateSnapshot) -> PolicySpec:
        """Generate a policy (action plan) to pursue the goal from the current state."""
        ...

    @abstractmethod
    def select_action(self, policy: PolicySpec) -> ActionSpec:
        """Select a concrete action from the policy for immediate execution."""
        ...

    # -- lifecycle hooks (overridable) --------------------------------------

    def on_goal_revised(self, revision: GoalRevision) -> None:
        """Hook called after a goal revision has been recorded.

        Default implementation is a no-op.  Override to trigger side-effects
        such as logging, metric recording, or notifying collaborators.
        """

    def on_constraint_violated(self, violation: str) -> None:
        """Hook called when a constraint check detects a violation.

        Parameters
        ----------
        violation:
            Human-readable description of the constraint violation.
        """

    def on_reflection_triggered(self) -> None:
        """Hook called when the agent enters the REFLECTING state.

        Use for meta-learning, performance introspection, or self-critique.
        """

    # -- state machine mechanics --------------------------------------------

    def _transition_to(self, new_state: AgentState) -> None:
        """Transition to *new_state* if the move is legal.

        Raises
        ------
        ValueError
            If the transition from the current state to *new_state* is not
            permitted by the lifecycle FSM.
        """
        valid = _VALID_TRANSITIONS.get(self._state, set())
        if new_state not in valid:
            raise ValueError(
                f"Invalid state transition: {self._state.value} -> {new_state.value}. "
                f"Valid targets from {self._state.value}: "
                f"{sorted(s.value for s in valid)}"
            )
        logger.debug(
            "Agent %s: %s -> %s",
            self.id,
            self._state.value,
            new_state.value,
        )
        self._state = new_state

    def _record_revision(self, new_goal: Goal, revision: GoalRevision) -> None:
        """Record a goal revision in the agent's identity aggregate.

        Updates the current goal, appends to the revision log, and invokes
        the :meth:`on_goal_revised` hook.
        """
        self._identity.record_revision(new_goal, revision)

    def _increment_step(self) -> None:
        """Increment the step counter (called at the end of a full cycle)."""
        self._step_count += 1

    def _reset_to_idle(self) -> None:
        """Force the agent back to IDLE state.

        This is a controlled reset used after completing a full cycle or
        after error recovery.  It bypasses normal transition validation.
        """
        logger.debug("Agent %s: forced reset to IDLE", self.id)
        self._state = AgentState.IDLE

    # -- dunder helpers -----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"id={self.id!r}, "
            f"state={self._state.value}, "
            f"goal={self.current_goal.goal_id!r}, "
            f"steps={self._step_count})"
        )
