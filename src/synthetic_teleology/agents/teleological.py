"""Teleological agent -- the primary concrete agent for the framework.

``TeleologicalAgent`` implements the full synthetic teleology loop by
delegating each phase to injected strategy objects (evaluator, updater,
planner) and coordinating constraint enforcement.  It wires together:

* :class:`BaseEvaluator`  -- computes Delta(G_t, S_t)
* :class:`BaseGoalUpdater` -- decides whether/how to revise G_t
* :class:`BasePlanner`     -- produces a policy pi_t
* :class:`ConstraintSet`   -- environment constraints E_t

All abstract methods from :class:`BaseAgent` are fully implemented here.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from synthetic_teleology.agents.base import BaseAgent
from synthetic_teleology.domain.aggregates import ConstraintSet
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import AgentState
from synthetic_teleology.domain.events import (
    ActionExecuted,
    EvaluationCompleted,
    GoalRevised,
    LoopStepCompleted,
    PlanGenerated,
    StateChanged,
)
from synthetic_teleology.domain.values import (
    ActionSpec,
    EvalSignal,
    GoalRevision,
    PolicySpec,
    StateSnapshot,
)
from synthetic_teleology.infrastructure.event_bus import EventBus
from synthetic_teleology.services.evaluation import BaseEvaluator
from synthetic_teleology.services.goal_revision import BaseGoalUpdater
from synthetic_teleology.services.planning import BasePlanner

if TYPE_CHECKING:
    from synthetic_teleology.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)


class TeleologicalAgent(BaseAgent):
    """Full synthetic teleology agent with strategy injection.

    Composes an evaluator, goal updater, planner, and constraint set to
    execute the perceive-evaluate-revise-plan-act cycle.  Each phase emits
    the appropriate domain event through the event bus.

    Parameters
    ----------
    agent_id:
        Unique identifier for this agent.
    initial_goal:
        The starting goal.
    event_bus:
        Event bus for publishing domain events.
    evaluator:
        Strategy for computing Delta(G_t, S_t).
    updater:
        Strategy for revising goals based on evaluation signals.
    planner:
        Strategy for generating action policies.
    constraints:
        Initial constraint set (E_t).  Defaults to an empty set.

    Example
    -------
    ::

        agent = TeleologicalAgent(
            agent_id="agent-1",
            initial_goal=goal,
            event_bus=bus,
            evaluator=NumericEvaluator(),
            updater=ThresholdUpdater(threshold=0.5),
            planner=planner,
        )
    """

    def __init__(
        self,
        agent_id: str,
        initial_goal: Goal,
        event_bus: EventBus,
        evaluator: BaseEvaluator,
        updater: BaseGoalUpdater,
        planner: BasePlanner,
        constraints: ConstraintSet | None = None,
    ) -> None:
        super().__init__(agent_id, initial_goal, event_bus)
        self._evaluator = evaluator
        self._updater = updater
        self._planner = planner
        self._constraints = constraints or ConstraintSet()
        self._last_snapshot: StateSnapshot | None = None
        self._last_eval: EvalSignal | None = None
        self._last_policy: PolicySpec | None = None

    # -- strategy accessors -------------------------------------------------

    @property
    def evaluator(self) -> BaseEvaluator:
        """The evaluation strategy."""
        return self._evaluator

    @property
    def updater(self) -> BaseGoalUpdater:
        """The goal-revision strategy."""
        return self._updater

    @property
    def planner(self) -> BasePlanner:
        """The planning strategy."""
        return self._planner

    @property
    def constraints(self) -> ConstraintSet:
        """The current constraint set (E_t)."""
        return self._constraints

    @constraints.setter
    def constraints(self, value: ConstraintSet) -> None:
        self._constraints = value

    @property
    def last_snapshot(self) -> StateSnapshot | None:
        """Most recent state observation."""
        return self._last_snapshot

    @property
    def last_eval(self) -> EvalSignal | None:
        """Most recent evaluation signal."""
        return self._last_eval

    @property
    def last_policy(self) -> PolicySpec | None:
        """Most recent generated policy."""
        return self._last_policy

    # -- teleological loop implementation -----------------------------------

    def perceive(self, env: BaseEnvironment) -> StateSnapshot:
        """Observe the environment and transition through PERCEIVING -> EVALUATING.

        Parameters
        ----------
        env:
            The environment to observe.

        Returns
        -------
        StateSnapshot
            Current state of the environment.
        """
        prev_state = self._state
        self._transition_to(AgentState.PERCEIVING)
        state = env.observe()
        self._last_snapshot = state

        self._event_bus.publish(
            StateChanged(
                source_id=self.id,
                previous_state=prev_state,
                new_state=AgentState.PERCEIVING,
                timestamp=time.time(),
            )
        )

        self._transition_to(AgentState.EVALUATING)
        return state

    def evaluate(self, goal: Goal, state: StateSnapshot) -> EvalSignal:
        """Evaluate how well the current state satisfies the goal.

        Delegates to the injected :class:`BaseEvaluator` strategy.

        Parameters
        ----------
        goal:
            The goal to evaluate against.
        state:
            The current state snapshot.

        Returns
        -------
        EvalSignal
            Scalar evaluation in ``[-1, 1]`` plus per-dimension detail.
        """
        signal = self._evaluator.evaluate(goal, state)
        self._last_eval = signal

        self._event_bus.publish(
            EvaluationCompleted(
                source_id=self.id,
                goal_id=goal.goal_id,
                eval_signal=signal,
                revision_triggered=not signal.is_satisfactory,
                timestamp=time.time(),
            )
        )

        return signal

    def revise_goal(
        self,
        goal: Goal,
        state: StateSnapshot,
        delta: EvalSignal,
        constraints: ConstraintSet,
    ) -> Goal | None:
        """Decide whether to revise the goal and perform the revision if needed.

        Transitions through ``REVISING`` -> ``PLANNING``.  Delegates the
        revision decision to the injected :class:`BaseGoalUpdater`.

        Parameters
        ----------
        goal:
            Current goal.
        state:
            Current state snapshot.
        delta:
            Evaluation signal from :meth:`evaluate`.
        constraints:
            Active constraint set.

        Returns
        -------
        Goal | None
            The new revised goal, or ``None`` if no revision was needed.
        """
        self._transition_to(AgentState.REVISING)

        proposed_goal = self._updater.update(goal, state, delta, constraints)

        new_goal: Goal | None = None
        if proposed_goal is not None and proposed_goal.objective != goal.objective:
            previous_objective = goal.objective
            new_goal, revision = goal.revise(
                new_objective=proposed_goal.objective,
                reason="teleological_update",
                eval_signal=delta,
            )
            self._record_revision(new_goal, revision)
            self.on_goal_revised(revision)

            self._event_bus.publish(
                GoalRevised(
                    source_id=self.id,
                    revision=revision,
                    previous_objective=previous_objective,
                    new_objective=new_goal.objective,
                    timestamp=time.time(),
                )
            )

            logger.info(
                "Agent %s revised goal: %s -> %s (score=%.3f)",
                self.id,
                goal.goal_id,
                new_goal.goal_id,
                delta.score,
            )

        self._transition_to(AgentState.PLANNING)
        return new_goal

    def plan(self, goal: Goal, state: StateSnapshot) -> PolicySpec:
        """Generate an action policy for the current goal and state.

        Delegates to the injected :class:`BasePlanner`.

        Parameters
        ----------
        goal:
            The goal to plan for (may have been revised).
        state:
            The current state snapshot.

        Returns
        -------
        PolicySpec
            An ordered action sequence or stochastic action distribution.
        """
        t0 = time.monotonic()
        policy = self._planner.plan(goal, state)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._last_policy = policy

        self._event_bus.publish(
            PlanGenerated(
                source_id=self.id,
                goal_id=goal.goal_id,
                policy=policy,
                planning_time_ms=elapsed_ms,
                timestamp=time.time(),
            )
        )

        return policy

    def select_action(self, policy: PolicySpec) -> ActionSpec:
        """Select a concrete action from the policy.

        For stochastic policies, samples according to the probability
        distribution.  For deterministic policies, returns the first action.
        Falls back to a ``noop`` action if the policy is empty.

        Parameters
        ----------
        policy:
            The policy to select from.

        Returns
        -------
        ActionSpec
            The selected action.
        """
        self._transition_to(AgentState.ACTING)

        if not policy.actions:
            logger.warning("Agent %s: empty policy, selecting noop", self.id)
            return ActionSpec(name="noop")

        if policy.is_stochastic and policy.probabilities:
            idx = int(np.random.choice(len(policy.actions), p=policy.probabilities))
            action = policy.actions[idx]
        else:
            action = policy.actions[0]

        self._event_bus.publish(
            ActionExecuted(
                source_id=self.id,
                action=action,
                timestamp=time.time(),
            )
        )

        return action

    # -- full cycle convenience method --------------------------------------

    def run_cycle(self, env: BaseEnvironment) -> ActionSpec:
        """Execute one full perceive-evaluate-revise-plan-act cycle.

        This is a convenience method that orchestrates all phases in sequence.
        After acting, the agent transitions to REFLECTING (if the hook is
        non-trivial) or directly back to IDLE.

        Parameters
        ----------
        env:
            The environment to interact with.

        Returns
        -------
        ActionSpec
            The action that was selected and should be applied to the environment.
        """
        t0 = time.monotonic()

        # Perceive
        state = self.perceive(env)

        # Evaluate
        goal = self.current_goal
        delta = self.evaluate(goal, state)

        # Revise (may update self.current_goal)
        new_goal = self.revise_goal(goal, state, delta, self._constraints)
        active_goal = new_goal if new_goal is not None else goal

        # Plan
        policy = self.plan(active_goal, state)

        # Act
        action = self.select_action(policy)

        # Reflect / return to idle
        self._transition_to(AgentState.REFLECTING)
        self.on_reflection_triggered()
        self._transition_to(AgentState.IDLE)

        self._increment_step()

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._event_bus.publish(
            LoopStepCompleted(
                source_id=self.id,
                step_number=self.step_count,
                agent_state=AgentState.IDLE,
                elapsed_ms=elapsed_ms,
                goal_id=active_goal.goal_id,
                timestamp=time.time(),
            )
        )

        return action

    # -- hooks --------------------------------------------------------------

    def on_goal_revised(self, revision: GoalRevision) -> None:
        """Log goal revision at INFO level."""
        logger.info(
            "Agent %s: goal revised (%s -> %s), reason: %s",
            self.id,
            revision.previous_goal_id,
            revision.new_goal_id,
            revision.reason,
        )

    def on_constraint_violated(self, violation: str) -> None:
        """Log constraint violation at WARNING level."""
        logger.warning(
            "Agent %s: constraint violated: %s",
            self.id,
            violation,
        )

    def on_reflection_triggered(self) -> None:
        """Log reflection trigger at DEBUG level."""
        logger.debug(
            "Agent %s: reflection triggered at step %d (score=%.3f)",
            self.id,
            self.step_count,
            self._last_eval.score if self._last_eval else 0.0,
        )
