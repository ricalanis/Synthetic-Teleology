"""BDI (Belief-Desire-Intention) agent for the Synthetic Teleology framework.

Implements the classical BDI architecture mapped onto the framework's
teleological loop:

* **Beliefs** -> S_t  (state observations about the world)
* **Desires** -> G_t  (prioritised list of goals the agent would like to achieve)
* **Intentions** -> pi_t (the current committed plan / policy)

The agent maintains an explicit belief base (key-value observations), a
desire set (goals ranked by priority), and an intention structure (the
currently executing policy).  The BDI-specific deliberation cycle is:

1. **Perceive** -- update beliefs from the environment.
2. **Evaluate** -- compare beliefs against the current active desire.
3. **Reconsider** -- re-evaluate desire priorities based on new beliefs.
4. **Revise goal** -- select the highest-priority achievable desire.
5. **Plan** -- form an intention (policy) from the selected desire.
6. **Select action** -- pick the next action from the intention.

Classes
-------
BDIAgent
    Full BDI agent extending :class:`BaseAgent` with belief, desire,
    and intention management.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from synthetic_teleology.agents.base import BaseAgent
from synthetic_teleology.domain.aggregates import ConstraintSet
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import AgentState, GoalStatus
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
    ObjectiveVector,
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


# ===================================================================== #
#  BDI Agent                                                              #
# ===================================================================== #


class BDIAgent(BaseAgent):
    """Classical BDI agent mapped onto the Synthetic Teleology loop.

    The agent maintains three core data structures:

    * **Beliefs** -- a mutable dictionary of observations about the world,
      updated during the perceive phase.  Keys are string identifiers;
      values are arbitrary (typically floats or state descriptions).

    * **Desires** -- an ordered list of ``(priority, Goal)`` pairs.  The
      highest-priority *achievable* desire becomes the agent's active
      intention.  Priorities are floats (higher = more important).

    * **Intentions** -- the currently committed plan (a ``PolicySpec``
      derived from the active desire).  Actions are drawn from the
      intention until the desire is achieved or reconsidered.

    The BDI agent delegates evaluation, goal revision, and planning to
    injected strategy objects, consistent with the framework's Strategy
    pattern.

    Parameters
    ----------
    agent_id:
        Unique identifier for this agent.
    initial_goal:
        The starting goal (also added as the initial top desire).
    event_bus:
        Event bus for publishing domain events.
    evaluator:
        Strategy for computing Delta(G_t, S_t).
    updater:
        Strategy for revising goals based on evaluation signals.
    planner:
        Strategy for generating action policies.
    constraints:
        Initial constraint set (E_t). Defaults to an empty set.
    initial_beliefs:
        Optional initial belief base (shallow-copied).

    Example
    -------
    ::

        agent = BDIAgent(
            agent_id="bdi-1",
            initial_goal=goal,
            event_bus=bus,
            evaluator=NumericEvaluator(),
            updater=ThresholdUpdater(threshold=0.5),
            planner=GreedyPlanner(action_space=actions),
            initial_beliefs={"position": 0.0},
        )
        agent.add_desire(secondary_goal, priority=0.5)
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
        initial_beliefs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(agent_id, initial_goal, event_bus)
        self._evaluator = evaluator
        self._updater = updater
        self._planner = planner
        self._constraints = constraints or ConstraintSet()

        # BDI core: Beliefs, Desires, Intentions
        self._beliefs: dict[str, Any] = dict(initial_beliefs or {})
        self._desires: list[tuple[float, Goal]] = [(1.0, initial_goal)]
        self._intentions: PolicySpec | None = None

        # Tracking state for the current cycle
        self._last_snapshot: StateSnapshot | None = None
        self._last_eval: EvalSignal | None = None
        self._intention_index: int = 0

    # -- public read-only properties ----------------------------------------

    @property
    def beliefs(self) -> dict[str, Any]:
        """A copy of the current belief base."""
        return dict(self._beliefs)

    @property
    def desires(self) -> list[tuple[float, Goal]]:
        """Current desires sorted by priority (highest first).

        Returns a copy; mutate via :meth:`add_desire` / :meth:`remove_desire`.
        """
        return sorted(self._desires, key=lambda x: x[0], reverse=True)

    @property
    def intentions(self) -> PolicySpec | None:
        """The current intention (committed plan), or ``None`` if not planning."""
        return self._intentions

    @property
    def active_desire(self) -> Goal | None:
        """The highest-priority active desire, or ``None`` if none exist."""
        for _, goal in self.desires:
            if goal.is_active:
                return goal
        return None

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

    # -- BDI-specific methods -----------------------------------------------

    def get_beliefs(self) -> dict[str, Any]:
        """Return the full belief base as a dictionary.

        This is the BDI-canonical accessor for the agent's world model.
        Equivalent to accessing :attr:`beliefs` but named for BDI
        convention clarity.

        Returns
        -------
        dict[str, Any]
            A copy of the belief base.
        """
        return dict(self._beliefs)

    def add_desire(self, goal: Goal, priority: float = 0.5) -> None:
        """Add a desire (goal) with a given priority.

        If a desire with the same ``goal_id`` already exists, its priority
        is updated to the new value.

        Parameters
        ----------
        goal:
            The goal to add as a desire.
        priority:
            Priority weight (higher = more important). Defaults to 0.5.
        """
        # Remove existing entry with the same goal_id, if any
        self._desires = [
            (p, g) for p, g in self._desires
            if g.goal_id != goal.goal_id
        ]
        self._desires.append((priority, goal))
        logger.debug(
            "BDIAgent %s: added desire %s (priority=%.3f, total_desires=%d)",
            self.id,
            goal.goal_id,
            priority,
            len(self._desires),
        )

    def remove_desire(self, goal_id: str) -> bool:
        """Remove a desire by its goal id.

        Parameters
        ----------
        goal_id:
            The identifier of the goal to remove from the desire set.

        Returns
        -------
        bool
            ``True`` if the desire was found and removed, ``False`` otherwise.
        """
        original_len = len(self._desires)
        self._desires = [
            (p, g) for p, g in self._desires
            if g.goal_id != goal_id
        ]
        removed = len(self._desires) < original_len
        if removed:
            logger.debug(
                "BDIAgent %s: removed desire %s (remaining=%d)",
                self.id,
                goal_id,
                len(self._desires),
            )
        return removed

    def reconsider(self, state: StateSnapshot) -> Goal | None:
        """Re-evaluate desire priorities based on the current state.

        This is the BDI "filter" function that determines whether the
        agent should switch its active desire.  The method:

        1. Scores each active desire by evaluating it against the state.
        2. Adjusts priorities: desires closer to achievement get a small
           boost; desires far from achievement and low-priority may be
           deprioritized.
        3. Returns the new top-priority active desire if it differs from
           the current goal, or ``None`` if no change is warranted.

        Parameters
        ----------
        state:
            The current state snapshot for evaluation.

        Returns
        -------
        Goal | None
            The new top-priority desire if a switch should occur,
            or ``None`` if the current intention should be maintained.
        """
        if not self._desires:
            return None

        current_goal = self.current_goal
        evaluated_desires: list[tuple[float, float, Goal]] = []

        for base_priority, goal in self._desires:
            if not goal.is_active:
                continue
            if goal.objective is None:
                evaluated_desires.append((base_priority, 0.0, goal))
                continue

            try:
                signal = self._evaluator.evaluate(goal, state)
                # Adjust priority: boost goals that are close to achievement
                # (positive score) and have high base priority
                achievement_bonus = max(0.0, signal.score) * 0.2
                adjusted_priority = base_priority + achievement_bonus
                evaluated_desires.append((adjusted_priority, signal.score, goal))
            except (ValueError, Exception) as exc:
                logger.debug(
                    "BDIAgent %s: cannot evaluate desire %s: %s",
                    self.id,
                    goal.goal_id,
                    exc,
                )
                evaluated_desires.append((base_priority, 0.0, goal))

        if not evaluated_desires:
            return None

        # Sort by adjusted priority descending
        evaluated_desires.sort(key=lambda x: x[0], reverse=True)

        # Update internal desire priorities
        self._desires = [
            (adj_priority, goal)
            for adj_priority, _, goal in evaluated_desires
        ]

        # Check if top desire is different from current goal
        top_goal = evaluated_desires[0][2]
        if top_goal.goal_id != current_goal.goal_id:
            logger.info(
                "BDIAgent %s: reconsidering desires: switching from %s to %s "
                "(priority=%.3f)",
                self.id,
                current_goal.goal_id,
                top_goal.goal_id,
                evaluated_desires[0][0],
            )
            return top_goal

        return None

    def update_belief(self, key: str, value: Any) -> None:
        """Update a single belief in the belief base.

        Parameters
        ----------
        key:
            The belief key (identifier).
        value:
            The new belief value.
        """
        self._beliefs[key] = value

    def update_beliefs(self, observations: dict[str, Any]) -> None:
        """Bulk-update beliefs from a dictionary of observations.

        Parameters
        ----------
        observations:
            Key-value pairs to merge into the belief base.
        """
        self._beliefs.update(observations)

    def clear_intentions(self) -> None:
        """Clear the current intention, forcing re-planning on the next cycle."""
        self._intentions = None
        self._intention_index = 0
        logger.debug("BDIAgent %s: intentions cleared", self.id)

    # -- abstract teleological loop implementation --------------------------

    def perceive(self, env: BaseEnvironment) -> StateSnapshot:
        """Observe the environment and update the belief base.

        Transitions through ``PERCEIVING -> EVALUATING``.  Beliefs are
        updated with the raw state values and metadata from the observation.

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

        # Update beliefs from the environment observation
        self._beliefs["_state_values"] = state.values
        self._beliefs["_state_timestamp"] = state.timestamp
        self._beliefs["_state_source"] = state.source.value
        self._beliefs["_state_dimension"] = state.dimension

        # Store individual dimension values for fine-grained belief access
        for dim_idx, val in enumerate(state.values):
            self._beliefs[f"dim_{dim_idx}"] = val

        # Merge any metadata from the snapshot
        if state.metadata:
            for key, value in state.metadata.items():
                self._beliefs[f"env_{key}"] = value

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
        """Evaluate how well the current beliefs satisfy the active desire.

        Delegates to the injected :class:`BaseEvaluator`.  The belief
        ``_last_eval_score`` is updated with the result.

        Parameters
        ----------
        goal:
            The goal (desire) to evaluate against.
        state:
            The current state snapshot.

        Returns
        -------
        EvalSignal
            Scalar evaluation in ``[-1, 1]`` plus per-dimension detail.
        """
        signal = self._evaluator.evaluate(goal, state)
        self._last_eval = signal

        # Update beliefs with the evaluation outcome
        self._beliefs["_last_eval_score"] = signal.score
        self._beliefs["_last_eval_confidence"] = signal.confidence
        self._beliefs["_last_eval_satisfactory"] = signal.is_satisfactory

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
        """Select the highest-priority achievable desire as the active goal.

        The BDI revision process:
        1. Run the standard goal updater to check if the current goal
           needs objective adjustment.
        2. Call :meth:`reconsider` to check if a different desire should
           take priority.
        3. If a desire switch occurs, record the revision and update
           the internal goal tracking.

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
            The new active goal if a revision or desire switch occurred,
            or ``None`` if the current goal is maintained.
        """
        self._transition_to(AgentState.REVISING)

        # Phase 1: Check if the standard updater wants to revise the objective
        proposed_goal = self._updater.update(goal, state, delta, constraints)

        new_goal: Goal | None = None

        if proposed_goal is not None and proposed_goal.objective != goal.objective:
            previous_objective = goal.objective
            new_goal, revision = goal.revise(
                new_objective=proposed_goal.objective,
                reason="bdi_objective_update",
                eval_signal=delta,
            )
            self._record_revision(new_goal, revision)
            self.on_goal_revised(revision)

            # Update the desire entry for this goal
            self._desires = [
                (p, new_goal if g.goal_id == goal.goal_id else g)
                for p, g in self._desires
            ]

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
                "BDIAgent %s: revised goal objective %s -> %s (score=%.3f)",
                self.id,
                goal.goal_id,
                new_goal.goal_id,
                delta.score,
            )

        # Phase 2: BDI reconsideration -- should we switch desires entirely?
        switched_desire = self.reconsider(state)
        if switched_desire is not None:
            active = new_goal or goal
            if switched_desire.goal_id != active.goal_id:
                # Record the desire switch as a revision
                switch_revision = GoalRevision(
                    timestamp=time.time(),
                    previous_goal_id=active.goal_id,
                    new_goal_id=switched_desire.goal_id,
                    reason="bdi_desire_switch",
                    eval_signal=delta,
                )
                self._record_revision(switched_desire, switch_revision)
                self.on_goal_revised(switch_revision)

                # Clear intentions since we are switching goals
                self.clear_intentions()

                new_goal = switched_desire
                logger.info(
                    "BDIAgent %s: switched active desire to %s",
                    self.id,
                    switched_desire.goal_id,
                )

        self._transition_to(AgentState.PLANNING)
        return new_goal

    def plan(self, goal: Goal, state: StateSnapshot) -> PolicySpec:
        """Form an intention (policy) from the current desire and beliefs.

        If the agent already has a valid intention (non-exhausted policy)
        for the current goal, it is reused.  Otherwise, the planner is
        invoked to generate a new intention.

        Parameters
        ----------
        goal:
            The goal (desire) to plan for.
        state:
            The current state snapshot.

        Returns
        -------
        PolicySpec
            The committed intention (action plan).
        """
        # Check if we can reuse the existing intention
        if (
            self._intentions is not None
            and self._intentions.actions
            and self._intention_index < len(self._intentions.actions)
        ):
            logger.debug(
                "BDIAgent %s: reusing existing intention "
                "(action %d/%d remaining)",
                self.id,
                self._intention_index + 1,
                len(self._intentions.actions),
            )
            return self._intentions

        # Generate a new intention via the planner
        t0 = time.monotonic()
        policy = self._planner.plan(goal, state)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        self._intentions = policy
        self._intention_index = 0

        self._event_bus.publish(
            PlanGenerated(
                source_id=self.id,
                goal_id=goal.goal_id,
                policy=policy,
                planning_time_ms=elapsed_ms,
                timestamp=time.time(),
            )
        )

        logger.debug(
            "BDIAgent %s: new intention formed with %d actions (%.1f ms)",
            self.id,
            policy.size,
            elapsed_ms,
        )

        return policy

    def select_action(self, policy: PolicySpec) -> ActionSpec:
        """Select the next action from the intention.

        For deterministic intentions, actions are consumed sequentially.
        For stochastic intentions, an action is sampled from the
        distribution.  Falls back to a ``noop`` if the intention is
        empty or exhausted.

        Parameters
        ----------
        policy:
            The policy (intention) to select from.

        Returns
        -------
        ActionSpec
            The selected action.
        """
        self._transition_to(AgentState.ACTING)

        if not policy.actions:
            logger.warning("BDIAgent %s: empty intention, selecting noop", self.id)
            return ActionSpec(name="noop")

        if policy.is_stochastic and policy.probabilities:
            # Stochastic: sample according to probability distribution
            idx = int(
                np.random.choice(len(policy.actions), p=policy.probabilities)
            )
            action = policy.actions[idx]
        else:
            # Deterministic: consume actions sequentially
            if self._intention_index < len(policy.actions):
                action = policy.actions[self._intention_index]
                self._intention_index += 1
            else:
                # Intention exhausted
                logger.debug(
                    "BDIAgent %s: intention exhausted, selecting last action",
                    self.id,
                )
                action = policy.actions[-1]

        # Update beliefs with the action taken
        self._beliefs["_last_action_name"] = action.name
        self._beliefs["_last_action_id"] = action.action_id

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
        """Execute one full BDI perceive-evaluate-revise-plan-act cycle.

        Orchestrates all phases in sequence.  After acting, the agent
        transitions to REFLECTING and then back to IDLE.

        Parameters
        ----------
        env:
            The environment to interact with.

        Returns
        -------
        ActionSpec
            The action that was selected and should be applied.
        """
        t0 = time.monotonic()

        # Perceive: update beliefs
        state = self.perceive(env)

        # Evaluate: beliefs vs active desire
        goal = self.current_goal
        delta = self.evaluate(goal, state)

        # Revise: reconsider desires and possibly switch
        new_goal = self.revise_goal(goal, state, delta, self._constraints)
        active_goal = new_goal if new_goal is not None else goal

        # Plan: form or reuse intention
        policy = self.plan(active_goal, state)

        # Act: select action from intention
        action = self.select_action(policy)

        # Reflect and return to idle
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

    # -- lifecycle hooks (overridden from BaseAgent) ------------------------

    def on_goal_revised(self, revision: GoalRevision) -> None:
        """Update beliefs with the revision information."""
        self._beliefs["_last_revision_reason"] = revision.reason
        self._beliefs["_last_revision_from"] = revision.previous_goal_id
        self._beliefs["_last_revision_to"] = revision.new_goal_id
        logger.info(
            "BDIAgent %s: goal revised (%s -> %s), reason: %s",
            self.id,
            revision.previous_goal_id,
            revision.new_goal_id,
            revision.reason,
        )

    def on_constraint_violated(self, violation: str) -> None:
        """Record constraint violation in beliefs."""
        self._beliefs["_last_constraint_violation"] = violation
        logger.warning(
            "BDIAgent %s: constraint violated: %s",
            self.id,
            violation,
        )

    def on_reflection_triggered(self) -> None:
        """Update beliefs with reflection metadata."""
        self._beliefs["_reflection_step"] = self.step_count
        self._beliefs["_reflection_score"] = (
            self._last_eval.score if self._last_eval else 0.0
        )
        logger.debug(
            "BDIAgent %s: reflection at step %d (score=%.3f, "
            "desires=%d, beliefs=%d)",
            self.id,
            self.step_count,
            self._last_eval.score if self._last_eval else 0.0,
            len(self._desires),
            len(self._beliefs),
        )

    # -- dunder helpers -----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BDIAgent("
            f"id={self.id!r}, "
            f"state={self._state.value}, "
            f"goal={self.current_goal.goal_id!r}, "
            f"beliefs={len(self._beliefs)}, "
            f"desires={len(self._desires)}, "
            f"has_intention={self._intentions is not None}, "
            f"steps={self._step_count})"
        )
