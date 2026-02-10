"""Agentic loop implementation for the Synthetic Teleology framework.

Implements the Template Method pattern for the core teleological loop:
perceive -> evaluate -> revise -> check constraints -> plan -> filter ->
act -> transition -> emit event -> check stop.

Classes
-------
RunResult
    Dataclass capturing the outcome of a loop run.
BaseAgenticLoop
    Abstract base with the invariant loop algorithm.
SyncAgenticLoop
    Synchronous implementation.
AsyncAgenticLoop
    Asynchronous implementation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Sequence

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import AgentState, GoalStatus
from synthetic_teleology.domain.values import (
    ActionSpec,
    EvalSignal,
    PolicySpec,
    StateSnapshot,
)
from synthetic_teleology.services.constraint_engine import ConstraintPipeline, PolicyFilter
from synthetic_teleology.services.evaluation import BaseEvaluator
from synthetic_teleology.services.goal_revision import BaseGoalUpdater
from synthetic_teleology.services.planning import BasePlanner

if TYPE_CHECKING:
    from synthetic_teleology.agents.base import BaseAgent
    from synthetic_teleology.domain.aggregates import ConstraintSet
    from synthetic_teleology.domain.events import DomainEvent, LoopStepCompleted
    from synthetic_teleology.environments.base import BaseEnvironment
    from synthetic_teleology.infrastructure.config import LoopConfig
    from synthetic_teleology.infrastructure.event_bus import AsyncEventBus, EventBus

logger = logging.getLogger(__name__)


# ===================================================================== #
#  Stop Reason Enum                                                      #
# ===================================================================== #


class StopReason(Enum):
    """Reason the agentic loop terminated."""

    MAX_STEPS = "max_steps"
    GOAL_ACHIEVED = "goal_achieved"
    GOAL_ABANDONED = "goal_abandoned"
    CONSTRAINT_VIOLATION = "constraint_violation"
    ERROR = "error"
    MANUAL = "manual"
    EMPTY_POLICY = "empty_policy"


# ===================================================================== #
#  Run Result                                                            #
# ===================================================================== #


@dataclass
class RunResult:
    """Captures the outcome of an agentic loop run.

    Attributes
    ----------
    steps_completed:
        Number of loop iterations completed.
    final_goal:
        The goal state when the loop terminated.
    final_state:
        The last observed state snapshot.
    events:
        All domain events emitted during the run.
    stopped_reason:
        Why the loop terminated.
    elapsed_seconds:
        Wall-clock time of the run.
    metadata:
        Additional run metadata (per-step signals, etc.).
    """

    steps_completed: int = 0
    final_goal: Goal | None = None
    final_state: StateSnapshot | None = None
    events: list[Any] = field(default_factory=list)
    stopped_reason: StopReason = StopReason.MAX_STEPS
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ===================================================================== #
#  Base Agentic Loop (ABC / Template Method)                             #
# ===================================================================== #


class BaseAgenticLoop(ABC):
    """Abstract base class implementing the invariant teleological loop.

    The loop algorithm (template method) is:

    1. **Perceive**: observe the environment -> StateSnapshot
    2. **Evaluate**: compute Delta(G_t, S_t) -> EvalSignal
    3. **Revise**: optionally update G_t based on the signal
    4. **Check constraints**: validate current state
    5. **Plan**: generate pi_t from G_t and S_t
    6. **Filter**: remove constraint-violating actions from pi_t
    7. **Act**: execute the filtered policy
    8. **Transition**: environment transitions to S_{t+1}
    9. **Emit event**: publish a LoopStepCompleted event
    10. **Check stop**: determine whether to continue

    Subclasses implement the synchronous or asynchronous variants.

    Parameters
    ----------
    evaluator:
        The evaluation strategy.
    goal_updater:
        The goal-revision strategy.
    planner:
        The planning strategy.
    constraint_pipeline:
        The constraint checking pipeline.
    policy_filter:
        The policy filter (uses the constraint pipeline).
    max_steps:
        Maximum number of loop iterations. Defaults to 100.
    goal_achieved_threshold:
        Eval score above which the goal is considered achieved.
        Defaults to 0.9.
    stop_on_empty_policy:
        Whether to stop when the policy is empty after filtering.
        Defaults to ``True``.
    """

    def __init__(
        self,
        evaluator: BaseEvaluator,
        goal_updater: BaseGoalUpdater,
        planner: BasePlanner,
        constraint_pipeline: ConstraintPipeline,
        policy_filter: PolicyFilter | None = None,
        max_steps: int = 100,
        goal_achieved_threshold: float = 0.9,
        stop_on_empty_policy: bool = True,
    ) -> None:
        self._evaluator = evaluator
        self._goal_updater = goal_updater
        self._planner = planner
        self._constraint_pipeline = constraint_pipeline
        self._policy_filter = policy_filter or PolicyFilter(constraint_pipeline)
        self._max_steps = max_steps
        self._goal_achieved_threshold = goal_achieved_threshold
        self._stop_on_empty_policy = stop_on_empty_policy

    @property
    def evaluator(self) -> BaseEvaluator:
        """Return the evaluation strategy."""
        return self._evaluator

    @property
    def goal_updater(self) -> BaseGoalUpdater:
        """Return the goal-revision strategy."""
        return self._goal_updater

    @property
    def planner(self) -> BasePlanner:
        """Return the planning strategy."""
        return self._planner

    @property
    def constraint_pipeline(self) -> ConstraintPipeline:
        """Return the constraint pipeline."""
        return self._constraint_pipeline

    @property
    def max_steps(self) -> int:
        """Maximum number of steps."""
        return self._max_steps

    # -- Abstract hooks for environment / agent interaction ---------------

    @abstractmethod
    def _perceive(self) -> StateSnapshot:
        """Observe the current environment state."""

    @abstractmethod
    def _act(self, policy: PolicySpec, state: StateSnapshot) -> ActionSpec | None:
        """Execute the policy and return the executed action (if any)."""

    @abstractmethod
    def _transition(self, action: ActionSpec | None) -> None:
        """Trigger environment state transition after action execution."""

    @abstractmethod
    def _emit_event(self, step: int, goal: Goal, state: StateSnapshot,
                     eval_signal: EvalSignal, action: ActionSpec | None) -> Any:
        """Emit a loop step event. Returns the event object."""

    # -- Template method: the invariant loop algorithm --------------------

    def _should_stop(
        self,
        step: int,
        goal: Goal,
        eval_signal: EvalSignal,
        policy: PolicySpec,
    ) -> StopReason | None:
        """Check stopping conditions. Returns a reason or None to continue."""
        if step >= self._max_steps:
            return StopReason.MAX_STEPS

        if goal.status == GoalStatus.ACHIEVED:
            return StopReason.GOAL_ACHIEVED

        if goal.status == GoalStatus.ABANDONED:
            return StopReason.GOAL_ABANDONED

        if eval_signal.score >= self._goal_achieved_threshold:
            goal.achieve()
            return StopReason.GOAL_ACHIEVED

        if self._stop_on_empty_policy and policy.size == 0:
            return StopReason.EMPTY_POLICY

        return None

    def _check_constraints(
        self,
        goal: Goal,
        state: StateSnapshot,
    ) -> tuple[bool, list[str]]:
        """Run the constraint pipeline on the current state."""
        return self._constraint_pipeline.check_all(goal, state)

    def _filter_policy(
        self,
        policy: PolicySpec,
        goal: Goal,
        state: StateSnapshot,
    ) -> PolicySpec:
        """Filter the policy through the constraint pipeline."""
        return self._policy_filter.filter(policy, goal, state)


# ===================================================================== #
#  Synchronous Agentic Loop                                              #
# ===================================================================== #


class SyncAgenticLoop(BaseAgenticLoop):
    """Synchronous implementation of the teleological loop.

    Parameters
    ----------
    evaluator:
        The evaluation strategy.
    goal_updater:
        The goal-revision strategy.
    planner:
        The planning strategy.
    constraint_pipeline:
        The constraint checking pipeline.
    policy_filter:
        Optional policy filter.
    event_bus:
        Optional synchronous event bus for publishing loop events.
    max_steps:
        Maximum loop iterations.
    goal_achieved_threshold:
        Score threshold for goal achievement.
    stop_on_empty_policy:
        Stop when policy is empty after filtering.
    perceive_fn:
        Callable that returns a StateSnapshot (environment observation).
    act_fn:
        Callable that takes (PolicySpec, StateSnapshot) and returns the
        executed ActionSpec or None.
    transition_fn:
        Callable that takes an optional ActionSpec and triggers state
        transition.
    """

    def __init__(
        self,
        evaluator: BaseEvaluator,
        goal_updater: BaseGoalUpdater,
        planner: BasePlanner,
        constraint_pipeline: ConstraintPipeline,
        policy_filter: PolicyFilter | None = None,
        event_bus: EventBus | None = None,
        max_steps: int = 100,
        goal_achieved_threshold: float = 0.9,
        stop_on_empty_policy: bool = True,
        perceive_fn: Any = None,
        act_fn: Any = None,
        transition_fn: Any = None,
    ) -> None:
        super().__init__(
            evaluator=evaluator,
            goal_updater=goal_updater,
            planner=planner,
            constraint_pipeline=constraint_pipeline,
            policy_filter=policy_filter,
            max_steps=max_steps,
            goal_achieved_threshold=goal_achieved_threshold,
            stop_on_empty_policy=stop_on_empty_policy,
        )
        self._event_bus = event_bus
        self._perceive_fn = perceive_fn
        self._act_fn = act_fn
        self._transition_fn = transition_fn

    def _perceive(self) -> StateSnapshot:
        """Call the perceive function to get the current state."""
        if self._perceive_fn is None:
            raise RuntimeError(
                "SyncAgenticLoop: perceive_fn not set. Provide a callable "
                "that returns a StateSnapshot."
            )
        return self._perceive_fn()

    def _act(self, policy: PolicySpec, state: StateSnapshot) -> ActionSpec | None:
        """Execute the policy via the act function."""
        if self._act_fn is None:
            # Default: select first action from the policy
            if policy.size > 0:
                return policy.actions[0]
            return None
        return self._act_fn(policy, state)

    def _transition(self, action: ActionSpec | None) -> None:
        """Trigger state transition."""
        if self._transition_fn is not None:
            self._transition_fn(action)

    def _emit_event(
        self,
        step: int,
        goal: Goal,
        state: StateSnapshot,
        eval_signal: EvalSignal,
        action: ActionSpec | None,
    ) -> Any:
        """Emit a loop step event to the event bus."""
        event_data = {
            "step": step,
            "goal_id": goal.goal_id,
            "goal_status": goal.status.value,
            "eval_score": eval_signal.score,
            "eval_confidence": eval_signal.confidence,
            "action_name": action.name if action else None,
            "timestamp": time.time(),
        }

        if self._event_bus is not None:
            try:
                # Try to create a LoopStepCompleted event
                from synthetic_teleology.domain.events import LoopStepCompleted

                event = LoopStepCompleted(
                    step=step,
                    goal_id=goal.goal_id,
                    eval_score=eval_signal.score,
                    action_name=action.name if action else "",
                )
                self._event_bus.publish(event)
                return event
            except ImportError:
                logger.debug(
                    "SyncAgenticLoop: LoopStepCompleted not available, "
                    "skipping event emission"
                )
            except Exception as exc:
                logger.warning("SyncAgenticLoop: event emission failed: %s", exc)

        return event_data

    def run(self, goal: Goal) -> RunResult:
        """Execute the synchronous teleological loop.

        Parameters
        ----------
        goal:
            The initial goal to pursue.

        Returns
        -------
        RunResult
            The outcome of the loop run.
        """
        start_time = time.time()
        events: list[Any] = []
        step = 0
        current_goal = goal
        current_state: StateSnapshot | None = None
        last_eval: EvalSignal = EvalSignal(score=0.0, confidence=0.0)
        stop_reason = StopReason.MAX_STEPS

        try:
            for step in range(1, self._max_steps + 1):
                # 1. PERCEIVE
                current_state = self._perceive()
                logger.debug("Step %d: perceived state dim=%d", step, current_state.dimension)

                # 2. EVALUATE
                if self._evaluator.validate(current_goal, current_state):
                    last_eval = self._evaluator.evaluate(current_goal, current_state)
                else:
                    last_eval = EvalSignal(
                        score=0.0,
                        confidence=0.0,
                        explanation="Evaluator validation failed",
                    )
                logger.debug("Step %d: eval score=%.4f conf=%.4f",
                             step, last_eval.score, last_eval.confidence)

                # 3. REVISE (goal update)
                revised = self._goal_updater.update(
                    current_goal, current_state, last_eval
                )
                if revised is not None:
                    logger.info(
                        "Step %d: goal revised %s -> %s",
                        step,
                        current_goal.goal_id,
                        revised.goal_id,
                    )
                    current_goal = revised

                # 4. CHECK CONSTRAINTS (on current state)
                constraints_ok, violations = self._check_constraints(
                    current_goal, current_state
                )
                if not constraints_ok:
                    logger.warning(
                        "Step %d: constraint violations: %s", step, violations
                    )
                    # Continue but log -- hard constraint handling is in the filter

                # 5. PLAN
                policy = self._planner.plan(current_goal, current_state)
                logger.debug("Step %d: planned %d actions", step, policy.size)

                # 6. FILTER
                filtered_policy = self._filter_policy(
                    policy, current_goal, current_state
                )
                logger.debug(
                    "Step %d: filtered to %d actions", step, filtered_policy.size
                )

                # 7. CHECK STOP (pre-action)
                reason = self._should_stop(
                    step, current_goal, last_eval, filtered_policy
                )
                if reason is not None:
                    stop_reason = reason
                    # Still emit event before stopping
                    event = self._emit_event(
                        step, current_goal, current_state, last_eval, None
                    )
                    events.append(event)
                    break

                # 8. ACT
                executed_action = self._act(filtered_policy, current_state)
                logger.debug(
                    "Step %d: executed action=%s",
                    step,
                    executed_action.name if executed_action else "None",
                )

                # 9. TRANSITION
                self._transition(executed_action)

                # 10. EMIT EVENT
                event = self._emit_event(
                    step, current_goal, current_state, last_eval, executed_action
                )
                events.append(event)

            else:
                # Loop completed without break -> max steps reached
                stop_reason = StopReason.MAX_STEPS

        except Exception as exc:
            logger.exception("SyncAgenticLoop: error at step %d", step)
            stop_reason = StopReason.ERROR
            events.append({
                "error": str(exc),
                "step": step,
                "timestamp": time.time(),
            })

        elapsed = time.time() - start_time

        return RunResult(
            steps_completed=step,
            final_goal=current_goal,
            final_state=current_state,
            events=events,
            stopped_reason=stop_reason,
            elapsed_seconds=elapsed,
            metadata={
                "last_eval_score": last_eval.score,
                "last_eval_confidence": last_eval.confidence,
            },
        )


# ===================================================================== #
#  Asynchronous Agentic Loop                                             #
# ===================================================================== #


class AsyncAgenticLoop(BaseAgenticLoop):
    """Asynchronous implementation of the teleological loop.

    Parameters
    ----------
    evaluator:
        The evaluation strategy.
    goal_updater:
        The goal-revision strategy.
    planner:
        The planning strategy.
    constraint_pipeline:
        The constraint checking pipeline.
    policy_filter:
        Optional policy filter.
    event_bus:
        Optional async event bus for publishing loop events.
    max_steps:
        Maximum loop iterations.
    goal_achieved_threshold:
        Score threshold for goal achievement.
    stop_on_empty_policy:
        Stop when policy is empty after filtering.
    perceive_fn:
        Async callable that returns a StateSnapshot.
    act_fn:
        Async callable that takes (PolicySpec, StateSnapshot) and returns
        ActionSpec or None.
    transition_fn:
        Async callable that takes an optional ActionSpec and triggers
        state transition.
    step_delay:
        Optional delay in seconds between loop steps (for rate limiting).
    """

    def __init__(
        self,
        evaluator: BaseEvaluator,
        goal_updater: BaseGoalUpdater,
        planner: BasePlanner,
        constraint_pipeline: ConstraintPipeline,
        policy_filter: PolicyFilter | None = None,
        event_bus: AsyncEventBus | None = None,
        max_steps: int = 100,
        goal_achieved_threshold: float = 0.9,
        stop_on_empty_policy: bool = True,
        perceive_fn: Any = None,
        act_fn: Any = None,
        transition_fn: Any = None,
        step_delay: float = 0.0,
    ) -> None:
        super().__init__(
            evaluator=evaluator,
            goal_updater=goal_updater,
            planner=planner,
            constraint_pipeline=constraint_pipeline,
            policy_filter=policy_filter,
            max_steps=max_steps,
            goal_achieved_threshold=goal_achieved_threshold,
            stop_on_empty_policy=stop_on_empty_policy,
        )
        self._event_bus = event_bus
        self._perceive_fn = perceive_fn
        self._act_fn = act_fn
        self._transition_fn = transition_fn
        self._step_delay = step_delay

    def _perceive(self) -> StateSnapshot:
        """Synchronous fallback -- not used in async loop."""
        raise NotImplementedError("Use async run()")

    def _act(self, policy: PolicySpec, state: StateSnapshot) -> ActionSpec | None:
        """Synchronous fallback -- not used in async loop."""
        raise NotImplementedError("Use async run()")

    def _transition(self, action: ActionSpec | None) -> None:
        """Synchronous fallback -- not used in async loop."""
        raise NotImplementedError("Use async run()")

    def _emit_event(
        self,
        step: int,
        goal: Goal,
        state: StateSnapshot,
        eval_signal: EvalSignal,
        action: ActionSpec | None,
    ) -> Any:
        """Synchronous fallback -- not used in async loop."""
        raise NotImplementedError("Use async run()")

    async def _async_perceive(self) -> StateSnapshot:
        """Call the async perceive function."""
        if self._perceive_fn is None:
            raise RuntimeError(
                "AsyncAgenticLoop: perceive_fn not set. Provide an async callable "
                "that returns a StateSnapshot."
            )
        result = self._perceive_fn()
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _async_act(
        self, policy: PolicySpec, state: StateSnapshot
    ) -> ActionSpec | None:
        """Execute the policy via the async act function."""
        if self._act_fn is None:
            if policy.size > 0:
                return policy.actions[0]
            return None
        result = self._act_fn(policy, state)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _async_transition(self, action: ActionSpec | None) -> None:
        """Trigger async state transition."""
        if self._transition_fn is not None:
            result = self._transition_fn(action)
            if asyncio.iscoroutine(result):
                await result

    async def _async_emit_event(
        self,
        step: int,
        goal: Goal,
        state: StateSnapshot,
        eval_signal: EvalSignal,
        action: ActionSpec | None,
    ) -> Any:
        """Emit a loop step event to the async event bus."""
        event_data = {
            "step": step,
            "goal_id": goal.goal_id,
            "goal_status": goal.status.value,
            "eval_score": eval_signal.score,
            "eval_confidence": eval_signal.confidence,
            "action_name": action.name if action else None,
            "timestamp": time.time(),
        }

        if self._event_bus is not None:
            try:
                from synthetic_teleology.domain.events import LoopStepCompleted

                event = LoopStepCompleted(
                    step=step,
                    goal_id=goal.goal_id,
                    eval_score=eval_signal.score,
                    action_name=action.name if action else "",
                )
                await self._event_bus.publish(event)
                return event
            except ImportError:
                logger.debug(
                    "AsyncAgenticLoop: LoopStepCompleted not available"
                )
            except Exception as exc:
                logger.warning("AsyncAgenticLoop: event emission failed: %s", exc)

        return event_data

    async def run(self, goal: Goal) -> RunResult:
        """Execute the asynchronous teleological loop.

        Parameters
        ----------
        goal:
            The initial goal to pursue.

        Returns
        -------
        RunResult
            The outcome of the loop run.
        """
        start_time = time.time()
        events: list[Any] = []
        step = 0
        current_goal = goal
        current_state: StateSnapshot | None = None
        last_eval: EvalSignal = EvalSignal(score=0.0, confidence=0.0)
        stop_reason = StopReason.MAX_STEPS

        try:
            for step in range(1, self._max_steps + 1):
                # Optional inter-step delay
                if self._step_delay > 0 and step > 1:
                    await asyncio.sleep(self._step_delay)

                # 1. PERCEIVE
                current_state = await self._async_perceive()
                logger.debug(
                    "Async step %d: perceived state dim=%d",
                    step,
                    current_state.dimension,
                )

                # 2. EVALUATE
                if self._evaluator.validate(current_goal, current_state):
                    last_eval = self._evaluator.evaluate(current_goal, current_state)
                else:
                    last_eval = EvalSignal(
                        score=0.0,
                        confidence=0.0,
                        explanation="Evaluator validation failed",
                    )
                logger.debug(
                    "Async step %d: eval score=%.4f conf=%.4f",
                    step,
                    last_eval.score,
                    last_eval.confidence,
                )

                # 3. REVISE
                revised = self._goal_updater.update(
                    current_goal, current_state, last_eval
                )
                if revised is not None:
                    logger.info(
                        "Async step %d: goal revised %s -> %s",
                        step,
                        current_goal.goal_id,
                        revised.goal_id,
                    )
                    current_goal = revised

                # 4. CHECK CONSTRAINTS
                constraints_ok, violations = self._check_constraints(
                    current_goal, current_state
                )
                if not constraints_ok:
                    logger.warning(
                        "Async step %d: constraint violations: %s",
                        step,
                        violations,
                    )

                # 5. PLAN
                policy = self._planner.plan(current_goal, current_state)
                logger.debug(
                    "Async step %d: planned %d actions", step, policy.size
                )

                # 6. FILTER
                filtered_policy = self._filter_policy(
                    policy, current_goal, current_state
                )
                logger.debug(
                    "Async step %d: filtered to %d actions",
                    step,
                    filtered_policy.size,
                )

                # 7. CHECK STOP
                reason = self._should_stop(
                    step, current_goal, last_eval, filtered_policy
                )
                if reason is not None:
                    stop_reason = reason
                    event = await self._async_emit_event(
                        step, current_goal, current_state, last_eval, None
                    )
                    events.append(event)
                    break

                # 8. ACT
                executed_action = await self._async_act(
                    filtered_policy, current_state
                )
                logger.debug(
                    "Async step %d: executed action=%s",
                    step,
                    executed_action.name if executed_action else "None",
                )

                # 9. TRANSITION
                await self._async_transition(executed_action)

                # 10. EMIT EVENT
                event = await self._async_emit_event(
                    step, current_goal, current_state, last_eval, executed_action
                )
                events.append(event)

            else:
                stop_reason = StopReason.MAX_STEPS

        except Exception as exc:
            logger.exception("AsyncAgenticLoop: error at step %d", step)
            stop_reason = StopReason.ERROR
            events.append({
                "error": str(exc),
                "step": step,
                "timestamp": time.time(),
            })

        elapsed = time.time() - start_time

        return RunResult(
            steps_completed=step,
            final_goal=current_goal,
            final_state=current_state,
            events=events,
            stopped_reason=stop_reason,
            elapsed_seconds=elapsed,
            metadata={
                "last_eval_score": last_eval.score,
                "last_eval_confidence": last_eval.confidence,
                "async": True,
            },
        )
