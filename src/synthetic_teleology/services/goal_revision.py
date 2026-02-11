"""Goal revision strategies for the Synthetic Teleology framework.

Implements Strategy + Chain of Responsibility patterns for goal updating.
Each updater decides whether a goal needs revision given the current state
and evaluation signal, and if so, produces a new goal with a revised
objective vector.

Classes
-------
BaseGoalUpdater
    Abstract base class for all updaters.
ThresholdUpdater
    Revises when |eval_signal.score| exceeds a threshold.
GradientUpdater
    Gradient descent on goal objective using dimension scores.
GoalUpdaterChain
    Chain of Responsibility -- first non-None result wins.
HierarchicalUpdater
    Updates considering parent goal alignment in a GoalTree.
UncertaintyAwareUpdater
    Active inference style -- revises when confidence is low.
ConstrainedUpdater
    Validates proposed revisions against a ConstraintSet.
LLMGoalEditor
    LLM-based goal editing via a provider interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction, RevisionReason
from synthetic_teleology.domain.values import (
    EvalSignal,
    StateSnapshot,
)

if TYPE_CHECKING:
    from synthetic_teleology.domain.aggregates import ConstraintSet, GoalTree

logger = logging.getLogger(__name__)


# ===================================================================== #
#  Base Goal Updater (ABC)                                               #
# ===================================================================== #


class BaseGoalUpdater(ABC):
    """Abstract base class for goal-revision strategies.

    Each updater examines the goal, state, evaluation signal, and constraints,
    and either returns a *new* revised goal or ``None`` if no revision is needed.
    """

    @abstractmethod
    def update(
        self,
        goal: Goal,
        state: StateSnapshot,
        eval_signal: EvalSignal,
        constraints: ConstraintSet | None = None,
    ) -> Goal | None:
        """Attempt to revise the goal.

        Parameters
        ----------
        goal:
            The current goal entity.
        state:
            The current perceived state.
        eval_signal:
            The evaluation signal Delta(G_t, S_t).
        constraints:
            Optional constraint set to validate the proposed revision against.

        Returns
        -------
        Goal | None
            A new goal with revised objective, or ``None`` if no revision
            is warranted.
        """


# ===================================================================== #
#  Threshold Updater                                                     #
# ===================================================================== #


class ThresholdUpdater(BaseGoalUpdater):
    """Revise the goal when the evaluation score magnitude exceeds a threshold.

    Moves the objective vector toward the current state by a ``learning_rate``,
    blending old objective values with observed state values.

    Parameters
    ----------
    threshold:
        Minimum |score| to trigger revision. Defaults to 0.5.
    learning_rate:
        Blending factor in (0, 1]. Higher values move the goal faster
        toward the observed state. Defaults to 0.1.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        learning_rate: float = 0.1,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        self._threshold = threshold
        self._learning_rate = learning_rate

    def update(
        self,
        goal: Goal,
        state: StateSnapshot,
        eval_signal: EvalSignal,
        constraints: ConstraintSet | None = None,
    ) -> Goal | None:
        """Revise if |eval_signal.score| exceeds threshold."""
        if goal.objective is None:
            return None

        if eval_signal.magnitude <= self._threshold:
            return None

        if goal.objective.dimension != state.dimension:
            logger.warning(
                "ThresholdUpdater: dimension mismatch goal=%d state=%d",
                goal.objective.dimension,
                state.dimension,
            )
            return None

        # Blend objective toward state
        lr = self._learning_rate
        new_values = tuple(
            g + lr * (s - g)
            for g, s in zip(goal.objective.values, state.values)
        )

        new_objective = goal.objective.with_values(new_values)
        new_goal, _ = goal.revise(
            new_objective=new_objective,
            reason=RevisionReason.THRESHOLD_EXCEEDED.value,
            eval_signal=eval_signal,
        )
        return new_goal


# ===================================================================== #
#  Gradient Updater                                                      #
# ===================================================================== #


class GradientUpdater(BaseGoalUpdater):
    """Gradient descent on the goal objective using per-dimension eval scores.

    The dimension scores of the eval signal are treated as the negative
    gradient: dimensions with low scores need larger adjustments.

    Parameters
    ----------
    learning_rate:
        Step size for gradient descent. Defaults to 0.05.
    min_gradient_norm:
        Minimum L2 norm of the gradient to trigger an update.
        Prevents negligible revisions. Defaults to 0.01.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        min_gradient_norm: float = 0.01,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        self._learning_rate = learning_rate
        self._min_gradient_norm = min_gradient_norm

    def update(
        self,
        goal: Goal,
        state: StateSnapshot,
        eval_signal: EvalSignal,
        constraints: ConstraintSet | None = None,
    ) -> Goal | None:
        """Update the goal objective via gradient descent on dimension scores."""
        if goal.objective is None:
            return None

        if not eval_signal.dimension_scores:
            return None

        if len(eval_signal.dimension_scores) != goal.objective.dimension:
            logger.warning(
                "GradientUpdater: dimension mismatch eval_dims=%d goal_dims=%d",
                len(eval_signal.dimension_scores),
                goal.objective.dimension,
            )
            return None

        # Gradient: for each dimension, the deficit is (1 - dim_score)
        # which indicates how far we are from perfect alignment.
        # Move the objective toward the state proportionally.
        gradient = np.array(
            [1.0 - s for s in eval_signal.dimension_scores], dtype=np.float64
        )

        grad_norm = float(np.linalg.norm(gradient))
        if grad_norm < self._min_gradient_norm:
            return None

        # Compute state-relative direction for gradient step
        goal_arr = np.array(goal.objective.values, dtype=np.float64)
        state_arr = np.array(state.values, dtype=np.float64)

        # Direction vector: state - goal, scaled by gradient magnitude per dim
        direction = state_arr - goal_arr
        step = self._learning_rate * gradient * np.sign(direction)

        # Apply direction-aware adjustments
        new_values_arr = goal_arr.copy()
        for i, d in enumerate(goal.objective.directions):
            if d == Direction.MAXIMIZE:
                # For maximize: increase goal if state suggests we can go higher
                new_values_arr[i] += self._learning_rate * gradient[i]
            elif d == Direction.MINIMIZE:
                # For minimize: decrease goal if state suggests we can go lower
                new_values_arr[i] -= self._learning_rate * gradient[i]
            else:
                # APPROACH / MAINTAIN: move toward state
                new_values_arr[i] += step[i]

        new_values = tuple(float(v) for v in new_values_arr)
        new_objective = goal.objective.with_values(new_values)

        new_goal, _ = goal.revise(
            new_objective=new_objective,
            reason=RevisionReason.GRADIENT_DESCENT.value,
            eval_signal=eval_signal,
        )
        return new_goal


# ===================================================================== #
#  Goal Updater Chain (Chain of Responsibility)                          #
# ===================================================================== #


class GoalUpdaterChain(BaseGoalUpdater):
    """Chain of Responsibility: tries updaters in order, first non-None wins.

    Parameters
    ----------
    updaters:
        Ordered sequence of updaters to try.
    """

    def __init__(self, updaters: Sequence[BaseGoalUpdater]) -> None:
        if not updaters:
            raise ValueError("GoalUpdaterChain requires at least one updater")
        self._updaters = list(updaters)

    @property
    def updaters(self) -> list[BaseGoalUpdater]:
        """Return the ordered list of updaters in the chain."""
        return list(self._updaters)

    def update(
        self,
        goal: Goal,
        state: StateSnapshot,
        eval_signal: EvalSignal,
        constraints: ConstraintSet | None = None,
    ) -> Goal | None:
        """Try each updater in order; return the first non-None result."""
        for updater in self._updaters:
            try:
                result = updater.update(goal, state, eval_signal, constraints)
                if result is not None:
                    logger.debug(
                        "GoalUpdaterChain: revision by %s",
                        type(updater).__name__,
                    )
                    return result
            except Exception:
                logger.exception(
                    "GoalUpdaterChain: error in %s", type(updater).__name__
                )
                continue
        return None


# ===================================================================== #
#  Hierarchical Updater                                                  #
# ===================================================================== #


class HierarchicalUpdater(BaseGoalUpdater):
    """Update considering parent goal alignment in a GoalTree.

    When revising a child goal, this updater adds a regularization term
    that keeps the child's objective aligned with its parent's objective.

    Parameters
    ----------
    goal_tree:
        The goal tree containing the hierarchical relationships.
    inner:
        The base updater to compute the raw revision.
    regularization_strength:
        Weight of the parent-alignment penalty in [0, 1].
        0.0 = no regularization; 1.0 = fully driven by parent.
        Defaults to 0.3.
    """

    def __init__(
        self,
        goal_tree: GoalTree,
        inner: BaseGoalUpdater,
        regularization_strength: float = 0.3,
    ) -> None:
        if not 0.0 <= regularization_strength <= 1.0:
            raise ValueError(
                f"regularization_strength must be in [0, 1], "
                f"got {regularization_strength}"
            )
        self._goal_tree = goal_tree
        self._inner = inner
        self._reg_strength = regularization_strength

    def update(
        self,
        goal: Goal,
        state: StateSnapshot,
        eval_signal: EvalSignal,
        constraints: ConstraintSet | None = None,
    ) -> Goal | None:
        """Revise goal with parent-alignment regularization."""
        # Get raw revision from inner updater
        raw_revision = self._inner.update(goal, state, eval_signal, constraints)
        if raw_revision is None:
            return None

        if raw_revision.objective is None:
            return raw_revision

        # Find parent goal
        parent = self._find_parent(goal)
        if parent is None or parent.objective is None:
            return raw_revision

        if parent.objective.dimension != raw_revision.objective.dimension:
            logger.warning(
                "HierarchicalUpdater: parent dimension (%d) != child dimension (%d)",
                parent.objective.dimension,
                raw_revision.objective.dimension,
            )
            return raw_revision

        # Apply regularization: blend revised values toward parent's objective
        reg = self._reg_strength
        new_values = tuple(
            (1 - reg) * child_v + reg * parent_v
            for child_v, parent_v in zip(
                raw_revision.objective.values, parent.objective.values
            )
        )
        regularized_objective = raw_revision.objective.with_values(new_values)

        # Create a new goal with the regularized objective
        # We need to work off the original goal (not raw_revision which already
        # consumed the revise() call on the original)
        regularized_goal = Goal(
            name=raw_revision.name,
            description=raw_revision.description,
            objective=regularized_objective,
            parent_id=raw_revision.parent_id,
            version=raw_revision.version,
            metadata={
                **dict(raw_revision.metadata),
                "hierarchical_regularization": reg,
                "parent_goal_id": parent.goal_id,
            },
        )
        return regularized_goal

    def _find_parent(self, goal: Goal) -> Goal | None:
        """Look up the parent goal in the tree."""
        if goal.parent_id is None:
            return None
        try:
            return self._goal_tree.get(goal.parent_id)
        except (KeyError, AttributeError):
            logger.debug(
                "HierarchicalUpdater: parent %s not found in tree",
                goal.parent_id,
            )
            return None


# ===================================================================== #
#  Uncertainty-Aware Updater                                             #
# ===================================================================== #


class UncertaintyAwareUpdater(BaseGoalUpdater):
    """Active-inference-style updater: revises when confidence is *low*.

    In the active inference framework, the agent should act to reduce
    uncertainty.  This updater triggers goal revision specifically when the
    evaluation confidence is below a threshold, treating low confidence as
    a signal that the goal may be poorly specified or the environment poorly
    understood.

    Parameters
    ----------
    confidence_threshold:
        Below this confidence level, revision is triggered. Defaults to 0.4.
    adaptation_rate:
        How aggressively to adapt when uncertain. Defaults to 0.15.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.4,
        adaptation_rate: float = 0.15,
    ) -> None:
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in (0, 1], got {confidence_threshold}"
            )
        if not 0.0 < adaptation_rate <= 1.0:
            raise ValueError(
                f"adaptation_rate must be in (0, 1], got {adaptation_rate}"
            )
        self._confidence_threshold = confidence_threshold
        self._adaptation_rate = adaptation_rate

    def update(
        self,
        goal: Goal,
        state: StateSnapshot,
        eval_signal: EvalSignal,
        constraints: ConstraintSet | None = None,
    ) -> Goal | None:
        """Revise the goal when the eval signal confidence is below threshold."""
        if goal.objective is None:
            return None

        if eval_signal.confidence >= self._confidence_threshold:
            return None

        if goal.objective.dimension != state.dimension:
            return None

        # Under uncertainty, move the goal toward the observed state
        # The adaptation rate is scaled by the uncertainty magnitude
        uncertainty = 1.0 - eval_signal.confidence
        effective_rate = self._adaptation_rate * uncertainty

        new_values = tuple(
            g + effective_rate * (s - g)
            for g, s in zip(goal.objective.values, state.values)
        )

        new_objective = goal.objective.with_values(new_values)
        new_goal, _ = goal.revise(
            new_objective=new_objective,
            reason=RevisionReason.UNCERTAINTY_REDUCTION.value,
            eval_signal=eval_signal,
        )
        return new_goal


# ===================================================================== #
#  Constrained Updater                                                   #
# ===================================================================== #


class ConstrainedUpdater(BaseGoalUpdater):
    """Updater that validates proposed revisions against a ConstraintSet.

    Wraps another updater and rejects revisions that violate the constraint
    set.  If the revision is rejected, ``None`` is returned.

    Parameters
    ----------
    inner:
        The updater whose proposed revision is validated.
    constraint_set:
        The constraint set to validate against.
    """

    def __init__(
        self,
        inner: BaseGoalUpdater,
        constraint_set: ConstraintSet,
    ) -> None:
        self._inner = inner
        self._constraint_set = constraint_set

    def update(
        self,
        goal: Goal,
        state: StateSnapshot,
        eval_signal: EvalSignal,
        constraints: ConstraintSet | None = None,
    ) -> Goal | None:
        """Propose a revision and validate it against constraints.

        Uses the ``constraint_set`` provided at construction time,
        ignoring the ``constraints`` parameter (which is part of the
        standard interface).
        """
        proposed = self._inner.update(goal, state, eval_signal, constraints)
        if proposed is None:
            return None

        # Validate the proposed revision against the constraint set
        if not self._validate_against_constraints(proposed, state):
            logger.info(
                "ConstrainedUpdater: rejected revision for goal %s "
                "(constraint violation)",
                goal.goal_id,
            )
            return None

        return proposed

    def _validate_against_constraints(
        self,
        proposed_goal: Goal,
        state: StateSnapshot,
    ) -> bool:
        """Check whether the proposed goal satisfies all active constraints.

        Iterates over active constraints in the set and validates that:
        1. For HARD constraints: the proposed objective values fall within
           any specified bounds.
        2. For BOUNDARY constraints: values respect upper/lower limits.
        3. For SOFT constraints: violations are logged but allowed.
        """
        try:
            active = self._constraint_set.active_constraints
        except AttributeError:
            # If constraint_set doesn't have active_constraints, try iteration
            try:
                active = list(self._constraint_set)
            except TypeError:
                logger.warning("ConstrainedUpdater: cannot iterate constraint_set")
                return True

        if proposed_goal.objective is None:
            return True

        from synthetic_teleology.domain.enums import ConstraintType

        for constraint in active:
            params = constraint.spec.parameters if hasattr(constraint, "spec") else {}
            ctype = (
                constraint.constraint_type
                if hasattr(constraint, "constraint_type")
                else None
            )

            # Check bounds constraints
            lower = params.get("lower_bounds")
            upper = params.get("upper_bounds")

            if lower is not None:
                for i, (val, lb) in enumerate(
                    zip(proposed_goal.objective.values, lower)
                ):
                    if val < lb and ctype in (
                        ConstraintType.HARD,
                        ConstraintType.BOUNDARY,
                    ):
                        logger.debug(
                            "ConstrainedUpdater: dim %d value %.4f < lower bound %.4f",
                            i,
                            val,
                            lb,
                        )
                        return False

            if upper is not None:
                for i, (val, ub) in enumerate(
                    zip(proposed_goal.objective.values, upper)
                ):
                    if val > ub and ctype in (
                        ConstraintType.HARD,
                        ConstraintType.BOUNDARY,
                    ):
                        logger.debug(
                            "ConstrainedUpdater: dim %d value %.4f > upper bound %.4f",
                            i,
                            val,
                            ub,
                        )
                        return False

        return True


# ===================================================================== #
#  LLM Goal Editor                                                       #
# ===================================================================== #


class LLMGoalEditor(BaseGoalUpdater):
    """LLM-based goal editing via a provider interface.

    Uses an LLM provider to propose goal revisions based on the
    evaluation signal, current state, and goal context.

    Parameters
    ----------
    provider:
        An LLM provider with a ``generate(prompt: str, **kwargs) -> str``
        method.
    system_prompt:
        Optional system-level instructions for goal editing.
    temperature:
        Sampling temperature for the LLM. Defaults to 0.4.
    """

    def __init__(
        self,
        provider: Any,
        system_prompt: str = "",
        temperature: float = 0.4,
    ) -> None:
        self._provider = provider
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._temperature = temperature

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are a goal-revision expert for a teleological AI agent. "
            "Given the current goal, state, and evaluation signal, propose "
            "revised objective values that better align the goal with reality. "
            "Respond with a JSON object: "
            '{"revised_values": [<list of floats>], '
            '"reason": "<brief explanation>", '
            '"confidence": <float in [0,1]>}.'
        )

    def update(
        self,
        goal: Goal,
        state: StateSnapshot,
        eval_signal: EvalSignal,
        constraints: ConstraintSet | None = None,
    ) -> Goal | None:
        """Use the LLM to propose a goal revision."""
        if goal.objective is None:
            return None

        prompt = self._build_prompt(goal, state, eval_signal)

        try:
            response = self._provider.generate(
                prompt,
                system_prompt=self._system_prompt,
                temperature=self._temperature,
            )
            return self._parse_and_apply(goal, eval_signal, response)
        except Exception as exc:
            logger.warning("LLMGoalEditor: provider call failed: %s", exc)
            return None

    def _build_prompt(
        self,
        goal: Goal,
        state: StateSnapshot,
        eval_signal: EvalSignal,
    ) -> str:
        """Build the revision prompt."""
        obj_repr = "None"
        if goal.objective is not None:
            obj_repr = (
                f"values={list(goal.objective.values)}, "
                f"directions=[{', '.join(d.value for d in goal.objective.directions)}]"
            )

        dim_scores_repr = (
            list(eval_signal.dimension_scores)
            if eval_signal.dimension_scores
            else "N/A"
        )

        return (
            f"## Goal Revision Request\n\n"
            f"**Goal**: {goal.name} (v{goal.version})\n"
            f"  Description: {goal.description}\n"
            f"  Objective: {obj_repr}\n\n"
            f"**Current State**: values={list(state.values)}\n\n"
            f"**Evaluation Signal**:\n"
            f"  Score: {eval_signal.score:.4f}\n"
            f"  Confidence: {eval_signal.confidence:.4f}\n"
            f"  Dimension Scores: {dim_scores_repr}\n"
            f"  Explanation: {eval_signal.explanation}\n\n"
            f"Propose revised objective values (same dimensionality). "
            f"Respond with a JSON object containing revised_values, reason, "
            f"and confidence."
        )

    def _parse_and_apply(
        self,
        goal: Goal,
        eval_signal: EvalSignal,
        response: str,
    ) -> Goal | None:
        """Parse LLM response and create the revised goal."""
        import json

        cleaned = response.strip()
        if "```json" in cleaned:
            start = cleaned.index("```json") + 7
            end = cleaned.index("```", start)
            cleaned = cleaned[start:end].strip()
        elif "```" in cleaned:
            start = cleaned.index("```") + 3
            end = cleaned.index("```", start)
            cleaned = cleaned[start:end].strip()

        try:
            data = json.loads(cleaned)
            revised_values = data.get("revised_values")
            reason = data.get("reason", "LLM-proposed revision")

            if revised_values is None or not isinstance(revised_values, list):
                logger.warning("LLMGoalEditor: no valid revised_values in response")
                return None

            if goal.objective is None:
                return None

            if len(revised_values) != goal.objective.dimension:
                logger.warning(
                    "LLMGoalEditor: dimension mismatch in revised_values "
                    "(expected %d, got %d)",
                    goal.objective.dimension,
                    len(revised_values),
                )
                return None

            new_values = tuple(float(v) for v in revised_values)
            new_objective = goal.objective.with_values(new_values)

            new_goal, _ = goal.revise(
                new_objective=new_objective,
                reason=f"{RevisionReason.LLM_CRITIQUE.value}: {reason}",
                eval_signal=eval_signal,
            )
            return new_goal

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("LLMGoalEditor: failed to parse response: %s", exc)
            return None
