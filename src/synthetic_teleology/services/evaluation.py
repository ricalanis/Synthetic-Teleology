"""Evaluation strategies for the Synthetic Teleology framework.

Implements the Strategy pattern for computing evaluation signals
Delta(G_t, S_t) that measure how well the current state satisfies the goal.

Classes
-------
BaseEvaluator
    Abstract base class for all evaluation strategies.
NumericEvaluator
    Weighted Euclidean distance between goal objective and state.
CompositeEvaluator
    Aggregates multiple evaluators via weighted average.
ReflectiveEvaluator
    Decorator that tracks evaluation history and applies confidence adjustment.
LLMCriticEvaluator
    Placeholder for LLM-based evaluation via a provider interface.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import EvalSignal, ObjectiveVector, StateSnapshot

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ===================================================================== #
#  Base Evaluator (ABC)                                                  #
# ===================================================================== #


class BaseEvaluator(ABC):
    """Abstract base class for evaluation strategies.

    Subclasses must implement :meth:`evaluate`.  The default :meth:`validate`
    returns ``True``; override it when specific preconditions must hold.
    """

    @abstractmethod
    def evaluate(self, goal: Goal, state: StateSnapshot) -> EvalSignal:
        """Compute an evaluation signal for *goal* given *state*.

        Parameters
        ----------
        goal:
            The goal entity containing the target objective vector.
        state:
            The current state snapshot to evaluate against the goal.

        Returns
        -------
        EvalSignal
            A signal in [-1, 1] describing alignment of state to goal.
        """

    def validate(self, goal: Goal, state: StateSnapshot) -> bool:
        """Check whether evaluation is feasible for the given goal and state.

        The default implementation always returns ``True``.  Override this
        to enforce preconditions (e.g., dimension matching).
        """
        return True


# ===================================================================== #
#  Numeric Evaluator                                                     #
# ===================================================================== #


class NumericEvaluator(BaseEvaluator):
    """Compute evaluation via weighted Euclidean distance between goal and state.

    The overall score is normalized to [-1, 1]:
    *  1.0 = perfect match (distance is 0)
    * -1.0 = maximum divergence (distance >= ``max_distance``)

    Per-dimension scores follow the same convention, respecting each
    dimension's :class:`Direction`.

    Parameters
    ----------
    max_distance:
        The distance at which the score saturates to -1.  Defaults to 10.0.
        Distances beyond this value are clamped.
    """

    def __init__(self, max_distance: float = 10.0) -> None:
        if max_distance <= 0:
            raise ValueError(f"max_distance must be positive, got {max_distance}")
        self._max_distance = max_distance

    def validate(self, goal: Goal, state: StateSnapshot) -> bool:
        """Return ``True`` if goal has an objective and dimensions match."""
        if goal.objective is None:
            return False
        return goal.objective.dimension == state.dimension

    def evaluate(self, goal: Goal, state: StateSnapshot) -> EvalSignal:
        """Compute the weighted Euclidean evaluation signal.

        Raises
        ------
        ValueError
            If ``validate`` would return ``False``.
        """
        if not self.validate(goal, state):
            raise ValueError(
                "Cannot evaluate: goal objective is None or dimension mismatch "
                f"(goal={goal.objective.dimension if goal.objective else 'None'}, "
                f"state={state.dimension})"
            )

        objective: ObjectiveVector = goal.objective  # type: ignore[assignment]
        weights = objective.weights or tuple(1.0 for _ in objective.values)

        dimension_scores: list[float] = []
        for i, (goal_val, state_val, direction, w) in enumerate(
            zip(objective.values, state.values, objective.directions, weights)
        ):
            dim_score = self._score_dimension(goal_val, state_val, direction, w)
            dimension_scores.append(dim_score)

        # Overall score: weighted average of dimension scores
        total_weight = sum(weights)
        if total_weight == 0:
            overall = 0.0
        else:
            overall = sum(
                w * s for w, s in zip(weights, dimension_scores)
            ) / total_weight

        # Clamp to [-1, 1]
        overall = max(-1.0, min(1.0, overall))

        return EvalSignal(
            score=overall,
            dimension_scores=tuple(dimension_scores),
            confidence=1.0,
            explanation=(
                f"NumericEvaluator: overall={overall:.4f}, "
                f"max_dist={self._max_distance}"
            ),
        )

    def _score_dimension(
        self,
        goal_val: float,
        state_val: float,
        direction: Direction,
        weight: float,
    ) -> float:
        """Score a single dimension in [-1, 1].

        Scoring rules per direction:
        - APPROACH / MAINTAIN: 1 - 2*(|delta| / max_distance), clamped to [-1, 1]
        - MAXIMIZE: positive if state > goal, negative if below
        - MINIMIZE: positive if state < goal, negative if above
        """
        delta = state_val - goal_val
        abs_delta = abs(delta)

        if direction in (Direction.APPROACH, Direction.MAINTAIN):
            # Perfect at 0, worst at max_distance
            ratio = abs_delta / self._max_distance
            return max(-1.0, min(1.0, 1.0 - 2.0 * ratio))

        if direction == Direction.MAXIMIZE:
            # Positive when state exceeds goal target
            ratio = delta / self._max_distance
            return max(-1.0, min(1.0, ratio))

        if direction == Direction.MINIMIZE:
            # Positive when state is below goal target
            ratio = -delta / self._max_distance
            return max(-1.0, min(1.0, ratio))

        # Fallback: treat as APPROACH
        ratio = abs_delta / self._max_distance
        return max(-1.0, min(1.0, 1.0 - 2.0 * ratio))


# ===================================================================== #
#  Composite Evaluator                                                   #
# ===================================================================== #


class CompositeEvaluator(BaseEvaluator):
    """Aggregate multiple evaluators via weighted average.

    Parameters
    ----------
    evaluators:
        Ordered list of ``(evaluator, weight)`` pairs.
    """

    def __init__(
        self,
        evaluators: Sequence[tuple[BaseEvaluator, float]],
    ) -> None:
        if not evaluators:
            raise ValueError("CompositeEvaluator requires at least one evaluator")
        self._evaluators = list(evaluators)

    def validate(self, goal: Goal, state: StateSnapshot) -> bool:
        """True if at least one sub-evaluator can validate."""
        return any(ev.validate(goal, state) for ev, _ in self._evaluators)

    def evaluate(self, goal: Goal, state: StateSnapshot) -> EvalSignal:
        """Weighted average of all sub-evaluator signals.

        Sub-evaluators that fail validation are skipped.  If no evaluator
        produces a signal, a neutral signal (score=0, confidence=0) is returned.
        """
        scores: list[float] = []
        weights: list[float] = []
        confidences: list[float] = []
        all_dim_scores: list[tuple[float, ...]] = []
        explanations: list[str] = []

        for evaluator, weight in self._evaluators:
            if not evaluator.validate(goal, state):
                logger.debug(
                    "CompositeEvaluator: skipping %s (validation failed)",
                    type(evaluator).__name__,
                )
                continue
            try:
                signal = evaluator.evaluate(goal, state)
            except Exception:
                logger.exception(
                    "CompositeEvaluator: error in %s", type(evaluator).__name__
                )
                continue

            scores.append(signal.score)
            weights.append(weight)
            confidences.append(signal.confidence)
            if signal.dimension_scores:
                all_dim_scores.append(signal.dimension_scores)
            explanations.append(signal.explanation)

        if not scores:
            return EvalSignal(
                score=0.0,
                confidence=0.0,
                explanation="CompositeEvaluator: no sub-evaluator produced a signal",
            )

        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        weighted_confidence = (
            sum(c * w for c, w in zip(confidences, weights)) / total_weight
        )

        # Aggregate dimension scores: weighted average if all have the same length
        merged_dims: tuple[float, ...] = ()
        if all_dim_scores:
            dim_len = len(all_dim_scores[0])
            if all(len(ds) == dim_len for ds in all_dim_scores):
                merged = []
                for d in range(dim_len):
                    dim_weighted = sum(
                        ds[d] * w for ds, w in zip(all_dim_scores, weights)
                    ) / total_weight
                    merged.append(max(-1.0, min(1.0, dim_weighted)))
                merged_dims = tuple(merged)

        overall = max(-1.0, min(1.0, weighted_score))
        conf = max(0.0, min(1.0, weighted_confidence))

        return EvalSignal(
            score=overall,
            dimension_scores=merged_dims,
            confidence=conf,
            explanation=(
                f"CompositeEvaluator: weighted avg of {len(scores)} evaluators | "
                + " ; ".join(explanations)
            ),
        )


# ===================================================================== #
#  Reflective Evaluator (Decorator)                                      #
# ===================================================================== #


@dataclass
class _EvalHistoryEntry:
    """Internal record of a past evaluation."""

    timestamp: float
    score: float
    confidence: float


class ReflectiveEvaluator(BaseEvaluator):
    """Decorator that wraps another evaluator and adds self-model reflection.

    Tracks evaluation history, detects drift in evaluation scores, and
    adjusts confidence when the evaluator appears to be oscillating or
    trending unexpectedly.

    Parameters
    ----------
    inner:
        The wrapped evaluator that performs the actual computation.
    history_size:
        Maximum number of past evaluations to retain for drift detection.
    drift_threshold:
        Standard deviation of recent scores above which confidence is reduced.
    smoothing_factor:
        Exponential smoothing factor for drift detection (0, 1].
    """

    def __init__(
        self,
        inner: BaseEvaluator,
        history_size: int = 50,
        drift_threshold: float = 0.3,
        smoothing_factor: float = 0.2,
    ) -> None:
        self._inner = inner
        self._history: deque[_EvalHistoryEntry] = deque(maxlen=history_size)
        self._drift_threshold = drift_threshold
        self._smoothing_factor = smoothing_factor
        self._ema_score: float | None = None

    @property
    def inner(self) -> BaseEvaluator:
        """Return the wrapped evaluator."""
        return self._inner

    @property
    def history(self) -> list[_EvalHistoryEntry]:
        """Return a copy of the evaluation history."""
        return list(self._history)

    def validate(self, goal: Goal, state: StateSnapshot) -> bool:
        """Delegate validation to the inner evaluator."""
        return self._inner.validate(goal, state)

    def evaluate(self, goal: Goal, state: StateSnapshot) -> EvalSignal:
        """Evaluate via the inner evaluator, then apply reflective adjustments.

        The adjustment reduces confidence when evaluation scores show high
        variance (drift) over the recent history window.
        """
        raw_signal = self._inner.evaluate(goal, state)

        now = time.time()
        self._history.append(
            _EvalHistoryEntry(
                timestamp=now,
                score=raw_signal.score,
                confidence=raw_signal.confidence,
            )
        )

        # Update exponential moving average
        if self._ema_score is None:
            self._ema_score = raw_signal.score
        else:
            alpha = self._smoothing_factor
            self._ema_score = alpha * raw_signal.score + (1 - alpha) * self._ema_score

        # Compute drift as standard deviation of recent scores
        adjusted_confidence = raw_signal.confidence
        drift_info = ""

        if len(self._history) >= 3:
            recent_scores = [e.score for e in self._history]
            std_dev = float(np.std(recent_scores))
            mean_score = float(np.mean(recent_scores))

            if std_dev > self._drift_threshold:
                # Reduce confidence proportionally to excess drift
                excess = (std_dev - self._drift_threshold) / self._drift_threshold
                reduction = min(0.5, excess * 0.3)  # cap reduction at 50%
                adjusted_confidence = max(
                    0.05, raw_signal.confidence - reduction
                )
                drift_info = (
                    f" | DRIFT detected: std={std_dev:.4f} > {self._drift_threshold}, "
                    f"confidence reduced by {reduction:.4f}"
                )
            else:
                drift_info = f" | stable: std={std_dev:.4f}, ema={self._ema_score:.4f}"

        return EvalSignal(
            score=raw_signal.score,
            dimension_scores=raw_signal.dimension_scores,
            confidence=max(0.0, min(1.0, adjusted_confidence)),
            explanation=(
                f"ReflectiveEvaluator(history={len(self._history)}): "
                f"{raw_signal.explanation}{drift_info}"
            ),
            metadata={
                **dict(raw_signal.metadata),
                "reflective_ema": self._ema_score,
                "reflective_history_len": len(self._history),
            },
        )


# ===================================================================== #
#  LLM Critic Evaluator                                                  #
# ===================================================================== #


class LLMCriticEvaluator(BaseEvaluator):
    """Evaluation strategy that uses an LLM provider to critique goal alignment.

    The evaluator formats a structured prompt containing the goal, state,
    and optional context, then delegates to an LLM provider for a qualitative
    assessment.  The provider's response is parsed into an ``EvalSignal``.

    Parameters
    ----------
    provider:
        An LLM provider instance implementing a ``generate(prompt: str) -> str``
        method.  The concrete type is imported conditionally to avoid hard
        dependencies on the infrastructure LLM layer.
    system_prompt:
        Optional system-level instructions for the LLM.
    temperature:
        Sampling temperature for the LLM call.
    """

    def __init__(
        self,
        provider: Any,
        system_prompt: str = "",
        temperature: float = 0.3,
    ) -> None:
        self._provider = provider
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._temperature = temperature

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are an evaluation critic for a goal-directed AI agent. "
            "Given a goal description and current state, assess how well "
            "the state aligns with the goal. Respond with a JSON object: "
            '{"score": <float in [-1,1]>, "confidence": <float in [0,1]>, '
            '"explanation": "<brief reasoning>"}. '
            "A score of 1.0 means perfect alignment, -1.0 means maximum "
            "divergence, and 0.0 means neutral."
        )

    def validate(self, goal: Goal, state: StateSnapshot) -> bool:
        """Validate that the provider is available and goal has an objective."""
        return self._provider is not None and goal.objective is not None

    def evaluate(self, goal: Goal, state: StateSnapshot) -> EvalSignal:
        """Format a prompt and call the LLM provider for evaluation.

        The method constructs a structured prompt from the goal and state,
        sends it to the LLM provider, and parses the response into an
        ``EvalSignal``.  If the provider call fails or the response cannot
        be parsed, a neutral signal with low confidence is returned.
        """
        prompt = self._build_prompt(goal, state)

        try:
            response = self._call_provider(prompt)
            return self._parse_response(response)
        except Exception as exc:
            logger.warning("LLMCriticEvaluator: provider call failed: %s", exc)
            return EvalSignal(
                score=0.0,
                confidence=0.1,
                explanation=f"LLMCriticEvaluator: provider error: {exc}",
            )

    def _build_prompt(self, goal: Goal, state: StateSnapshot) -> str:
        """Build the evaluation prompt from goal and state."""
        obj_repr = "None"
        if goal.objective is not None:
            obj_repr = (
                f"values={goal.objective.values}, "
                f"directions=[{', '.join(d.value for d in goal.objective.directions)}]"
            )

        return (
            f"## Goal Evaluation Request\n\n"
            f"**Goal**: {goal.name} (id={goal.goal_id}, v{goal.version})\n"
            f"  Description: {goal.description}\n"
            f"  Objective: {obj_repr}\n\n"
            f"**Current State**:\n"
            f"  Timestamp: {state.timestamp}\n"
            f"  Values: {state.values}\n"
            f"  Source: {state.source.value}\n\n"
            f"Evaluate the alignment between the current state and the goal. "
            f"Respond with a JSON object containing score, confidence, and explanation."
        )

    def _call_provider(self, prompt: str) -> str:
        """Call the LLM provider's generate method.

        Expects the provider to have a ``generate(prompt: str, **kwargs) -> str``
        method.  If the provider uses a different interface, subclass and
        override this method.
        """
        return self._provider.generate(
            prompt,
            system_prompt=self._system_prompt,
            temperature=self._temperature,
        )

    def _parse_response(self, response: str) -> EvalSignal:
        """Parse the LLM response into an EvalSignal.

        Attempts JSON parsing first; falls back to heuristic extraction
        if the response is not valid JSON.
        """
        import json

        # Try to extract JSON from the response
        cleaned = response.strip()

        # Handle markdown code blocks
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
            score = float(data.get("score", 0.0))
            score = max(-1.0, min(1.0, score))
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            explanation = str(data.get("explanation", ""))

            return EvalSignal(
                score=score,
                confidence=confidence,
                explanation=f"LLMCriticEvaluator: {explanation}",
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.debug("LLMCriticEvaluator: JSON parse failed: %s", exc)
            # Heuristic: treat the entire response as explanation with neutral score
            return EvalSignal(
                score=0.0,
                confidence=0.2,
                explanation=f"LLMCriticEvaluator (unparsed): {cleaned[:200]}",
            )
