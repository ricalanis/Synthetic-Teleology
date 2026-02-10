"""Domain exceptions for the Synthetic Teleology framework.

All domain-specific exceptions inherit from ``SyntheticTeleologyError`` so
callers can catch the full family with a single ``except`` clause when needed.
"""

from __future__ import annotations

from typing import Any


class SyntheticTeleologyError(Exception):
    """Base exception for all Synthetic Teleology domain errors."""

    def __init__(self, message: str = "", details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details: dict[str, Any] = details or {}


class GoalCoherenceError(SyntheticTeleologyError):
    """Raised when goal-tree coherence is violated.

    Examples: active child under abandoned parent, circular parent references,
    or dimension mismatches that make evaluation impossible.
    """

    def __init__(
        self,
        message: str = "Goal coherence violation",
        goal_id: str = "",
        issues: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.goal_id = goal_id
        self.issues: list[str] = issues or []


class ConstraintViolationError(SyntheticTeleologyError):
    """Raised when a HARD constraint is violated and cannot be recovered.

    Soft-constraint violations are handled via penalty signals; this exception
    is reserved for situations where the system must halt or abort an action.
    """

    def __init__(
        self,
        message: str = "Hard constraint violated",
        constraint_id: str = "",
        constraint_name: str = "",
        violation_details: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.constraint_id = constraint_id
        self.constraint_name = constraint_name
        self.violation_details = violation_details


class EvaluationError(SyntheticTeleologyError):
    """Raised when the evaluation function Delta(G_t, S_t) fails.

    This may occur when the LLM evaluator is unreachable, returns malformed
    output, or when the state dimensions are incompatible with the goal.
    """

    def __init__(
        self,
        message: str = "Evaluation failed",
        goal_id: str = "",
        evaluator: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.goal_id = goal_id
        self.evaluator = evaluator


class PlanningError(SyntheticTeleologyError):
    """Raised when policy generation (pi_t computation) fails.

    Common causes: infeasible constraints, LLM planner timeout, or an empty
    action space.
    """

    def __init__(
        self,
        message: str = "Planning failed",
        goal_id: str = "",
        planner: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.goal_id = goal_id
        self.planner = planner


class NegotiationDeadlock(SyntheticTeleologyError):
    """Raised when multi-agent negotiation cannot reach resolution.

    The negotiation protocol ran for the maximum number of rounds without
    achieving the required agreement threshold.
    """

    def __init__(
        self,
        message: str = "Negotiation deadlock",
        negotiation_id: str = "",
        participant_ids: tuple[str, ...] = (),
        rounds_completed: int = 0,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.negotiation_id = negotiation_id
        self.participant_ids = participant_ids
        self.rounds_completed = rounds_completed
