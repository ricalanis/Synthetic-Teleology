"""Domain enumerations for the Synthetic Teleology framework.

These enums capture the fixed vocabularies used across the domain layer:
objective directions, constraint types, agent lifecycle states, goal statuses,
revision reasons, negotiation strategies, and error-handling actions.
"""

from enum import Enum


class Direction(Enum):
    """Direction of optimization for an objective dimension."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    MAINTAIN = "maintain"  # maintain within range
    APPROACH = "approach"  # approach target value


class ConstraintType(Enum):
    """Classification of constraint severity."""

    HARD = "hard"  # must never be violated
    SOFT = "soft"  # penalty-based
    BOUNDARY = "boundary"  # upper/lower bounds


class StateSource(Enum):
    """Origin of a state observation."""

    ENVIRONMENT = "environment"
    INTERNAL = "internal"
    COMMUNICATION = "communication"


class AgentState(Enum):
    """Finite-state-machine states for the teleological loop."""

    IDLE = "idle"
    PERCEIVING = "perceiving"
    EVALUATING = "evaluating"
    REVISING = "revising"
    PLANNING = "planning"
    ACTING = "acting"
    REFLECTING = "reflecting"


class GoalStatus(Enum):
    """Lifecycle status of a Goal entity."""

    ACTIVE = "active"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"
    SUSPENDED = "suspended"
    REVISED = "revised"  # was revised, now superseded


class RevisionReason(Enum):
    """Canonical reasons that trigger a goal revision."""

    THRESHOLD_EXCEEDED = "threshold_exceeded"
    GRADIENT_DESCENT = "gradient_descent"
    CONSTRAINT_VIOLATION = "constraint_violation"
    HIERARCHICAL_PROPAGATION = "hierarchical_propagation"
    UNCERTAINTY_REDUCTION = "uncertainty_reduction"
    LLM_CRITIQUE = "llm_critique"
    EVALUATION_FEEDBACK = "evaluation_feedback"
    EXTERNAL_DIRECTIVE = "external_directive"


class NegotiationStrategy(Enum):
    """Multi-agent goal-negotiation strategies."""

    CONSENSUS = "consensus"
    VOTING = "voting"
    AUCTION = "auction"


class ErrorAction(Enum):
    """Actions to take when an error occurs in the teleological loop."""

    RETRY = "retry"
    SKIP = "skip"
    ABORT = "abort"
