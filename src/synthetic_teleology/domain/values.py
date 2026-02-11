"""Value objects for the Synthetic Teleology framework.

All types here are frozen dataclasses — immutable, compared by value.
They represent measurements, specifications, and snapshots that have no
identity beyond their content.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .enums import ConstraintType, Direction, GoalOrigin, StateSource

# ---------------------------------------------------------------------------
# ObjectiveVector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectiveVector:
    """Multi-dimensional objective -- the target state G_t.

    Each dimension has a value, a direction of optimization, and an optional
    weight.  The vector supports distance computation for evaluation signals.
    """

    values: tuple[float, ...]
    directions: tuple[Direction, ...]
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.values) != len(self.directions):
            raise ValueError(
                f"values length ({len(self.values)}) and directions length "
                f"({len(self.directions)}) must match"
            )
        if self.weights is not None and len(self.weights) != len(self.values):
            raise ValueError(
                f"weights length ({len(self.weights)}) must match "
                f"values length ({len(self.values)})"
            )

    @property
    def dimension(self) -> int:
        """Number of objective dimensions."""
        return len(self.values)

    def distance_to(self, other: ObjectiveVector) -> float:
        """Weighted Euclidean distance between two objective vectors.

        Parameters
        ----------
        other:
            Another ``ObjectiveVector`` with the same dimensionality.

        Returns
        -------
        float
            The scalar distance value (always >= 0).
        """
        if self.dimension != other.dimension:
            raise ValueError(
                f"Dimension mismatch: {self.dimension} vs {other.dimension}"
            )
        w = self.weights or tuple(1.0 for _ in self.values)
        return float(
            np.sqrt(
                sum(
                    wi * (a - b) ** 2
                    for wi, a, b in zip(w, self.values, other.values)
                )
            )
        )

    def with_values(self, new_values: tuple[float, ...]) -> ObjectiveVector:
        """Return a copy with new numeric values, keeping directions and weights."""
        return ObjectiveVector(
            values=new_values,
            directions=self.directions,
            weights=self.weights,
        )


# ---------------------------------------------------------------------------
# EvalSignal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalSignal:
    """Result of evaluating the current state against a goal -- Delta(G_t, S_t).

    ``score`` is a scalar in [-1, 1] where positive values indicate progress
    toward the goal and negative values indicate regression.
    """

    score: float  # overall evaluation [-1, 1]
    dimension_scores: tuple[float, ...] = ()  # per-dimension scores
    confidence: float = 1.0  # evaluator confidence [0, 1]
    explanation: str = ""
    reasoning: str = ""  # LLM chain-of-thought reasoning
    criteria_scores: Mapping[str, float] = field(default_factory=dict)  # per-criterion
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [-1, 1], got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def is_satisfactory(self) -> bool:
        """True when the evaluation is non-negative (goal is being met)."""
        return self.score >= 0.0

    @property
    def magnitude(self) -> float:
        """Absolute magnitude of the evaluation signal."""
        return abs(self.score)


# ---------------------------------------------------------------------------
# ConstraintSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstraintSpec:
    """Specification of a single constraint in E_t.

    This is a *value description* of what the constraint requires.
    The actual enforcement logic lives in service or infrastructure layers.
    """

    name: str
    constraint_type: ConstraintType
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # relevance weight for soft constraints


# ---------------------------------------------------------------------------
# ActionSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionSpec:
    """A concrete action the agent can take.

    Immutable specification; execution is handled by the services layer.
    In LLM mode, ``tool_name`` maps to a LangChain tool for execution.
    """

    action_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    tool_name: str | None = None
    reasoning: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    effect: tuple[float, ...] | None = None
    preconditions: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PolicySpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicySpec:
    """A policy pi_t -- an ordered sequence of actions or action distribution.

    If ``probabilities`` is provided the policy is stochastic; otherwise
    the actions list is treated as a deterministic sequence.
    """

    actions: tuple[ActionSpec, ...] = ()
    probabilities: tuple[float, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.probabilities is not None:
            if len(self.probabilities) != len(self.actions):
                raise ValueError(
                    f"probabilities length ({len(self.probabilities)}) must "
                    f"match actions length ({len(self.actions)})"
                )
            total = sum(self.probabilities)
            if not np.isclose(total, 1.0):
                raise ValueError(
                    f"probabilities must sum to 1.0, got {total}"
                )

    @property
    def is_stochastic(self) -> bool:
        """True when the policy carries a probability distribution."""
        return self.probabilities is not None

    @property
    def size(self) -> int:
        """Number of actions in the policy."""
        return len(self.actions)


# ---------------------------------------------------------------------------
# StateSnapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateSnapshot:
    """Immutable snapshot of the agent's perceived state S_t.

    In LLM mode, ``observation`` carries a natural language description of the
    current state, and ``context`` provides structured context from tools/env.
    """

    timestamp: float
    values: tuple[float, ...] = ()
    observation: str = ""
    context: Mapping[str, Any] = field(default_factory=dict)
    source: StateSource = StateSource.ENVIRONMENT
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        """Number of state dimensions."""
        return len(self.values)

    def as_array(self) -> NDArray[np.float64]:
        """Convert values to a numpy array."""
        return np.array(self.values, dtype=np.float64)


# ---------------------------------------------------------------------------
# GoalRevision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GoalRevision:
    """Immutable record of a goal-revision event.

    Links the previous goal to its successor and captures the evaluation
    signal that triggered the revision.
    """

    revision_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = 0.0
    previous_goal_id: str = ""
    new_goal_id: str = ""
    reason: str = ""
    eval_signal: EvalSignal | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Hypothesis:
    """A candidate plan with confidence and reasoning.

    Generated by the LLMPlanner during multi-hypothesis planning.
    """

    actions: tuple[ActionSpec, ...] = ()
    confidence: float = 0.5
    reasoning: str = ""
    expected_outcome: str = ""
    risk_assessment: str = ""


# ---------------------------------------------------------------------------
# GoalProvenance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GoalProvenance:
    """Immutable provenance record for a goal.

    Tracks who/what created or revised a goal, enabling full lineage tracing
    per Haidemariam (2026) Section on intentional grounding.
    """

    origin: GoalOrigin = GoalOrigin.DESIGN
    source_agent_id: str = ""
    source_description: str = ""
    timestamp: float = 0.0
    parent_provenance_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GoalAuditEntry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GoalAuditEntry:
    """Immutable entry in the goal audit trail.

    Captures a single revision event with full context for replay/analysis.
    """

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = 0.0
    goal_id: str = ""
    previous_goal_id: str = ""
    revision_reason: str = ""
    eval_score: float = 0.0
    eval_confidence: float = 1.0
    provenance: GoalProvenance | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# KnowledgeEntry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KnowledgeEntry:
    """Immutable entry in the metacognitive knowledge store.

    Represents a piece of learned knowledge with typed metadata, tags,
    and source tracking.
    """

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    key: str = ""
    value: Any = None
    source: str = ""
    tags: tuple[str, ...] = ()
    timestamp: float = 0.0
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ConstraintResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstraintResult:
    """Detailed result of a single constraint check.

    Provides richer information than the simple ``(bool, str)`` tuple
    returned by ``BaseConstraintChecker.check()``.
    """

    passed: bool
    message: str = ""
    severity: float = 0.0  # 0=trivial, 1=critical
    checker_name: str = ""
    suggested_mitigation: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
