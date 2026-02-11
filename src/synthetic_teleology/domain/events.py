"""Domain events for the Synthetic Teleology framework.

Every event is a frozen dataclass inheriting from ``DomainEvent``.  Events are
the primary integration mechanism between bounded contexts: the teleological
loop emits events; listeners (infrastructure, services, presentation) react.

All events carry a ``timestamp`` and a ``source_id`` identifying the
originating entity or component.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .enums import AgentState, NegotiationStrategy, RevisionReason
from .values import ActionSpec, EvalSignal, GoalRevision, ObjectiveVector, PolicySpec

# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainEvent:
    """Base class for all domain events.

    Subclasses should remain frozen (immutable) and should *not* override
    ``__eq__`` or ``__hash__`` (the default dataclass identity semantics
    based on field values are intentional).
    """

    timestamp: float = field(default_factory=time.time)
    source_id: str = ""


# ---------------------------------------------------------------------------
# Goal lifecycle events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GoalCreated(DomainEvent):
    """A new goal was created."""

    goal_id: str = ""
    goal_name: str = ""
    objective: ObjectiveVector | None = None
    parent_id: str | None = None


@dataclass(frozen=True)
class GoalRevised(DomainEvent):
    """An existing goal was revised, producing a new successor goal."""

    revision: GoalRevision | None = None
    reason: RevisionReason | None = None
    previous_objective: ObjectiveVector | None = None
    new_objective: ObjectiveVector | None = None


@dataclass(frozen=True)
class GoalAbandoned(DomainEvent):
    """A goal was abandoned (permanently deactivated)."""

    goal_id: str = ""
    goal_name: str = ""
    reason: str = ""


@dataclass(frozen=True)
class GoalAchieved(DomainEvent):
    """A goal was marked as achieved."""

    goal_id: str = ""
    goal_name: str = ""
    final_eval: EvalSignal | None = None


# ---------------------------------------------------------------------------
# Evaluation events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationCompleted(DomainEvent):
    """The evaluation function Delta(G_t, S_t) finished producing a signal."""

    goal_id: str = ""
    eval_signal: EvalSignal | None = None
    revision_triggered: bool = False


# ---------------------------------------------------------------------------
# Planning / action events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanGenerated(DomainEvent):
    """The planning subsystem produced a new policy pi_t."""

    goal_id: str = ""
    policy: PolicySpec | None = None
    planning_time_ms: float = 0.0


@dataclass(frozen=True)
class ActionExecuted(DomainEvent):
    """An action from the current policy was executed."""

    action: ActionSpec | None = None
    success: bool = True
    result: Mapping[str, Any] = field(default_factory=dict)
    error_message: str = ""


# ---------------------------------------------------------------------------
# Constraint events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstraintViolated(DomainEvent):
    """A constraint in E_t was violated."""

    constraint_id: str = ""
    constraint_name: str = ""
    violation_details: str = ""
    severity: float = 0.0  # 0 = marginal, 1 = catastrophic


@dataclass(frozen=True)
class ConstraintRestored(DomainEvent):
    """A previously violated constraint is now satisfied again."""

    constraint_id: str = ""
    constraint_name: str = ""
    restoration_details: str = ""


# ---------------------------------------------------------------------------
# Reflection events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReflectionTriggered(DomainEvent):
    """The agent entered the reflection phase of its teleological loop."""

    goal_id: str = ""
    trigger_reason: str = ""
    eval_signal: EvalSignal | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loop lifecycle events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoopStepCompleted(DomainEvent):
    """One full iteration of the teleological loop completed."""

    step_number: int = 0
    agent_state: AgentState = AgentState.IDLE
    elapsed_ms: float = 0.0
    goal_id: str = ""


# ---------------------------------------------------------------------------
# Agent events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentRegistered(DomainEvent):
    """A new agent was registered in the system."""

    agent_id: str = ""
    agent_name: str = ""
    initial_goal_id: str = ""


@dataclass(frozen=True)
class StateChanged(DomainEvent):
    """The agent's FSM state transitioned."""

    agent_id: str = ""
    previous_state: AgentState = AgentState.IDLE
    new_state: AgentState = AgentState.IDLE


# ---------------------------------------------------------------------------
# Multi-agent negotiation events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NegotiationStarted(DomainEvent):
    """A multi-agent goal negotiation round began."""

    negotiation_id: str = ""
    participant_ids: tuple[str, ...] = ()
    strategy: NegotiationStrategy = NegotiationStrategy.CONSENSUS
    contested_goal_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConsensusReached(DomainEvent):
    """Agents reached consensus on a shared or revised goal."""

    negotiation_id: str = ""
    agreed_objective: ObjectiveVector | None = None
    participant_ids: tuple[str, ...] = ()
    rounds: int = 0


# ---------------------------------------------------------------------------
# Environment / perturbation events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerturbationInjected(DomainEvent):
    """An external perturbation was injected into the environment.

    Used by benchmarking / experimentation infrastructure to stress-test
    the teleological loop's ability to adapt.
    """

    perturbation_type: str = ""
    magnitude: float = 0.0
    affected_dimensions: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Knowledge events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KnowledgeUpdated(DomainEvent):
    """A knowledge entry was added or updated in the metacognitive commons."""

    key: str = ""
    source: str = ""
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
