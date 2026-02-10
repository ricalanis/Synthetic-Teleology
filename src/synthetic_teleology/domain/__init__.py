"""Domain layer for the Synthetic Teleology framework.

Re-exports all public domain types so that consumers can write::

    from synthetic_teleology.domain import Goal, ObjectiveVector, Direction
"""

# -- Enumerations -------------------------------------------------------------
from .enums import (
    AgentState,
    ConstraintType,
    Direction,
    ErrorAction,
    GoalStatus,
    NegotiationStrategy,
    RevisionReason,
    StateSource,
)

# -- Value Objects ------------------------------------------------------------
from .values import (
    ActionSpec,
    ConstraintSpec,
    EvalSignal,
    GoalRevision,
    ObjectiveVector,
    PolicySpec,
    StateSnapshot,
)

# -- Entities -----------------------------------------------------------------
from .entities import Constraint, Goal

# -- Aggregates ---------------------------------------------------------------
from .aggregates import AgentIdentity, ConstraintSet, GoalTree

# -- Domain Events ------------------------------------------------------------
from .events import (
    ActionExecuted,
    AgentRegistered,
    ConsensusReached,
    ConstraintRestored,
    ConstraintViolated,
    DomainEvent,
    EvaluationCompleted,
    GoalAbandoned,
    GoalAchieved,
    GoalCreated,
    GoalRevised,
    LoopStepCompleted,
    NegotiationStarted,
    PerturbationInjected,
    PlanGenerated,
    ReflectionTriggered,
    StateChanged,
)

# -- Exceptions ---------------------------------------------------------------
from .exceptions import (
    ConstraintViolationError,
    EvaluationError,
    GoalCoherenceError,
    NegotiationDeadlock,
    PlanningError,
    SyntheticTeleologyError,
)

__all__ = [
    # enums
    "AgentState",
    "ConstraintType",
    "Direction",
    "ErrorAction",
    "GoalStatus",
    "NegotiationStrategy",
    "RevisionReason",
    "StateSource",
    # values
    "ActionSpec",
    "ConstraintSpec",
    "EvalSignal",
    "GoalRevision",
    "ObjectiveVector",
    "PolicySpec",
    "StateSnapshot",
    # entities
    "Constraint",
    "Goal",
    # aggregates
    "AgentIdentity",
    "ConstraintSet",
    "GoalTree",
    # events
    "ActionExecuted",
    "AgentRegistered",
    "ConsensusReached",
    "ConstraintRestored",
    "ConstraintViolated",
    "DomainEvent",
    "EvaluationCompleted",
    "GoalAbandoned",
    "GoalAchieved",
    "GoalCreated",
    "GoalRevised",
    "LoopStepCompleted",
    "NegotiationStarted",
    "PerturbationInjected",
    "PlanGenerated",
    "ReflectionTriggered",
    "StateChanged",
    # exceptions
    "ConstraintViolationError",
    "EvaluationError",
    "GoalCoherenceError",
    "NegotiationDeadlock",
    "PlanningError",
    "SyntheticTeleologyError",
]
