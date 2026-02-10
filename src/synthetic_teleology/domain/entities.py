"""Domain entities for the Synthetic Teleology framework.

Entities have *identity* (a unique id that persists across mutations) and a
mutable lifecycle.  ``Goal`` is versioned and tracks its own status transitions.
``Constraint`` is a runtime wrapper around the immutable ``ConstraintSpec``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .enums import ConstraintType, GoalStatus
from .values import ConstraintSpec, EvalSignal, GoalRevision, ObjectiveVector


# ---------------------------------------------------------------------------
# Goal entity
# ---------------------------------------------------------------------------

@dataclass
class Goal:
    """A goal entity -- has identity (``goal_id``), is versioned, and carries
    mutable status.

    Goals form a directed acyclic graph via ``parent_id``.  Revisions produce
    a *new* ``Goal`` instance with an incremented ``version`` while the
    original is marked ``GoalStatus.REVISED``.
    """

    goal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    objective: ObjectiveVector | None = None
    status: GoalStatus = GoalStatus.ACTIVE
    parent_id: str | None = None
    version: int = 1
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- lifecycle transitions ------------------------------------------------

    def revise(
        self,
        new_objective: ObjectiveVector,
        reason: str = "",
        eval_signal: EvalSignal | None = None,
    ) -> tuple[Goal, GoalRevision]:
        """Create a revised successor goal and a ``GoalRevision`` record.

        The *current* goal is marked ``GoalStatus.REVISED``; the returned goal
        inherits name, description, parent, and metadata but gets a fresh id,
        incremented version, and the supplied ``new_objective``.

        Returns
        -------
        tuple[Goal, GoalRevision]
            ``(new_goal, revision_record)``
        """
        new_goal = Goal(
            name=self.name,
            description=self.description,
            objective=new_objective,
            status=GoalStatus.ACTIVE,
            parent_id=self.parent_id,
            version=self.version + 1,
            metadata=dict(self.metadata),
        )
        revision = GoalRevision(
            timestamp=time.time(),
            previous_goal_id=self.goal_id,
            new_goal_id=new_goal.goal_id,
            reason=reason,
            eval_signal=eval_signal,
        )
        self.status = GoalStatus.REVISED
        return new_goal, revision

    def achieve(self) -> None:
        """Mark goal as achieved."""
        self.status = GoalStatus.ACHIEVED

    def abandon(self) -> None:
        """Mark goal as abandoned."""
        self.status = GoalStatus.ABANDONED

    def suspend(self) -> None:
        """Mark goal as suspended (can be reactivated later)."""
        self.status = GoalStatus.SUSPENDED

    def reactivate(self) -> None:
        """Reactivate a suspended goal."""
        if self.status != GoalStatus.SUSPENDED:
            raise ValueError(
                f"Only suspended goals can be reactivated, "
                f"current status is {self.status.value}"
            )
        self.status = GoalStatus.ACTIVE

    # -- queries --------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True when the goal is still being pursued."""
        return self.status == GoalStatus.ACTIVE

    @property
    def is_terminal(self) -> bool:
        """True when the goal has reached a terminal status."""
        return self.status in (
            GoalStatus.ACHIEVED,
            GoalStatus.ABANDONED,
            GoalStatus.REVISED,
        )


# ---------------------------------------------------------------------------
# Constraint entity
# ---------------------------------------------------------------------------

class Constraint:
    """Runtime constraint entity wrapping an immutable ``ConstraintSpec``.

    Provides identity (``constraint_id``), activation toggling, and priority
    weighting.
    """

    __slots__ = (
        "constraint_id",
        "name",
        "constraint_type",
        "spec",
        "weight",
        "is_active",
    )

    def __init__(
        self,
        name: str,
        constraint_type: ConstraintType,
        spec: ConstraintSpec,
        weight: float = 1.0,
    ) -> None:
        self.constraint_id: str = str(uuid.uuid4())[:8]
        self.name = name
        self.constraint_type = constraint_type
        self.spec = spec
        self.weight = weight
        self.is_active: bool = True

    def deactivate(self) -> None:
        """Temporarily deactivate this constraint."""
        self.is_active = False

    def activate(self) -> None:
        """Re-activate a deactivated constraint."""
        self.is_active = True

    def __repr__(self) -> str:
        return (
            f"Constraint(id={self.constraint_id!r}, name={self.name!r}, "
            f"type={self.constraint_type.value}, weight={self.weight}, "
            f"active={self.is_active})"
        )
