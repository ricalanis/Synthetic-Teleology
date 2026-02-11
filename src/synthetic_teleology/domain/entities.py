"""Domain entities for the Synthetic Teleology framework.

Entities have *identity* (a unique id that persists across mutations) and a
mutable lifecycle.  ``Goal`` is versioned and tracks its own status transitions.
``Constraint`` is a runtime wrapper around the immutable ``ConstraintSpec``.
"""

from __future__ import annotations

import dataclasses
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .enums import ConstraintType, GoalStatus
from .values import ConstraintSpec, EvalSignal, GoalProvenance, GoalRevision, ObjectiveVector

# ---------------------------------------------------------------------------
# Goal entity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Goal:
    """An immutable goal entity -- has identity (``goal_id``) and is versioned.

    Goals form a directed acyclic graph via ``parent_id``.  Revisions produce
    a *new* ``Goal`` instance with an incremented ``version``; the original
    goal is left unchanged (immutability per Haidemariam 2026: G_t -> G_{t+1}
    produces a NEW goal entity).

    In LLM mode, the primary goal representation is ``description`` (natural
    language) with optional ``success_criteria``.  The ``objective`` vector is
    optional and used for numeric evaluation when available.

    Note: ``metadata`` is a dict (mutable contents). With ``frozen=True``,
    the dict reference can't be reassigned but its contents can still be
    mutated. This is intentional — same as PyTorch's frozen dataclass patterns.
    """

    goal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    objective: ObjectiveVector | None = None
    success_criteria: list[str] = field(default_factory=list)
    priority: float = 1.0
    status: GoalStatus = GoalStatus.ACTIVE
    parent_id: str | None = None
    version: int = 1
    created_at: float = field(default_factory=time.time)
    provenance: GoalProvenance | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- lifecycle transitions ------------------------------------------------

    def revise(
        self,
        new_objective: ObjectiveVector | None = None,
        reason: str = "",
        eval_signal: EvalSignal | None = None,
        *,
        new_description: str | None = None,
        new_criteria: list[str] | None = None,
    ) -> tuple["Goal", GoalRevision]:
        """Create a revised successor goal and a ``GoalRevision`` record.

        The original goal is left unchanged (immutable). The returned goal
        inherits name, description, parent, and metadata but gets a fresh id,
        incremented version, and the supplied changes.

        Parameters
        ----------
        new_objective:
            Revised objective vector. If ``None``, keeps the current objective.
        reason:
            Human-readable reason for the revision.
        eval_signal:
            The evaluation signal that triggered this revision.
        new_description:
            Revised natural language description. If ``None``, keeps current.
        new_criteria:
            Revised success criteria. If ``None``, keeps current.

        Returns
        -------
        tuple[Goal, GoalRevision]
            ``(new_goal, revision_record)``
        """
        new_goal = Goal(
            name=self.name,
            description=new_description if new_description is not None else self.description,
            objective=new_objective if new_objective is not None else self.objective,
            success_criteria=(
                list(new_criteria) if new_criteria is not None else list(self.success_criteria)
            ),
            priority=self.priority,
            status=GoalStatus.ACTIVE,
            parent_id=self.parent_id,
            version=self.version + 1,
            provenance=self.provenance,
            metadata=dict(self.metadata),
        )
        revision = GoalRevision(
            timestamp=time.time(),
            previous_goal_id=self.goal_id,
            new_goal_id=new_goal.goal_id,
            reason=reason,
            eval_signal=eval_signal,
        )
        return new_goal, revision

    def achieve(self) -> "Goal":
        """Return a new Goal with ACHIEVED status."""
        return dataclasses.replace(self, status=GoalStatus.ACHIEVED)

    def abandon(self) -> "Goal":
        """Return a new Goal with ABANDONED status."""
        return dataclasses.replace(self, status=GoalStatus.ABANDONED)

    def suspend(self) -> "Goal":
        """Return a new Goal with SUSPENDED status."""
        return dataclasses.replace(self, status=GoalStatus.SUSPENDED)

    def reactivate(self) -> "Goal":
        """Return a new Goal reactivated from SUSPENDED status."""
        if self.status != GoalStatus.SUSPENDED:
            raise ValueError(
                f"Only suspended goals can be reactivated, "
                f"current status is {self.status.value}"
            )
        return dataclasses.replace(self, status=GoalStatus.ACTIVE)

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
