"""Aggregate roots for the Synthetic Teleology framework.

Aggregates enforce consistency boundaries.  External code should only mutate
domain state through aggregate methods, never by reaching into child entities
directly.

* ``GoalTree`` -- Composite pattern for hierarchical goals.
* ``ConstraintSet`` -- Ordered, prioritised set of constraints (E_t).
* ``AgentIdentity`` -- Owns goal history and revision log for a single agent.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Iterator, Sequence

from .entities import Constraint, Goal
from .enums import ConstraintType
from .values import GoalRevision

# ---------------------------------------------------------------------------
# GoalTree
# ---------------------------------------------------------------------------

class GoalTree:
    """Composite pattern: hierarchical goal structure.

    The tree is rooted at a single ``Goal``.  Sub-goals can be added to any
    existing node.  The aggregate exposes traversal helpers (lineage, DFS
    flatten), coherence validation, and top-down revision propagation.
    """

    def __init__(self, root: Goal) -> None:
        self._root = root
        self._children: dict[str, list[Goal]] = {root.goal_id: []}
        self._all_goals: dict[str, Goal] = {root.goal_id: root}
        self._lock = threading.Lock()

    # -- properties -----------------------------------------------------------

    @property
    def root(self) -> Goal:
        """The root goal of the hierarchy."""
        return self._root

    @property
    def size(self) -> int:
        """Total number of goals in the tree."""
        return len(self._all_goals)

    # -- mutations ------------------------------------------------------------

    def add_subgoal(self, parent_id: str, child: Goal) -> None:
        """Attach *child* as a sub-goal of *parent_id*.

        Raises ``KeyError`` if the parent does not exist in the tree.
        """
        with self._lock:
            if parent_id not in self._all_goals:
                raise KeyError(f"Parent goal {parent_id!r} not found in tree")
            child = dataclasses.replace(child, parent_id=parent_id)
            self._children.setdefault(parent_id, []).append(child)
            self._children.setdefault(child.goal_id, [])
            self._all_goals[child.goal_id] = child

    def remove_subgoal(self, goal_id: str) -> None:
        """Remove *goal_id* and its entire subtree from the tree.

        Raises ``ValueError`` if attempting to remove the root, or
        ``KeyError`` if the goal is not found.
        """
        with self._lock:
            if goal_id == self._root.goal_id:
                raise ValueError("Cannot remove root goal")
            goal = self._all_goals.get(goal_id)
            if goal is None:
                raise KeyError(f"Goal {goal_id!r} not found in tree")
            # Detach from parent's children list
            if goal.parent_id and goal.parent_id in self._children:
                self._children[goal.parent_id] = [
                    g for g in self._children[goal.parent_id]
                    if g.goal_id != goal_id
                ]
            # Remove the whole subtree
            for child_id in self._get_subtree_ids(goal_id):
                self._all_goals.pop(child_id, None)
                self._children.pop(child_id, None)

    # -- queries --------------------------------------------------------------

    def get_goal(self, goal_id: str) -> Goal:
        """Retrieve a goal by id.  Raises ``KeyError`` if absent."""
        if goal_id not in self._all_goals:
            raise KeyError(f"Goal {goal_id!r} not found in tree")
        return self._all_goals[goal_id]

    def get_children(self, goal_id: str) -> list[Goal]:
        """Return direct children of *goal_id*."""
        return list(self._children.get(goal_id, []))

    def get_lineage(self, goal_id: str) -> list[Goal]:
        """Return the path from root to *goal_id* (inclusive)."""
        if goal_id not in self._all_goals:
            raise KeyError(f"Goal {goal_id!r} not found in tree")
        lineage: list[Goal] = []
        current_id: str | None = goal_id
        while current_id is not None:
            goal = self._all_goals[current_id]
            lineage.append(goal)
            current_id = goal.parent_id
        lineage.reverse()
        return lineage

    def flatten(self) -> list[Goal]:
        """Depth-first traversal returning every goal in the tree."""
        return [
            self._all_goals[gid]
            for gid in self._get_subtree_ids(self._root.goal_id)
        ]

    def get_active_leaves(self) -> list[Goal]:
        """Return active goals that have no active children."""
        leaves: list[Goal] = []
        for goal_id, children in self._children.items():
            goal = self._all_goals.get(goal_id)
            if goal is None or not goal.is_active:
                continue
            active_children = [c for c in children if c.is_active]
            if not active_children:
                leaves.append(goal)
        return leaves

    # -- validation -----------------------------------------------------------

    def validate_coherence(self) -> list[str]:
        """Check structural consistency of the tree.

        Returns a (possibly empty) list of human-readable issue descriptions.
        """
        issues: list[str] = []
        for goal_id, children in self._children.items():
            parent = self._all_goals.get(goal_id)
            if parent is None:
                issues.append(
                    f"Orphaned children list for missing parent {goal_id!r}"
                )
                continue
            for child in children:
                if child.is_active and not parent.is_active:
                    issues.append(
                        f"Active child {child.goal_id!r} under "
                        f"non-active parent {parent.goal_id!r} "
                        f"(status={parent.status.value})"
                    )
                if child.parent_id != goal_id:
                    issues.append(
                        f"Child {child.goal_id!r} parent_id={child.parent_id!r} "
                        f"does not match tree parent {goal_id!r}"
                    )
        return issues

    # -- revision propagation -------------------------------------------------

    def propagate_revision(self, revised_goal: Goal) -> list[GoalRevision]:
        """Recursively propagate a parent revision to all active children.

        When a parent's objective changes, child goals are revised with a
        note that the change was propagated.  The tree's internal bookkeeping
        is updated accordingly.

        Returns
        -------
        list[GoalRevision]
            All revision records produced by the propagation.
        """
        with self._lock:
            return self._propagate_revision_locked(revised_goal)

    def _propagate_revision_locked(self, revised_goal: Goal) -> list[GoalRevision]:
        """Internal propagation — must be called with self._lock held."""
        revisions: list[GoalRevision] = []
        children = self._children.get(revised_goal.goal_id, [])
        for idx, child in enumerate(list(children)):
            if not child.is_active:
                continue
            if revised_goal.objective is None or child.objective is None:
                continue

            new_child, revision = child.revise(
                child.objective,
                reason=f"Propagated from parent {revised_goal.goal_id}",
            )

            # Update internal maps
            self._all_goals[new_child.goal_id] = new_child
            self._all_goals.pop(child.goal_id, None)
            self._children[revised_goal.goal_id][idx] = new_child
            self._children[new_child.goal_id] = self._children.pop(
                child.goal_id, []
            )
            revisions.append(revision)

            # Recurse into the new child's subtree
            revisions.extend(self._propagate_revision_locked(new_child))

        return revisions

    # -- internal helpers -----------------------------------------------------

    def _get_subtree_ids(self, goal_id: str) -> list[str]:
        """Return *goal_id* and all transitive descendant ids (DFS order)."""
        result = [goal_id]
        for child in self._children.get(goal_id, []):
            result.extend(self._get_subtree_ids(child.goal_id))
        return result


# ---------------------------------------------------------------------------
# ConstraintSet
# ---------------------------------------------------------------------------

class ConstraintSet:
    """Ordered, priority-sorted set of constraints -- E_t.

    Constraints are maintained sorted by descending weight so that the
    highest-priority constraints are checked first.
    """

    def __init__(self) -> None:
        self._constraints: list[Constraint] = []

    def add(self, constraint: Constraint) -> None:
        """Insert a constraint, maintaining priority order."""
        self._constraints.append(constraint)
        self._constraints.sort(key=lambda c: c.weight, reverse=True)

    def remove(self, constraint_id: str) -> None:
        """Remove a constraint by its id (no-op if not found)."""
        self._constraints = [
            c for c in self._constraints if c.constraint_id != constraint_id
        ]

    def get_active(self) -> list[Constraint]:
        """Return only the currently active constraints."""
        return [c for c in self._constraints if c.is_active]

    def filter_by_type(self, constraint_type: ConstraintType) -> list[Constraint]:
        """Return active constraints of the given type."""
        return [
            c
            for c in self._constraints
            if c.constraint_type == constraint_type and c.is_active
        ]

    def get_hard_constraints(self) -> list[Constraint]:
        """Convenience: return all active HARD constraints."""
        return self.filter_by_type(ConstraintType.HARD)

    # -- dunder protocols -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._constraints)

    def __iter__(self) -> Iterator[Constraint]:
        return iter(self._constraints)

    def __bool__(self) -> bool:
        return len(self._constraints) > 0

    def __repr__(self) -> str:
        return f"ConstraintSet(size={len(self._constraints)})"


# ---------------------------------------------------------------------------
# AgentIdentity
# ---------------------------------------------------------------------------

class AgentIdentity:
    """Aggregate root: owns goal history and revision log for one agent.

    ``AgentIdentity`` is the single source of truth for an agent's current
    goal and its complete revision lineage.
    """

    def __init__(self, agent_id: str, initial_goal: Goal) -> None:
        self.agent_id = agent_id
        self._current_goal = initial_goal
        self._goal_history: list[Goal] = [initial_goal]
        self._revision_log: list[GoalRevision] = []

    # -- properties -----------------------------------------------------------

    @property
    def current_goal(self) -> Goal:
        """The agent's currently active goal."""
        return self._current_goal

    @property
    def goal_history(self) -> Sequence[Goal]:
        """Chronologically ordered list of all goals this agent has held."""
        return list(self._goal_history)

    @property
    def revision_log(self) -> Sequence[GoalRevision]:
        """Chronologically ordered list of all revision records."""
        return list(self._revision_log)

    @property
    def revision_count(self) -> int:
        """Number of revisions that have occurred."""
        return len(self._revision_log)

    # -- mutations ------------------------------------------------------------

    def record_revision(self, new_goal: Goal, revision: GoalRevision) -> None:
        """Record a goal revision: update current goal and append to logs."""
        self._current_goal = new_goal
        self._goal_history.append(new_goal)
        self._revision_log.append(revision)

    # -- queries --------------------------------------------------------------

    def get_goal_at_version(self, version: int) -> Goal | None:
        """Retrieve a historical goal by version number, or None."""
        for goal in self._goal_history:
            if goal.version == version:
                return goal
        return None

    def __repr__(self) -> str:
        return (
            f"AgentIdentity(agent_id={self.agent_id!r}, "
            f"current_goal={self._current_goal.goal_id!r}, "
            f"revisions={self.revision_count})"
        )
