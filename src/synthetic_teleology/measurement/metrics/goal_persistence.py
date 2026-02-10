"""Goal Persistence (GP) metric.

Measures how long goals persist before being revised.  An agent that
constantly revises its goals has low persistence; one that commits to
goals has high persistence.

Formula
-------
::

    GP = 1 - (num_revisions / num_steps)

Range: [0, 1].  ``GP = 1`` means the agent never revised its goal.
``GP = 0`` means a revision occurred on every single step.

Interpretation
~~~~~~~~~~~~~~
High GP indicates strong goal commitment and purposeful action.  Low GP
may indicate either a volatile environment or an over-reactive revision
policy.
"""

from __future__ import annotations

from synthetic_teleology.measurement.collector import AgentLog
from synthetic_teleology.measurement.metrics.base import BaseMetric


class GoalPersistence(BaseMetric):
    """GP = 1 - (num_revisions / num_steps)."""

    @property
    def name(self) -> str:
        return "goal_persistence"

    def _compute(self, log: AgentLog) -> float:
        num_revisions = log.revision_count
        num_steps = log.num_steps

        # Clamp to [0, 1] -- more revisions than steps is theoretically
        # impossible but we guard defensively.
        ratio = min(num_revisions / num_steps, 1.0)
        return 1.0 - ratio

    def describe(self) -> str:
        return (
            "Goal Persistence (GP): measures goal stability as "
            "1 - (num_revisions / num_steps). Range [0, 1]."
        )
