"""Innovation Yield (IY) metric.

Measures the diversity and novelty of the agent's action repertoire.
An agent that always takes the same action has low innovation yield;
one that explores a wide variety of actions has high innovation yield.

Formula
-------
::

    IY = unique_actions / total_actions

Range: [0, 1].  ``IY = 1`` means every action was unique.
``IY`` approaches ``1/N`` when the same action is repeated N times.

Empty action names are excluded from the computation (they indicate
steps where no action was recorded).
"""

from __future__ import annotations

from synthetic_teleology.measurement.collector import AgentLog
from synthetic_teleology.measurement.metrics.base import BaseMetric


class InnovationYield(BaseMetric):
    """IY = unique_actions / total_actions."""

    @property
    def name(self) -> str:
        return "innovation_yield"

    def validate(self, log: AgentLog) -> bool:
        """Need at least 2 entries with non-empty action names."""
        actions = [name for name in log.get_action_names() if name]
        return len(actions) >= 2

    def _compute(self, log: AgentLog) -> float:
        actions = [name for name in log.get_action_names() if name]

        if not actions:
            return 0.0

        unique = len(set(actions))
        total = len(actions)
        return unique / total

    def describe(self) -> str:
        return (
            "Innovation Yield (IY): action diversity measured as "
            "unique_actions / total_actions. Range [0, 1]."
        )
