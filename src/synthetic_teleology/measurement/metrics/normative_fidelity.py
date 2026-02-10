"""Normative Fidelity (NF) metric.

Measures the agent's adherence to the ethical and operational constraints
defined in its environment envelope E_t.

Formula
-------
::

    NF = 1 - (num_violation_steps / num_steps)

Range: [0, 1].  ``NF = 1`` means no constraint was ever violated.
``NF = 0`` means a violation occurred on every step.

Note: ``num_violation_steps`` counts the number of *steps* on which at
least one constraint was violated, not the total number of violations
(an agent may violate multiple constraints simultaneously on a single
step).
"""

from __future__ import annotations

from synthetic_teleology.measurement.collector import AgentLog
from synthetic_teleology.measurement.metrics.base import BaseMetric


class NormativeFidelity(BaseMetric):
    """NF = 1 - (num_violation_steps / num_steps)."""

    @property
    def name(self) -> str:
        return "normative_fidelity"

    def _compute(self, log: AgentLog) -> float:
        num_steps = log.num_steps
        violation_steps = log.steps_where("constraint_violated")
        num_violation_steps = len(violation_steps)

        ratio = min(num_violation_steps / num_steps, 1.0)
        return 1.0 - ratio

    def describe(self) -> str:
        return (
            "Normative Fidelity (NF): constraint adherence measured as "
            "1 - (violation_steps / total_steps). Range [0, 1]."
        )
