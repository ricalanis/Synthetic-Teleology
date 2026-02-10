"""Teleological Coherence (TC) metric.

Measures the degree to which an agent's actions are consistently aligned
with its goals over time, using the evaluation signal as a proxy.

Formula
-------
::

    TC = (mean(eval_scores) + 1) / 2

The raw ``eval_score`` lies in [-1, 1]; the normalisation maps it to [0, 1]
so that ``TC = 1`` corresponds to perfect alignment and ``TC = 0`` to
complete misalignment.

Interpretation
~~~~~~~~~~~~~~
High TC indicates that the agent's chosen actions consistently advance the
current goal.  Low TC suggests a mismatch between the planning/action layer
and the evaluation function, or a fundamentally adversarial environment.
"""

from __future__ import annotations

from synthetic_teleology.measurement.collector import AgentLog
from synthetic_teleology.measurement.metrics.base import BaseMetric


class TeleologicalCoherence(BaseMetric):
    """TC = (mean(eval_scores) + 1) / 2, normalised to [0, 1]."""

    @property
    def name(self) -> str:
        return "teleological_coherence"

    def _compute(self, log: AgentLog) -> float:
        scores = log.get_scores()
        if not scores:
            return 0.0
        mean_score = sum(scores) / len(scores)
        # Normalise from [-1, 1] to [0, 1]
        return (mean_score + 1.0) / 2.0

    def describe(self) -> str:
        return (
            "Teleological Coherence (TC): mean of evaluation scores "
            "normalised to [0, 1]. Higher = actions consistently aligned "
            "with goals."
        )
