"""Teleological Coherence (TC) metric.

Measures the degree to which an agent's goal revisions are responsive to
evaluation signals, per Haidemariam (2026) Section 5.4.

Paper formula
-------------
::

    TC = corr(||G_{t+1} - G_t||, Delta_t)

A negative correlation means revisions respond to poor evaluations — i.e.,
the agent is coherent.  We map r to TC via ``TC = clip((1 - r) / 2, 0, 1)``
so that r = -1 → TC = 1 (perfect coherence), r = 0 → TC = 0.5, r = 1 → TC = 0.

Three computation tiers:
1. **Primary** (goal_values available): Pearson correlation between
   goal-change magnitude and eval_score at each step.
2. **Proxy** (revisions present, no goal_values): fraction of revisions
   that are "responsive" (happened during poor eval AND followed by
   improvement).
3. **Legacy** (no revisions): ``(mean(scores) + 1) / 2`` — backward compat.
"""

from __future__ import annotations

import math

from synthetic_teleology.measurement.collector import AgentLog
from synthetic_teleology.measurement.metrics.base import BaseMetric


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient between two equal-length lists."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return cov / (sx * sy)


class TeleologicalCoherence(BaseMetric):
    """TC: coherence of goal revisions with evaluation signals.

    Uses correlation when goal_values are available, proxy when only
    revisions are present, and legacy (mean-score) as final fallback.
    """

    @property
    def name(self) -> str:
        return "teleological_coherence"

    def _compute(self, log: AgentLog) -> float:
        scores = log.get_scores()
        if not scores:
            return 0.0

        # Tier 1: correlation when goal_values series is available
        goal_series = log.get_goal_value_series()
        has_goal_values = any(gv is not None for gv in goal_series)

        if has_goal_values:
            return self._compute_correlation(goal_series, scores)

        # Tier 2: proxy when revisions are present
        if log.revision_count > 0:
            return self._compute_proxy(log, scores)

        # Tier 3: legacy fallback
        return self._compute_legacy(scores)

    def _compute_correlation(
        self,
        goal_series: list[tuple[float, ...] | None],
        scores: list[float],
    ) -> float:
        """Primary: Pearson correlation between goal-change magnitude and eval."""
        magnitudes: list[float] = []
        eval_at_change: list[float] = []

        prev_gv: tuple[float, ...] | None = None
        for i, gv in enumerate(goal_series):
            if gv is not None:
                if prev_gv is not None and len(gv) == len(prev_gv):
                    mag = math.sqrt(sum((a - b) ** 2 for a, b in zip(gv, prev_gv)))
                    magnitudes.append(mag)
                    eval_at_change.append(scores[i] if i < len(scores) else 0.0)
                prev_gv = gv

        if len(magnitudes) < 2:
            # Not enough data points for correlation, fall back to proxy
            return self._compute_legacy(scores)

        r = _pearson(magnitudes, eval_at_change)
        # Map: r=-1 -> TC=1, r=0 -> TC=0.5, r=1 -> TC=0
        tc = max(0.0, min(1.0, (1.0 - r) / 2.0))
        return tc

    def _compute_proxy(self, log: AgentLog, scores: list[float]) -> float:
        """Proxy: fraction of revisions that are responsive."""
        revision_steps = log.steps_where("goal_revised")
        if not revision_steps:
            return self._compute_legacy(scores)

        responsive = 0
        for step in revision_steps:
            if step >= len(scores):
                continue
            # Revision during poor eval?
            poor_eval = scores[step] < 0.0
            # Followed by improvement?
            if step + 1 < len(scores):
                improved = scores[step + 1] > scores[step]
            else:
                improved = False
            if poor_eval and improved:
                responsive += 1

        return responsive / len(revision_steps) if revision_steps else 0.5

    @staticmethod
    def _compute_legacy(scores: list[float]) -> float:
        """Legacy: (mean(scores) + 1) / 2."""
        mean_score = sum(scores) / len(scores)
        return (mean_score + 1.0) / 2.0

    def describe(self) -> str:
        return (
            "Teleological Coherence (TC): correlation between goal-change "
            "magnitude and evaluation signals, normalised to [0, 1]. "
            "Higher = revisions more responsive to poor evaluations."
        )
