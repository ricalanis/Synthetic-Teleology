"""Reflective Efficiency (RE) metric.

Measures whether the agent's reflection phases actually lead to
improvement in subsequent evaluation scores.

Formula
-------
For each step where ``reflection_triggered == True``, compare the mean
evaluation score in the *window* before the reflection to the mean score
in the window after.  A reflection is counted as "effective" if the
post-reflection mean exceeds the pre-reflection mean.

::

    RE = effective_reflections / total_reflections

Range: [0, 1].  ``RE = 1`` means every reflection led to improvement.
``RE = 0`` means no reflection improved scores.

When no reflections occurred, the metric returns 1.0 (vacuously true --
never reflected, never failed) to avoid penalising agents that do not
use the reflection mechanism.

Parameters
~~~~~~~~~~
``window_size`` (default 3): number of steps before/after the reflection
event to average.
"""

from __future__ import annotations

from synthetic_teleology.measurement.collector import AgentLog
from synthetic_teleology.measurement.metrics.base import BaseMetric


class ReflectiveEfficiency(BaseMetric):
    """RE = fraction of reflections that improved eval scores."""

    def __init__(self, window_size: int = 3) -> None:
        self._window_size = max(1, window_size)

    @property
    def name(self) -> str:
        return "reflective_efficiency"

    def validate(self, log: AgentLog) -> bool:
        """Need at least 2 entries; reflections will be checked inside
        ``_compute`` with a graceful fallback."""
        return log.num_steps >= 2

    def _compute(self, log: AgentLog) -> float:
        reflection_steps = log.steps_where("reflection_triggered")

        if not reflection_steps:
            # No reflections occurred -- vacuously efficient
            return 1.0

        scores = log.get_scores()
        n = len(scores)
        w = self._window_size
        effective = 0

        for step_idx in reflection_steps:
            # Find the position of this step in the entries list.
            # step_idx is the step *number* from the entry; we need the
            # list index.
            entry_index = _find_entry_index(log, step_idx)
            if entry_index is None:
                continue

            # Compute pre-window mean
            pre_start = max(0, entry_index - w)
            pre_scores = scores[pre_start:entry_index]

            # Compute post-window mean (exclusive of the reflection step itself)
            post_end = min(n, entry_index + 1 + w)
            post_scores = scores[entry_index + 1 : post_end]

            if not pre_scores or not post_scores:
                continue

            pre_mean = sum(pre_scores) / len(pre_scores)
            post_mean = sum(post_scores) / len(post_scores)

            if post_mean > pre_mean:
                effective += 1

        evaluated = len(reflection_steps)
        if evaluated == 0:
            return 1.0

        return effective / evaluated

    def describe(self) -> str:
        return (
            f"Reflective Efficiency (RE): fraction of reflections that "
            f"improved eval scores (window={self._window_size}). Range [0, 1]."
        )


def _find_entry_index(log: AgentLog, step_number: int) -> int | None:
    """Return the list index of the entry with the given step number,
    or ``None`` if not found."""
    for idx, entry in enumerate(log.entries):
        if entry.step == step_number:
            return idx
    return None
