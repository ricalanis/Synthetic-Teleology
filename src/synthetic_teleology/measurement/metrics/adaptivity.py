"""Adaptivity (AD) metric.

Measures how quickly the agent recovers to its pre-revision performance
level after a goal revision.  Fast recovery indicates strong adaptive
capacity; slow recovery suggests rigidity or poor replanning.

Formula
-------
For each goal revision event, identify the step where it occurred and
record the evaluation score at that step (the "baseline").  Then count
how many subsequent steps it takes to reach or exceed that baseline.

::

    recovery_steps_i = steps from revision_i until score >= baseline_i
    mean_recovery = mean(recovery_steps_i)  for all revisions
    AD = 1 / (1 + mean_recovery)

Range: (0, 1].  ``AD = 1`` means instant recovery (0 steps).
Approaches 0 as recovery takes longer.

When no revisions occurred, returns 1.0 (the agent never needed to adapt
and thus is trivially maximally adaptive).
"""

from __future__ import annotations

from synthetic_teleology.measurement.collector import AgentLog
from synthetic_teleology.measurement.metrics.base import BaseMetric


class Adaptivity(BaseMetric):
    """AD = 1 / (1 + mean_recovery_steps) after goal revisions."""

    @property
    def name(self) -> str:
        return "adaptivity"

    def validate(self, log: AgentLog) -> bool:
        return log.num_steps >= 2

    def _compute(self, log: AgentLog) -> float:
        revision_steps = log.steps_where("goal_revised")

        if not revision_steps:
            # No revisions -- trivially adaptive
            return 1.0

        scores = log.get_scores()
        entries = log.entries
        n = len(entries)

        recovery_counts: list[int] = []

        for step_num in revision_steps:
            # Find the entry index for this revision step
            entry_idx = _find_entry_index(entries, step_num)
            if entry_idx is None or entry_idx >= n - 1:
                continue

            baseline = scores[entry_idx]

            # Walk forward from the next step to find recovery
            recovery_steps_count = 0
            recovered = False
            for future_idx in range(entry_idx + 1, n):
                recovery_steps_count += 1
                if scores[future_idx] >= baseline:
                    recovered = True
                    break

            if recovered:
                recovery_counts.append(recovery_steps_count)
            else:
                # Never recovered within the log -- penalise with remaining
                # distance (steps from revision to end of log)
                recovery_counts.append(n - entry_idx - 1)

        if not recovery_counts:
            return 1.0

        mean_recovery = sum(recovery_counts) / len(recovery_counts)
        return 1.0 / (1.0 + mean_recovery)

    def describe(self) -> str:
        return (
            "Adaptivity (AD): speed of recovery after goal revisions. "
            "AD = 1 / (1 + mean_recovery_steps). Range (0, 1]."
        )


def _find_entry_index(
    entries: list,
    step_number: int,
) -> int | None:
    """Return the list index of the entry with the given step number."""
    for idx, entry in enumerate(entries):
        if entry.step == step_number:
            return idx
    return None
