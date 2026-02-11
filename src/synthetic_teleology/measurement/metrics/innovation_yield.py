"""Innovation Yield (IY) metric.

Measures novelty and quality improvements attributable to goal revisions,
per Haidemariam (2026) Section 5.4.

When revisions are present:
::

    IY = 0.6 * novelty_ratio + 0.4 * quality_improvement
    novelty_ratio = |post_revision_only_actions| / |all_unique_actions|
    quality_improvement = sigmoid(mean_post_score - mean_pre_score)

When no revisions are present, falls back to the legacy formula:
::

    IY = unique_actions / total_actions

Range: [0, 1].
"""

from __future__ import annotations

import math

from synthetic_teleology.measurement.collector import AgentLog
from synthetic_teleology.measurement.metrics.base import BaseMetric


def _sigmoid(x: float) -> float:
    """Standard sigmoid function, clipped to avoid overflow."""
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


class InnovationYield(BaseMetric):
    """IY: novelty/quality improvements attributable to goal revisions.

    Uses attribution formula when revisions are present, falls back to
    unique/total ratio when no revisions occurred.
    """

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

        # When revisions are present, use attribution formula
        if log.revision_count > 0:
            return self._compute_attributed(log, actions)

        # Fallback: unique/total (backward compat)
        return len(set(actions)) / len(actions)

    def _compute_attributed(self, log: AgentLog, actions: list[str]) -> float:
        """Attribution formula: 0.6 * novelty_ratio + 0.4 * quality_improvement."""
        revision_steps = set(log.steps_where("goal_revised"))
        scores = log.get_scores()
        all_action_names = log.get_action_names()

        # Split actions into pre/post revision sets
        first_revision = min(revision_steps) if revision_steps else len(all_action_names)
        pre_actions: set[str] = set()
        post_actions: set[str] = set()

        for i, name in enumerate(all_action_names):
            if not name:
                continue
            entry = log.entries[i] if i < len(log.entries) else None
            step = entry.step if entry else i
            if step < first_revision:
                pre_actions.add(name)
            else:
                post_actions.add(name)

        # Novelty ratio: actions that appeared only after revisions
        all_unique = set(actions)
        post_only = post_actions - pre_actions
        novelty_ratio = len(post_only) / len(all_unique) if all_unique else 0.0

        # Quality improvement: sigmoid of score difference
        pre_scores = [
            scores[i] for i in range(len(scores))
            if i < len(log.entries) and log.entries[i].step < first_revision
        ]
        post_scores = [
            scores[i] for i in range(len(scores))
            if i < len(log.entries) and log.entries[i].step >= first_revision
        ]

        if pre_scores and post_scores:
            mean_pre = sum(pre_scores) / len(pre_scores)
            mean_post = sum(post_scores) / len(post_scores)
            quality_improvement = _sigmoid(mean_post - mean_pre)
        else:
            quality_improvement = 0.5  # neutral when insufficient data

        return 0.6 * novelty_ratio + 0.4 * quality_improvement

    def describe(self) -> str:
        return (
            "Innovation Yield (IY): novelty/quality improvements attributable "
            "to goal revisions. Range [0, 1]. Uses attribution formula when "
            "revisions present, unique/total fallback otherwise."
        )
