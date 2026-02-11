"""Reasoning Quality (RQ) metric.

Measures the coherence and consistency of LLM reasoning traces across steps.
This metric is specific to LLM-mode agents that populate the ``reasoning``
field on :class:`AgentLogEntry`.

Formula
-------
::

    RQ = (presence_ratio + diversity_ratio + avg_length_ratio) / 3

Components
~~~~~~~~~~
- **presence_ratio**: fraction of steps that have non-empty reasoning
- **diversity_ratio**: unique reasoning tokens / total tokens (measures
  whether the agent reasons differently each step vs. repeating itself)
- **avg_length_ratio**: min(avg_length / 200, 1.0) — penalises very short
  or absent reasoning, saturates at ~200 chars

Interpretation
~~~~~~~~~~~~~~
High RQ (close to 1) means the agent produces varied, substantive reasoning
at every step.  Low RQ suggests the agent is either not reasoning (numeric
mode) or producing repetitive/shallow reasoning.
"""

from __future__ import annotations

from synthetic_teleology.measurement.collector import AgentLog
from synthetic_teleology.measurement.metrics.base import BaseMetric


class ReasoningQuality(BaseMetric):
    """RQ = (presence + diversity + length) / 3, normalised to [0, 1]."""

    @property
    def name(self) -> str:
        return "reasoning_quality"

    def validate(self, log: AgentLog) -> bool:
        return log.num_steps >= 1

    def _compute(self, log: AgentLog) -> float:
        entries = log.entries
        if not entries:
            return 0.0

        n = len(entries)
        reasoning_texts = [e.reasoning for e in entries]

        # 1. Presence ratio: fraction of steps with non-empty reasoning
        non_empty = sum(1 for r in reasoning_texts if r.strip())
        presence_ratio = non_empty / n

        # 2. Diversity ratio: unique word set overlap
        all_words: list[str] = []
        for r in reasoning_texts:
            all_words.extend(r.lower().split())

        if all_words:
            unique_words = set(all_words)
            diversity_ratio = len(unique_words) / len(all_words)
        else:
            diversity_ratio = 0.0

        # 3. Length ratio: average reasoning length, saturating at 200 chars
        lengths = [len(r) for r in reasoning_texts]
        avg_length = sum(lengths) / n if n > 0 else 0.0
        length_ratio = min(avg_length / 200.0, 1.0)

        return (presence_ratio + diversity_ratio + length_ratio) / 3.0

    def describe(self) -> str:
        return (
            "Reasoning Quality (RQ): composite of presence, diversity, and "
            "length of LLM reasoning traces. Higher = richer, more varied "
            "reasoning at each step. Range [0, 1]."
        )
