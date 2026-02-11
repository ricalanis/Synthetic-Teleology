"""Empowerment metric — information-theoretic teleology.

Implements Haidemariam (2026) Section 5.4.5: I(A; S'|S) — mutual information
between actions and state transitions, measuring how much influence the
agent's actions have on the next state.

Formula
-------
::

    E = (H(delta_S) - H(delta_S | A)) / H(delta_S)

where H is Shannon entropy computed over discretised transition magnitudes.

Normalised to [0, 1]:
* ``E = 1`` means actions perfectly determine state transitions.
* ``E = 0`` means actions have no influence on state transitions.
"""

from __future__ import annotations

import math
from collections import Counter

from synthetic_teleology.measurement.collector import AgentLog
from synthetic_teleology.measurement.metrics.base import BaseMetric


def _entropy(counts: Counter) -> float:
    """Shannon entropy (nats) from a Counter of bin labels."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def _discretise(value: float, num_bins: int = 8) -> int:
    """Map a non-negative float to a discrete bin index."""
    # Use log-scale binning for better resolution at small magnitudes
    if value <= 0.0:
        return 0
    log_val = math.log1p(value)
    # Map to bins based on quantile-like spacing
    bin_width = 1.0  # each bin covers ~1 unit of log-space
    return min(int(log_val / bin_width), num_bins - 1)


class Empowerment(BaseMetric):
    """Empowerment: I(A; S'|S) — mutual information between actions and transitions.

    Opt-in metric. Add via ``engine.add_metric(Empowerment())``.

    Parameters
    ----------
    num_bins:
        Number of bins for discretising transition magnitudes. Default 8.
    """

    def __init__(self, num_bins: int = 8) -> None:
        self._num_bins = num_bins

    @property
    def name(self) -> str:
        return "empowerment"

    def validate(self, log: AgentLog) -> bool:
        """Need at least 3 entries with state values and actions."""
        if log.num_steps < 3:
            return False
        has_states = sum(1 for e in log.entries if e.state_values) >= 2
        has_actions = sum(1 for e in log.entries if e.action_name) >= 2
        return has_states and has_actions

    def _compute(self, log: AgentLog) -> float:
        entries = log.entries

        # Compute transition magnitudes
        transitions: list[tuple[str, float]] = []  # (action, delta_mag)
        for i in range(len(entries) - 1):
            curr = entries[i]
            nxt = entries[i + 1]

            action = curr.action_name or "none"

            if curr.state_values and nxt.state_values:
                if len(curr.state_values) == len(nxt.state_values):
                    delta_mag = math.sqrt(
                        sum(
                            (a - b) ** 2
                            for a, b in zip(nxt.state_values, curr.state_values, strict=False)
                        )
                    )
                else:
                    continue
            else:
                # Use eval score change as proxy for state transition
                delta_mag = abs(nxt.eval_score - curr.eval_score)

            transitions.append((action, delta_mag))

        if len(transitions) < 2:
            return 0.0

        # H(delta_S): unconditional entropy of transition magnitudes
        all_bins: Counter = Counter()
        for _, delta in transitions:
            all_bins[_discretise(delta, self._num_bins)] += 1
        h_delta = _entropy(all_bins)

        if h_delta == 0.0:
            return 0.0  # no variation in transitions

        # H(delta_S | A): conditional entropy — average entropy per action
        action_groups: dict[str, Counter] = {}
        for action, delta in transitions:
            if action not in action_groups:
                action_groups[action] = Counter()
            action_groups[action][_discretise(delta, self._num_bins)] += 1

        n_total = len(transitions)
        h_cond = 0.0
        for _action, bins in action_groups.items():
            n_action = sum(bins.values())
            h_cond += (n_action / n_total) * _entropy(bins)

        # Normalised mutual information
        empowerment = (h_delta - h_cond) / h_delta
        return max(0.0, min(1.0, empowerment))

    def describe(self) -> str:
        return (
            "Empowerment: I(A; S'|S) — normalised mutual information between "
            "actions and state transitions. Range [0, 1]. Higher = actions "
            "have more influence on state transitions."
        )
