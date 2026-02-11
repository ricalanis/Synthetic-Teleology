"""Metrics report value object.

:class:`MetricsReport` is an immutable summary of all metric results for
a single agent run.  It supports lookup by metric name, dictionary
serialisation, and a human-readable text summary.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

from synthetic_teleology.measurement.metrics.base import MetricResult


@dataclass(frozen=True)
class MetricsReport:
    """Immutable report containing all metric results for one agent.

    Attributes
    ----------
    agent_id:
        Identifier of the agent this report describes.
    results:
        Tuple of :class:`MetricResult` instances in computation order.
    timestamp:
        Unix timestamp of when the report was generated.
    metadata:
        Arbitrary extra data (e.g. run parameters, environment info).
    """

    agent_id: str
    results: tuple[MetricResult, ...]
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- lookup ---------------------------------------------------------------

    def get_metric(self, name: str) -> MetricResult | None:
        """Return the :class:`MetricResult` with the given *name*, or ``None``.

        Parameters
        ----------
        name:
            Metric identifier to look up (e.g. ``"goal_persistence"``).
        """
        for r in self.results:
            if r.name == name:
                return r
        return None

    @property
    def metric_names(self) -> list[str]:
        """Return the names of all metrics in this report."""
        return [r.name for r in self.results]

    @property
    def values(self) -> dict[str, float]:
        """Return a dict mapping metric name to its scalar value."""
        return {r.name: r.value for r in self.results}

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a plain dictionary.

        Returns a structure suitable for JSON serialisation::

            {
                "agent_id": "...",
                "timestamp": 1234567890.0,
                "timestamp_iso": "2025-01-01T00:00:00",
                "metadata": {...},
                "metrics": {
                    "goal_persistence": {
                        "value": 0.85,
                        "explanation": "...",
                        "metadata": {}
                    },
                    ...
                }
            }
        """
        iso = datetime.datetime.fromtimestamp(
            self.timestamp, tz=datetime.UTC
        ).isoformat()

        metrics_dict: dict[str, Any] = {}
        for r in self.results:
            metrics_dict[r.name] = {
                "value": r.value,
                "explanation": r.explanation,
                "metadata": dict(r.metadata) if r.metadata else {},
            }

        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "timestamp_iso": iso,
            "metadata": dict(self.metadata),
            "metrics": metrics_dict,
        }

    # -- human-readable summary -----------------------------------------------

    def summary(self) -> str:
        """Return a human-readable table string.

        Example output::

            === Metrics Report: agent-1 ===
            Metric                      Value
            -----------------------------------
            goal_persistence            0.8500
            teleological_coherence      0.7200
            ...
            -----------------------------------
        """
        header = f"=== Metrics Report: {self.agent_id} ==="
        col_metric = "Metric"
        col_value = "Value"
        metric_width = max(
            len(col_metric),
            *(len(r.name) for r in self.results),
        ) if self.results else len(col_metric)
        value_width = max(len(col_value), 8)

        sep = "-" * (metric_width + value_width + 4)
        lines: list[str] = [
            header,
            f"{col_metric:<{metric_width}}  {col_value:>{value_width}}",
            sep,
        ]

        for r in self.results:
            lines.append(f"{r.name:<{metric_width}}  {r.value:>{value_width}.4f}")

        lines.append(sep)

        # Add metadata summary if present
        if self.metadata:
            lines.append("")
            for key, val in self.metadata.items():
                lines.append(f"  {key}: {val}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        metric_count = len(self.results)
        return (
            f"MetricsReport(agent_id={self.agent_id!r}, "
            f"metrics={metric_count})"
        )
