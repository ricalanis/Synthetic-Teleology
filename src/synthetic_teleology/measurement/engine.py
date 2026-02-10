"""Metrics engine -- composite runner for all measurement metrics.

The :class:`MetricsEngine` aggregates :class:`BaseMetric` instances and
exposes a single entry point to compute them all against one or more
:class:`AgentLog` objects.  It ships with a **default set** containing all
seven canonical metrics; callers may add, remove, or replace individual
metrics before running.

Usage::

    engine = MetricsEngine()                # includes all 7 defaults
    results = engine.compute_all(agent_log)
    report = engine.build_report("agent-1", agent_log)
"""

from __future__ import annotations

import time
from typing import Sequence

from synthetic_teleology.measurement.collector import AgentLog
from synthetic_teleology.measurement.metrics.base import BaseMetric, MetricResult
from synthetic_teleology.measurement.metrics.adaptivity import Adaptivity
from synthetic_teleology.measurement.metrics.goal_persistence import GoalPersistence
from synthetic_teleology.measurement.metrics.innovation_yield import InnovationYield
from synthetic_teleology.measurement.metrics.lyapunov_stability import LyapunovStability
from synthetic_teleology.measurement.metrics.normative_fidelity import NormativeFidelity
from synthetic_teleology.measurement.metrics.reflective_efficiency import ReflectiveEfficiency
from synthetic_teleology.measurement.metrics.teleological_coherence import TeleologicalCoherence
from synthetic_teleology.measurement.report import MetricsReport


def _default_metrics() -> list[BaseMetric]:
    """Return the canonical set of seven teleological metrics."""
    return [
        GoalPersistence(),
        TeleologicalCoherence(),
        ReflectiveEfficiency(),
        Adaptivity(),
        NormativeFidelity(),
        InnovationYield(),
        LyapunovStability(),
    ]


class MetricsEngine:
    """Composite metric runner.

    Parameters
    ----------
    metrics:
        Optional explicit list of metrics.  If ``None`` (the default),
        all seven canonical metrics are registered automatically.
    """

    def __init__(self, metrics: Sequence[BaseMetric] | None = None) -> None:
        if metrics is not None:
            self._metrics: list[BaseMetric] = list(metrics)
        else:
            self._metrics = _default_metrics()

    # -- mutation -------------------------------------------------------------

    def add_metric(self, metric: BaseMetric) -> None:
        """Append a metric to the engine.

        Raises :class:`ValueError` if a metric with the same name already
        exists.
        """
        existing_names = {m.name for m in self._metrics}
        if metric.name in existing_names:
            raise ValueError(
                f"Metric with name {metric.name!r} is already registered"
            )
        self._metrics.append(metric)

    def remove_metric(self, name: str) -> bool:
        """Remove the metric with the given *name*.

        Returns ``True`` if a metric was removed, ``False`` if not found.
        """
        before = len(self._metrics)
        self._metrics = [m for m in self._metrics if m.name != name]
        return len(self._metrics) < before

    def replace_metric(self, metric: BaseMetric) -> None:
        """Replace the metric with the same name, or add it if absent."""
        self._metrics = [m for m in self._metrics if m.name != metric.name]
        self._metrics.append(metric)

    # -- query ----------------------------------------------------------------

    @property
    def metric_names(self) -> list[str]:
        """Return the names of all registered metrics in order."""
        return [m.name for m in self._metrics]

    def get_metric(self, name: str) -> BaseMetric | None:
        """Return the registered metric with the given *name*, or ``None``."""
        for m in self._metrics:
            if m.name == name:
                return m
        return None

    # -- computation ----------------------------------------------------------

    def compute_all(self, log: AgentLog) -> list[MetricResult]:
        """Compute every registered metric against *log*.

        Returns a list of :class:`MetricResult` in registration order.
        Metrics whose ``validate`` fails return a result with
        ``value=0.0`` and an appropriate explanation.
        """
        return [metric.compute(log) for metric in self._metrics]

    def compute_all_agents(
        self,
        logs: dict[str, AgentLog],
    ) -> dict[str, list[MetricResult]]:
        """Compute every metric for every agent in *logs*.

        Returns a dict mapping ``agent_id -> list[MetricResult]``.
        """
        return {
            agent_id: self.compute_all(log) for agent_id, log in logs.items()
        }

    # -- report generation ----------------------------------------------------

    def build_report(self, agent_id: str, log: AgentLog) -> MetricsReport:
        """Compute all metrics and package them into a :class:`MetricsReport`.

        Parameters
        ----------
        agent_id:
            Identifier for the agent (embedded in the report).
        log:
            The agent's log to evaluate.
        """
        results = self.compute_all(log)
        return MetricsReport(
            agent_id=agent_id,
            results=tuple(results),
            timestamp=time.time(),
            metadata={
                "num_steps": log.num_steps,
                "num_revisions": log.revision_count,
                "num_metrics": len(self._metrics),
            },
        )

    def build_all_reports(
        self,
        logs: dict[str, AgentLog],
    ) -> dict[str, MetricsReport]:
        """Build reports for every agent in *logs*."""
        return {
            agent_id: self.build_report(agent_id, log)
            for agent_id, log in logs.items()
        }
