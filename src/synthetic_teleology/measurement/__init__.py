"""Measurement layer for the Synthetic Teleology framework.

Provides event collection, metric computation, reporting, and benchmarking
infrastructure for quantifying agent behaviour in the teleological loop.

Public API
----------
- :class:`EventCollector` -- subscribes to domain events and builds ``AgentLog``
- :class:`AgentLog` / :class:`AgentLogEntry` -- structured timestep logs
- :class:`MetricsEngine` -- composite metric runner
- :class:`MetricsReport` -- immutable report value object
- :class:`BaseMetric` / :class:`MetricResult` -- metric protocol and results
- Individual metrics: ``GoalPersistence``, ``TeleologicalCoherence``,
  ``ReflectiveEfficiency``, ``Adaptivity``, ``NormativeFidelity``,
  ``InnovationYield``, ``LyapunovStability``
"""

from synthetic_teleology.measurement.collector import (
    AgentLog,
    AgentLogEntry,
    EventCollector,
)
from synthetic_teleology.measurement.engine import MetricsEngine
from synthetic_teleology.measurement.metrics.base import BaseMetric, MetricResult
from synthetic_teleology.measurement.metrics.adaptivity import Adaptivity
from synthetic_teleology.measurement.metrics.goal_persistence import GoalPersistence
from synthetic_teleology.measurement.metrics.innovation_yield import InnovationYield
from synthetic_teleology.measurement.metrics.lyapunov_stability import LyapunovStability
from synthetic_teleology.measurement.metrics.normative_fidelity import NormativeFidelity
from synthetic_teleology.measurement.metrics.reflective_efficiency import ReflectiveEfficiency
from synthetic_teleology.measurement.metrics.teleological_coherence import TeleologicalCoherence
from synthetic_teleology.measurement.report import MetricsReport

__all__ = [
    # Collector
    "AgentLog",
    "AgentLogEntry",
    "EventCollector",
    # Engine & report
    "MetricsEngine",
    "MetricsReport",
    # Metric base
    "BaseMetric",
    "MetricResult",
    # Concrete metrics
    "Adaptivity",
    "GoalPersistence",
    "InnovationYield",
    "LyapunovStability",
    "NormativeFidelity",
    "ReflectiveEfficiency",
    "TeleologicalCoherence",
]
