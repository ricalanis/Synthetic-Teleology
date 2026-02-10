"""Metric implementations for the Synthetic Teleology measurement layer.

Each metric extends :class:`BaseMetric` using the Template Method pattern
and computes a scalar value from an :class:`AgentLog`.

Available metrics
~~~~~~~~~~~~~~~~~
- :class:`GoalPersistence` -- stability of goal commitment
- :class:`TeleologicalCoherence` -- alignment between actions and goals
- :class:`ReflectiveEfficiency` -- effectiveness of reflection phases
- :class:`Adaptivity` -- speed of recovery after perturbations
- :class:`NormativeFidelity` -- adherence to constraints
- :class:`InnovationYield` -- diversity and novelty of actions
- :class:`LyapunovStability` -- convergence stability of evaluation scores
"""

from synthetic_teleology.measurement.metrics.base import BaseMetric, MetricResult
from synthetic_teleology.measurement.metrics.adaptivity import Adaptivity
from synthetic_teleology.measurement.metrics.goal_persistence import GoalPersistence
from synthetic_teleology.measurement.metrics.innovation_yield import InnovationYield
from synthetic_teleology.measurement.metrics.lyapunov_stability import LyapunovStability
from synthetic_teleology.measurement.metrics.normative_fidelity import NormativeFidelity
from synthetic_teleology.measurement.metrics.reflective_efficiency import ReflectiveEfficiency
from synthetic_teleology.measurement.metrics.teleological_coherence import TeleologicalCoherence

__all__ = [
    "BaseMetric",
    "MetricResult",
    "Adaptivity",
    "GoalPersistence",
    "InnovationYield",
    "LyapunovStability",
    "NormativeFidelity",
    "ReflectiveEfficiency",
    "TeleologicalCoherence",
]
