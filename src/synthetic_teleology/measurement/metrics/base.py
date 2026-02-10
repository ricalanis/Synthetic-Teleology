"""Base metric abstraction for the measurement layer.

:class:`BaseMetric` defines the **Template Method** pattern used by all
concrete metrics:

1. ``validate(log)`` -- check the log has sufficient data.
2. ``_compute(log)`` -- compute the raw scalar value (subclass responsibility).
3. ``describe()``    -- human-readable explanation.

The public entry point :meth:`compute` orchestrates these steps and returns
an immutable :class:`MetricResult`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from synthetic_teleology.measurement.collector import AgentLog


# ---------------------------------------------------------------------------
# Result value object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricResult:
    """Immutable result of a single metric computation.

    Attributes
    ----------
    name:
        Metric identifier (matches :attr:`BaseMetric.name`).
    value:
        Scalar metric value.  Semantics depend on the metric; most are
        normalised to [0, 1].
    metadata:
        Arbitrary extra data the metric wishes to expose (e.g. intermediate
        computations, thresholds used).
    explanation:
        Human-readable description or reasoning.
    """

    name: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""

    def __repr__(self) -> str:
        return f"MetricResult(name={self.name!r}, value={self.value:.4f})"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseMetric(ABC):
    """Template Method base class for all teleological metrics.

    Subclasses must implement:
    - ``name`` (property) -- unique metric identifier.
    - ``_compute(log)``   -- core computation returning a ``float``.

    Subclasses *may* override:
    - ``validate(log)``   -- guard; default requires >= 2 entries.
    - ``describe()``      -- human-readable explanation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric identifier, e.g. ``'goal_persistence'``."""
        ...

    @abstractmethod
    def _compute(self, log: AgentLog) -> float:
        """Core computation -- return the raw scalar value.

        Implementations can assume that ``validate(log)`` has already
        returned ``True``.
        """
        ...

    def validate(self, log: AgentLog) -> bool:
        """Check whether *log* contains enough data for meaningful computation.

        The default implementation requires at least 2 timestep entries.
        Override for metrics with different requirements.
        """
        return log.num_steps >= 2

    def describe(self) -> str:
        """Return a human-readable description of the metric."""
        return f"Metric: {self.name}"

    # -- template method (public API) -----------------------------------------

    def compute(self, log: AgentLog) -> MetricResult:
        """Compute the metric on *log* using the template-method pipeline.

        **Do not override** -- customise behaviour through the hook methods
        ``validate``, ``_compute``, and ``describe``.
        """
        if not self.validate(log):
            return MetricResult(
                name=self.name,
                value=0.0,
                explanation="Insufficient data for computation",
            )
        value = self._compute(log)
        return MetricResult(
            name=self.name,
            value=value,
            explanation=self.describe(),
        )
