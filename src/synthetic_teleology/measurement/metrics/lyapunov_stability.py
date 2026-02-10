"""Lyapunov Stability (LS) metric.

Measures whether the agent's evaluation scores are converging over time,
inspired by Lyapunov stability analysis from dynamical systems theory.

Formula
-------
Divide the score time series into overlapping sliding windows and compute
the variance of each window.  Fit a linear regression to the sequence
of variances.  If the slope is negative (variance is decreasing), the
system is converging.

::

    variances = [var(scores[i:i+w]) for i in range(0, n-w+1, stride)]
    slope = linear_regression_slope(variances)
    LS = sigmoid(-slope * scale)

Range: [0, 1].

- ``LS > 0.5`` indicates convergence (decreasing variance).
- ``LS < 0.5`` indicates divergence (increasing variance).
- ``LS = 0.5`` is neutral.

Parameters
~~~~~~~~~~
``window_size`` (default 5): number of steps per sliding window.
``stride`` (default 1): step increment between successive windows.
``scale`` (default 10.0): sigmoid steepness.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from synthetic_teleology.measurement.collector import AgentLog
from synthetic_teleology.measurement.metrics.base import BaseMetric


class LyapunovStability(BaseMetric):
    """LS: convergence stability of evaluation scores via variance trend."""

    def __init__(
        self,
        window_size: int = 5,
        stride: int = 1,
        scale: float = 10.0,
    ) -> None:
        self._window_size = max(2, window_size)
        self._stride = max(1, stride)
        self._scale = scale

    @property
    def name(self) -> str:
        return "lyapunov_stability"

    def validate(self, log: AgentLog) -> bool:
        """Need enough data points to form at least 2 windows."""
        min_entries = self._window_size + self._stride
        return log.num_steps >= min_entries

    def _compute(self, log: AgentLog) -> float:
        scores: NDArray[np.float64] = np.array(log.get_scores(), dtype=np.float64)
        n = len(scores)
        w = self._window_size

        # Compute windowed variances
        variances: list[float] = []
        for i in range(0, n - w + 1, self._stride):
            window = scores[i : i + w]
            variances.append(float(np.var(window)))

        if len(variances) < 2:
            # Not enough windows to determine a trend
            return 0.5  # neutral

        # Linear regression: slope of variances over time
        slope = _linear_regression_slope(variances)

        # Apply sigmoid: negative slope (converging) -> LS > 0.5
        ls = _sigmoid(-slope * self._scale)
        return ls

    def describe(self) -> str:
        return (
            f"Lyapunov Stability (LS): convergence of eval score variance "
            f"(window={self._window_size}, stride={self._stride}). "
            f"LS > 0.5 = converging, LS < 0.5 = diverging."
        )


def _linear_regression_slope(values: list[float]) -> float:
    """Compute the slope of a simple linear regression y = mx + b.

    Uses numpy for numerical stability.  The independent variable is
    the integer index ``[0, 1, 2, ...]``.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = np.array(values, dtype=np.float64)
    # slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
    x_mean = x.mean()
    y_mean = y.mean()
    numerator = float(np.sum((x - x_mean) * (y - y_mean)))
    denominator = float(np.sum((x - x_mean) ** 2))
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _sigmoid(x: float) -> float:
    """Standard logistic sigmoid, clamped to avoid overflow."""
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))
