"""N-dimensional continuous numeric environment.

``NumericEnvironment`` models a continuous N-dimensional state space where
actions apply delta vectors to the current position.  Supports configurable
Gaussian noise and optional custom dynamics functions.

Typical use: testing goal-tracking agents in a smooth, continuous space
where the "distance to goal" is directly interpretable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from synthetic_teleology.domain.values import ActionSpec, StateSnapshot
from synthetic_teleology.environments.base import BaseEnvironment
from synthetic_teleology.infrastructure.event_bus import EventBus

logger = logging.getLogger(__name__)

# Type alias for custom dynamics functions
DynamicsFn = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


class NumericEnvironment(BaseEnvironment):
    """N-dimensional continuous environment with configurable dynamics.

    State is an N-dimensional real-valued vector.  Each :meth:`step` applies
    the action's ``"delta"`` parameter as an additive displacement, optionally
    passed through a custom dynamics function, plus Gaussian noise.

    Parameters
    ----------
    dimensions:
        Number of state dimensions.
    initial_state:
        Starting state vector.  If ``None``, defaults to the origin.
    noise_std:
        Standard deviation of per-dimension Gaussian noise added at each
        step.  Set to ``0.0`` for deterministic dynamics.
    dynamics_fn:
        Optional callable ``(state, delta) -> new_state`` for custom
        dynamics.  If ``None``, the default linear dynamics is used:
        ``new_state = state + delta + noise``.
    bounds:
        Optional ``(low, high)`` tuple for clamping state values.
        Each element is an N-dimensional array or scalar.  ``None`` means
        unbounded.
    event_bus:
        Event bus for domain events.
    env_id:
        Identifier for this environment.

    Example
    -------
    ::

        env = NumericEnvironment(dimensions=3, noise_std=0.01)
        state = env.reset()
        action = ActionSpec(name="move", parameters={"delta": (0.1, -0.2, 0.05)})
        new_state = env.step(action)
    """

    def __init__(
        self,
        dimensions: int = 2,
        initial_state: tuple[float, ...] | NDArray[np.float64] | None = None,
        noise_std: float = 0.0,
        dynamics_fn: DynamicsFn | None = None,
        bounds: tuple[NDArray[np.float64] | float | None, NDArray[np.float64] | float | None]
        | None = None,
        event_bus: EventBus | None = None,
        env_id: str = "numeric-env",
    ) -> None:
        super().__init__(event_bus=event_bus, env_id=env_id)

        if dimensions < 1:
            raise ValueError(f"dimensions must be >= 1, got {dimensions}")

        self._dimensions = dimensions
        self._noise_std = noise_std
        self._dynamics_fn = dynamics_fn

        # Initial state
        if initial_state is not None:
            self._initial_state = np.array(initial_state, dtype=np.float64)
            if self._initial_state.shape != (dimensions,):
                raise ValueError(
                    f"initial_state shape {self._initial_state.shape} does not "
                    f"match dimensions ({dimensions},)"
                )
        else:
            self._initial_state = np.zeros(dimensions, dtype=np.float64)

        # Current state (mutable)
        self._state: NDArray[np.float64] = self._initial_state.copy()

        # Bounds
        if bounds is not None:
            low, high = bounds
            self._low = (
                np.full(dimensions, low, dtype=np.float64)
                if isinstance(low, (int, float))
                else (np.array(low, dtype=np.float64) if low is not None else None)
            )
            self._high = (
                np.full(dimensions, high, dtype=np.float64)
                if isinstance(high, (int, float))
                else (np.array(high, dtype=np.float64) if high is not None else None)
            )
        else:
            self._low = None
            self._high = None

    # -- properties ---------------------------------------------------------

    @property
    def dimensions(self) -> int:
        """Number of state dimensions."""
        return self._dimensions

    @property
    def noise_std(self) -> float:
        """Standard deviation of per-step Gaussian noise."""
        return self._noise_std

    @property
    def state_array(self) -> NDArray[np.float64]:
        """Current state as a numpy array (read-only copy)."""
        return self._state.copy()

    # -- BaseEnvironment implementation -------------------------------------

    def step(self, action: ActionSpec, **kwargs: Any) -> StateSnapshot:
        """Apply action delta to the state with optional noise and dynamics.

        The action's ``parameters`` dict should contain a ``"delta"`` key
        with an iterable of floats matching the environment dimensions.
        If ``"delta"`` is missing, the step is a no-op (zero displacement).

        Parameters
        ----------
        action:
            Action specification.  Expected parameter: ``delta``.

        Returns
        -------
        StateSnapshot
            The state after applying the action.
        """
        # Parse delta from action parameters
        raw_delta = action.parameters.get("delta", None)
        if raw_delta is not None:
            delta = np.array(raw_delta, dtype=np.float64)
            if delta.shape != (self._dimensions,):
                raise ValueError(
                    f"Action delta shape {delta.shape} does not match "
                    f"environment dimensions ({self._dimensions},)"
                )
        else:
            delta = np.zeros(self._dimensions, dtype=np.float64)

        # Apply dynamics
        if self._dynamics_fn is not None:
            new_state = self._dynamics_fn(self._state, delta)
        else:
            new_state = self._state + delta

        # Add noise
        if self._noise_std > 0.0:
            noise = np.random.normal(0.0, self._noise_std, size=self._dimensions)
            new_state = new_state + noise

        # Clamp to bounds
        new_state = self._clamp(new_state)

        self._state = new_state

        snapshot = self._make_snapshot(
            values=tuple(float(v) for v in self._state),
            metadata={
                "step": self._step_count + 1,
                "action_name": action.name,
                "delta_applied": tuple(float(v) for v in delta),
            },
        )
        self._post_step(snapshot)
        return snapshot

    def observe(self, **kwargs: Any) -> StateSnapshot:
        """Return the current state without modifying the environment.

        Returns
        -------
        StateSnapshot
            A snapshot of the current state values.
        """
        return self._make_snapshot(
            values=tuple(float(v) for v in self._state),
            metadata={"step": self._step_count},
        )

    def reset(self) -> StateSnapshot:
        """Reset state to the initial configuration.

        Returns
        -------
        StateSnapshot
            The initial state.
        """
        self._state = self._initial_state.copy()
        self._post_reset()

        return self._make_snapshot(
            values=tuple(float(v) for v in self._state),
            metadata={"step": 0, "reset": True},
        )

    # -- state serialization -------------------------------------------------

    def _state_dict_impl(self) -> dict[str, Any]:
        return {
            "state": list(float(v) for v in self._state),
            "dimensions": self._dimensions,
            "noise_std": self._noise_std,
        }

    def _load_state_dict_impl(self, state: dict[str, Any]) -> None:
        if "state" in state:
            self._state = np.array(state["state"], dtype=np.float64)
        if "noise_std" in state:
            self._noise_std = float(state["noise_std"])

    # -- perturbation -------------------------------------------------------

    def _apply_perturbation(self, perturbation: dict[str, Any]) -> None:
        """Apply a perturbation to the numeric state.

        Supported perturbation keys:

        * ``"shift"`` -- additive displacement vector (tuple/list of floats).
        * ``"noise_std"`` -- temporarily override the noise standard deviation.
        * ``"set_state"`` -- directly set the state to a new vector.
        * ``"scale"`` -- multiplicative scaling factor (scalar or per-dim).

        Parameters
        ----------
        perturbation:
            Dictionary describing the perturbation.
        """
        if "shift" in perturbation:
            shift = np.array(perturbation["shift"], dtype=np.float64)
            self._state = self._clamp(self._state + shift)

        if "set_state" in perturbation:
            new = np.array(perturbation["set_state"], dtype=np.float64)
            if new.shape == (self._dimensions,):
                self._state = self._clamp(new)
            else:
                logger.warning(
                    "Perturbation set_state shape %s ignored (expected (%d,))",
                    new.shape,
                    self._dimensions,
                )

        if "scale" in perturbation:
            scale = perturbation["scale"]
            if isinstance(scale, (int, float)):
                self._state = self._clamp(self._state * scale)
            else:
                scale_arr = np.array(scale, dtype=np.float64)
                self._state = self._clamp(self._state * scale_arr)

        if "noise_std" in perturbation:
            self._noise_std = float(perturbation["noise_std"])

    # -- internal helpers ---------------------------------------------------

    def _clamp(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Clamp state to configured bounds."""
        result = state
        if self._low is not None:
            result = np.maximum(result, self._low)
        if self._high is not None:
            result = np.minimum(result, self._high)
        return result

    def __repr__(self) -> str:
        return (
            f"NumericEnvironment("
            f"dims={self._dimensions}, "
            f"noise_std={self._noise_std}, "
            f"steps={self._step_count})"
        )
