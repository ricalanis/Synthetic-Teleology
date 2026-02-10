"""Competing resource allocation environment with scarcity dynamics.

``ResourceEnvironment`` models a world with multiple finite resources that
agents can allocate, consume, and release.  Resources regenerate over time
at a configurable rate, creating scarcity pressure that forces agents to
balance competing needs.

Typical use: testing goal-revision under resource constraints, multi-agent
resource competition, and constraint satisfaction scenarios.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from synthetic_teleology.domain.values import ActionSpec, StateSnapshot
from synthetic_teleology.environments.base import BaseEnvironment
from synthetic_teleology.infrastructure.event_bus import EventBus

logger = logging.getLogger(__name__)


class ResourceEnvironment(BaseEnvironment):
    """Resource allocation environment with regeneration and scarcity.

    State is represented as a vector of current resource levels.  Agents
    interact by allocating (consuming) or releasing resources.

    Parameters
    ----------
    num_resources:
        Number of distinct resource types.
    total_resources:
        Maximum capacity per resource.  Can be a scalar (same for all)
        or a tuple (one per resource type).
    initial_levels:
        Starting resource levels.  If ``None``, all resources start at
        their maximum capacity.
    regeneration_rate:
        Per-step regeneration rate (fraction of deficit recovered).
        Can be a scalar (same for all) or a tuple (one per resource type).
        Value in ``[0.0, 1.0]``.
    consumption_efficiency:
        Fraction of requested allocation actually consumed.  Models
        inefficiency or waste.  Default is ``1.0`` (perfect efficiency).
    event_bus:
        Event bus for domain events.
    env_id:
        Identifier for this environment.

    Example
    -------
    ::

        env = ResourceEnvironment(
            num_resources=3,
            total_resources=100.0,
            regeneration_rate=0.05,
        )
        state = env.reset()
        # Consume 10 units of resource 0, 5 of resource 1
        action = ActionSpec(
            name="allocate",
            parameters={"allocations": {0: 10.0, 1: 5.0}}
        )
        new_state = env.step(action)
    """

    def __init__(
        self,
        num_resources: int = 3,
        total_resources: float | tuple[float, ...] = 100.0,
        initial_levels: tuple[float, ...] | None = None,
        regeneration_rate: float | tuple[float, ...] = 0.05,
        consumption_efficiency: float = 1.0,
        event_bus: EventBus | None = None,
        env_id: str = "resource-env",
    ) -> None:
        super().__init__(event_bus=event_bus, env_id=env_id)

        if num_resources < 1:
            raise ValueError(f"num_resources must be >= 1, got {num_resources}")

        self._num_resources = num_resources

        # Capacity per resource
        if isinstance(total_resources, (int, float)):
            self._capacity = np.full(num_resources, float(total_resources), dtype=np.float64)
        else:
            self._capacity = np.array(total_resources, dtype=np.float64)
            if self._capacity.shape != (num_resources,):
                raise ValueError(
                    f"total_resources shape {self._capacity.shape} does not "
                    f"match num_resources ({num_resources},)"
                )

        # Regeneration rate per resource
        if isinstance(regeneration_rate, (int, float)):
            self._regen_rate = np.full(num_resources, float(regeneration_rate), dtype=np.float64)
        else:
            self._regen_rate = np.array(regeneration_rate, dtype=np.float64)
            if self._regen_rate.shape != (num_resources,):
                raise ValueError(
                    f"regeneration_rate shape {self._regen_rate.shape} does not "
                    f"match num_resources ({num_resources},)"
                )

        self._consumption_efficiency = max(0.0, min(1.0, consumption_efficiency))

        # Initial levels
        if initial_levels is not None:
            self._initial_levels = np.array(initial_levels, dtype=np.float64)
            if self._initial_levels.shape != (num_resources,):
                raise ValueError(
                    f"initial_levels shape {self._initial_levels.shape} does not "
                    f"match num_resources ({num_resources},)"
                )
            # Clamp to capacity
            self._initial_levels = np.minimum(self._initial_levels, self._capacity)
            self._initial_levels = np.maximum(self._initial_levels, 0.0)
        else:
            self._initial_levels = self._capacity.copy()

        # Current mutable state
        self._levels: NDArray[np.float64] = self._initial_levels.copy()

        # Track cumulative consumption and production
        self._total_consumed = np.zeros(num_resources, dtype=np.float64)
        self._total_regenerated = np.zeros(num_resources, dtype=np.float64)

    # -- properties ---------------------------------------------------------

    @property
    def num_resources(self) -> int:
        """Number of resource types."""
        return self._num_resources

    @property
    def capacity(self) -> NDArray[np.float64]:
        """Maximum capacity per resource (read-only copy)."""
        return self._capacity.copy()

    @property
    def levels(self) -> NDArray[np.float64]:
        """Current resource levels (read-only copy)."""
        return self._levels.copy()

    @property
    def scarcity(self) -> NDArray[np.float64]:
        """Per-resource scarcity: ``1 - (level / capacity)``.

        Values near 0 mean abundant; values near 1 mean critically scarce.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                self._capacity > 0,
                1.0 - (self._levels / self._capacity),
                1.0,
            )

    @property
    def total_consumed(self) -> NDArray[np.float64]:
        """Cumulative consumption per resource since last reset."""
        return self._total_consumed.copy()

    @property
    def total_regenerated(self) -> NDArray[np.float64]:
        """Cumulative regeneration per resource since last reset."""
        return self._total_regenerated.copy()

    # -- BaseEnvironment implementation -------------------------------------

    def step(self, action: ActionSpec, **kwargs: Any) -> StateSnapshot:
        """Apply resource allocation/release and regenerate.

        The action's ``parameters`` dict should contain one of:

        * ``"allocations"`` -- dict mapping resource index -> amount to consume.
          Positive values consume; negative values release.
        * ``"consume_all"`` -- consume a fraction (0-1) of each resource.
        * ``"release_all"`` -- release all consumed resources (reset to capacity).

        After applying the action, resources regenerate toward capacity
        at the configured rate.

        Parameters
        ----------
        action:
            Action specification.

        Returns
        -------
        StateSnapshot
            State after action + regeneration.
        """
        consumed_this_step = np.zeros(self._num_resources, dtype=np.float64)

        # Apply allocations
        allocations = action.parameters.get("allocations", None)
        if allocations is not None:
            for idx_key, amount in allocations.items():
                idx = int(idx_key)
                if 0 <= idx < self._num_resources:
                    effective = float(amount) * self._consumption_efficiency
                    if effective > 0:
                        # Consume: cannot go below 0
                        actual = min(effective, self._levels[idx])
                        self._levels[idx] -= actual
                        consumed_this_step[idx] += actual
                    else:
                        # Release: cannot exceed capacity
                        released = min(-effective, self._capacity[idx] - self._levels[idx])
                        self._levels[idx] += released

        # Handle bulk operations
        if action.parameters.get("consume_all", False):
            fraction = float(action.parameters.get("consume_fraction", 0.1))
            consumed = self._levels * fraction * self._consumption_efficiency
            self._levels -= consumed
            consumed_this_step += consumed

        if action.parameters.get("release_all", False):
            self._levels = self._capacity.copy()

        # Regenerate
        deficit = self._capacity - self._levels
        regeneration = deficit * self._regen_rate
        self._levels += regeneration
        self._levels = np.minimum(self._levels, self._capacity)
        self._levels = np.maximum(self._levels, 0.0)

        # Track totals
        self._total_consumed += consumed_this_step
        self._total_regenerated += regeneration

        snapshot = self._make_snapshot(
            values=tuple(float(v) for v in self._levels),
            metadata={
                "step": self._step_count + 1,
                "action_name": action.name,
                "consumed": tuple(float(v) for v in consumed_this_step),
                "regenerated": tuple(float(v) for v in regeneration),
                "scarcity": tuple(float(v) for v in self.scarcity),
            },
        )
        self._post_step(snapshot)
        return snapshot

    def observe(self, **kwargs: Any) -> StateSnapshot:
        """Return current resource levels as a state snapshot.

        Returns
        -------
        StateSnapshot
            Current resource levels plus scarcity metadata.
        """
        return self._make_snapshot(
            values=tuple(float(v) for v in self._levels),
            metadata={
                "step": self._step_count,
                "scarcity": tuple(float(v) for v in self.scarcity),
                "capacity": tuple(float(v) for v in self._capacity),
            },
        )

    def reset(self) -> StateSnapshot:
        """Reset all resources to initial levels and clear counters.

        Returns
        -------
        StateSnapshot
            The initial state.
        """
        self._levels = self._initial_levels.copy()
        self._total_consumed = np.zeros(self._num_resources, dtype=np.float64)
        self._total_regenerated = np.zeros(self._num_resources, dtype=np.float64)
        self._post_reset()

        return self._make_snapshot(
            values=tuple(float(v) for v in self._levels),
            metadata={"step": 0, "reset": True},
        )

    # -- perturbation -------------------------------------------------------

    def _apply_perturbation(self, perturbation: dict[str, Any]) -> None:
        """Apply resource-specific perturbations.

        Supported keys:

        * ``"deplete"`` -- dict of ``{resource_idx: amount}`` to subtract.
        * ``"restore"`` -- dict of ``{resource_idx: amount}`` to add.
        * ``"set_levels"`` -- tuple/list to directly set resource levels.
        * ``"scale_capacity"`` -- scalar multiplier for all capacities.
        * ``"regen_rate"`` -- new regeneration rate (scalar or tuple).

        Parameters
        ----------
        perturbation:
            Dictionary describing the perturbation.
        """
        if "deplete" in perturbation:
            for idx_key, amount in perturbation["deplete"].items():
                idx = int(idx_key)
                if 0 <= idx < self._num_resources:
                    self._levels[idx] = max(0.0, self._levels[idx] - float(amount))

        if "restore" in perturbation:
            for idx_key, amount in perturbation["restore"].items():
                idx = int(idx_key)
                if 0 <= idx < self._num_resources:
                    self._levels[idx] = min(
                        self._capacity[idx],
                        self._levels[idx] + float(amount),
                    )

        if "set_levels" in perturbation:
            new_levels = np.array(perturbation["set_levels"], dtype=np.float64)
            if new_levels.shape == (self._num_resources,):
                self._levels = np.clip(new_levels, 0.0, self._capacity)
            else:
                logger.warning(
                    "Perturbation set_levels shape %s ignored (expected (%d,))",
                    new_levels.shape,
                    self._num_resources,
                )

        if "scale_capacity" in perturbation:
            scale = float(perturbation["scale_capacity"])
            self._capacity *= scale
            # Clamp levels to new capacity
            self._levels = np.minimum(self._levels, self._capacity)

        if "regen_rate" in perturbation:
            rate = perturbation["regen_rate"]
            if isinstance(rate, (int, float)):
                self._regen_rate = np.full(
                    self._num_resources, float(rate), dtype=np.float64
                )
            else:
                self._regen_rate = np.array(rate, dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"ResourceEnvironment("
            f"resources={self._num_resources}, "
            f"avg_scarcity={float(self.scarcity.mean()):.2f}, "
            f"steps={self._step_count})"
        )
