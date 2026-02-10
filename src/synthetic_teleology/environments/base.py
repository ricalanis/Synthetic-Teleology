"""Base environment abstraction for the Synthetic Teleology framework.

Defines ``BaseEnvironment``, the abstract base class for all environments
that teleological agents interact with.  Environments are responsible for:

* Maintaining world state
* Applying agent actions to produce new states
* Providing observations to agents
* Accepting external perturbations
* Emitting domain events (PerturbationInjected) through the event bus
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from synthetic_teleology.domain.enums import StateSource
from synthetic_teleology.domain.events import PerturbationInjected
from synthetic_teleology.domain.values import ActionSpec, StateSnapshot
from synthetic_teleology.infrastructure.event_bus import EventBus

logger = logging.getLogger(__name__)


class BaseEnvironment(ABC):
    """Abstract base class for teleological environments.

    All environments must implement :meth:`step`, :meth:`observe`, and
    :meth:`reset`.  The base class provides event emission, step counting,
    and perturbation injection infrastructure.

    Parameters
    ----------
    event_bus:
        Event bus for publishing state-change and perturbation events.
        If ``None``, a new bus is created (events will not reach external
        subscribers unless they subscribe to this bus).
    env_id:
        Optional identifier for this environment instance.
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        env_id: str = "env",
    ) -> None:
        self._event_bus = event_bus or EventBus()
        self._env_id = env_id
        self._step_count: int = 0
        self._perturbation_history: list[dict[str, Any]] = []

    # -- public read-only properties ----------------------------------------

    @property
    def env_id(self) -> str:
        """Identifier for this environment instance."""
        return self._env_id

    @property
    def event_bus(self) -> EventBus:
        """The event bus this environment publishes to."""
        return self._event_bus

    @property
    def step_count(self) -> int:
        """Number of steps executed since last reset."""
        return self._step_count

    @property
    def perturbation_history(self) -> list[dict[str, Any]]:
        """Chronological list of all perturbations injected."""
        return list(self._perturbation_history)

    # -- abstract interface -------------------------------------------------

    @abstractmethod
    def step(self, action: ActionSpec, **kwargs: Any) -> StateSnapshot:
        """Apply an action to the environment and return the resulting state.

        Implementations should:
        1. Validate the action.
        2. Update internal state according to the action and dynamics.
        3. Increment step count (call ``_post_step``).
        4. Return a new :class:`StateSnapshot`.

        Parameters
        ----------
        action:
            The action to apply.
        **kwargs:
            Additional context (e.g. ``agent_id`` for multi-agent envs).

        Returns
        -------
        StateSnapshot
            The new state after applying the action.
        """
        ...

    @abstractmethod
    def observe(self, **kwargs: Any) -> StateSnapshot:
        """Return the current state without modifying the environment.

        Parameters
        ----------
        **kwargs:
            Additional context (e.g. ``agent_id`` for per-agent views).

        Returns
        -------
        StateSnapshot
            A snapshot of the current state.
        """
        ...

    @abstractmethod
    def reset(self) -> StateSnapshot:
        """Reset the environment to its initial configuration.

        Implementations should clear step count and perturbation history
        (call ``_post_reset``).

        Returns
        -------
        StateSnapshot
            The initial state after reset.
        """
        ...

    # -- perturbation injection ---------------------------------------------

    def inject_perturbation(self, perturbation: dict[str, Any]) -> None:
        """Inject an external perturbation (shock) into the environment.

        The base implementation records the perturbation and emits a
        :class:`PerturbationInjected` event.  Subclasses should override
        :meth:`_apply_perturbation` to define how the perturbation affects
        internal state.

        Parameters
        ----------
        perturbation:
            Dictionary describing the perturbation.  Keys and semantics
            are environment-specific.
        """
        self._perturbation_history.append(perturbation)
        self._apply_perturbation(perturbation)

        # Extract structured fields from the perturbation dict for the event
        perturbation_type = str(perturbation.get("type", "external"))
        magnitude = float(perturbation.get("magnitude", 0.0))
        affected_dims = tuple(perturbation.get("affected_dimensions", ()))

        self._event_bus.publish(
            PerturbationInjected(
                source_id=self._env_id,
                perturbation_type=perturbation_type,
                magnitude=magnitude,
                affected_dimensions=affected_dims,
                metadata=perturbation,
                timestamp=time.time(),
            )
        )

        logger.info(
            "Environment %s: perturbation injected at step %d: %s",
            self._env_id,
            self._step_count,
            perturbation,
        )

    def _apply_perturbation(self, perturbation: dict[str, Any]) -> None:
        """Apply perturbation effects to internal state.

        Default implementation is a no-op.  Subclasses should override
        this to implement perturbation-specific logic.

        Parameters
        ----------
        perturbation:
            The perturbation specification dictionary.
        """

    # -- post-action helpers ------------------------------------------------

    def _post_step(self, snapshot: StateSnapshot) -> None:
        """Housekeeping after a step: increment counter.

        Call this at the end of every :meth:`step` implementation.

        Parameters
        ----------
        snapshot:
            The state snapshot produced by the step.
        """
        self._step_count += 1

    def _post_reset(self) -> None:
        """Housekeeping after a reset: clear counters and history."""
        self._step_count = 0
        self._perturbation_history.clear()

    # -- snapshot construction helpers --------------------------------------

    def _make_snapshot(
        self,
        values: tuple[float, ...],
        source: StateSource = StateSource.ENVIRONMENT,
        metadata: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """Construct a :class:`StateSnapshot` with the current timestamp.

        Parameters
        ----------
        values:
            The state values.
        source:
            Origin of the observation.
        metadata:
            Optional metadata dictionary.

        Returns
        -------
        StateSnapshot
            A new frozen snapshot.
        """
        return StateSnapshot(
            timestamp=time.time(),
            values=values,
            source=source,
            metadata=metadata or {},
        )

    # -- dunder helpers -----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"env_id={self._env_id!r}, "
            f"steps={self._step_count})"
        )
