"""Composable environment wrappers (Decorator pattern).

``EnvironmentWrapper`` is the base class for wrapping a
:class:`BaseEnvironment` with additional behavior. Wrappers can be
stacked (composed) to build custom environment pipelines.

Concrete wrappers:

* ``NoisyObservationWrapper`` -- adds Gaussian noise to observations.
* ``HistoryTrackingWrapper`` -- records all (action, snapshot) pairs, bounded.
* ``ResourceQuotaWrapper`` -- caps per-resource usage.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from synthetic_teleology.domain.values import ActionSpec, StateSnapshot
from synthetic_teleology.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)


class EnvironmentWrapper(BaseEnvironment):
    """Composable wrapper around a :class:`BaseEnvironment`.

    Delegates all methods to the wrapped environment by default.
    Subclasses override only the methods they want to modify.

    Parameters
    ----------
    env:
        The environment to wrap.
    """

    def __init__(self, env: BaseEnvironment) -> None:
        # Do NOT call super().__init__() — we delegate to the wrapped env
        self._env = env

    @property
    def unwrapped(self) -> BaseEnvironment:
        """The innermost (non-wrapper) environment."""
        if isinstance(self._env, EnvironmentWrapper):
            return self._env.unwrapped
        return self._env

    # -- delegate BaseEnvironment interface ---------------------------------

    @property
    def env_id(self) -> str:
        return self._env.env_id

    @property
    def event_bus(self):
        return self._env.event_bus

    @property
    def step_count(self) -> int:
        return self._env.step_count

    @property
    def perturbation_history(self) -> list[dict[str, Any]]:
        return self._env.perturbation_history

    def step(self, action: ActionSpec, **kwargs: Any) -> StateSnapshot:
        return self._env.step(action, **kwargs)

    def observe(self, **kwargs: Any) -> StateSnapshot:
        return self._env.observe(**kwargs)

    def reset(self) -> StateSnapshot:
        return self._env.reset()

    def inject_perturbation(self, perturbation: dict[str, Any]) -> None:
        return self._env.inject_perturbation(perturbation)

    def state_dict(self) -> dict[str, Any]:
        return self._env.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return self._env.load_state_dict(state)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._env!r})"


class NoisyObservationWrapper(EnvironmentWrapper):
    """Adds Gaussian noise to ``observe()`` values.

    Useful for simulating sensor noise or partial observability.

    Parameters
    ----------
    env:
        The environment to wrap.
    noise_std:
        Standard deviation of Gaussian noise added to each dimension.
    """

    def __init__(self, env: BaseEnvironment, noise_std: float = 0.1) -> None:
        super().__init__(env)
        self._noise_std = noise_std

    def observe(self, **kwargs: Any) -> StateSnapshot:
        snapshot = self._env.observe(**kwargs)
        if snapshot.values and self._noise_std > 0.0:
            noisy_values = tuple(
                v + float(np.random.normal(0.0, self._noise_std))
                for v in snapshot.values
            )
            return StateSnapshot(
                timestamp=snapshot.timestamp,
                values=noisy_values,
                source=snapshot.source,
                metadata={**snapshot.metadata, "noise_std": self._noise_std},
                observation=snapshot.observation,
            )
        return snapshot


class HistoryTrackingWrapper(EnvironmentWrapper):
    """Records all ``(action, snapshot)`` pairs from ``step()``, bounded.

    Parameters
    ----------
    env:
        The environment to wrap.
    max_history:
        Maximum number of entries to retain. Oldest entries are dropped.
    """

    def __init__(self, env: BaseEnvironment, max_history: int = 1000) -> None:
        super().__init__(env)
        self._max_history = max_history
        self._history: list[tuple[ActionSpec, StateSnapshot]] = []

    @property
    def history(self) -> list[tuple[ActionSpec, StateSnapshot]]:
        """Chronological list of (action, resulting_snapshot) pairs."""
        return list(self._history)

    def step(self, action: ActionSpec, **kwargs: Any) -> StateSnapshot:
        snapshot = self._env.step(action, **kwargs)
        self._history.append((action, snapshot))
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        return snapshot

    def reset(self) -> StateSnapshot:
        self._history.clear()
        return self._env.reset()


class ResourceQuotaWrapper(EnvironmentWrapper):
    """Caps per-resource usage, clamping action effects to quota limits.

    Parameters
    ----------
    env:
        The environment to wrap.
    quotas:
        Mapping from resource name/index to maximum consumption allowed
        per step. Actions exceeding quotas are clamped.
    strict:
        If ``True``, raise ``ValueError`` when quotas are exceeded.
        If ``False`` (default), silently clamp to the quota.
    """

    def __init__(
        self,
        env: BaseEnvironment,
        quotas: dict[str | int, float] | None = None,
        strict: bool = False,
    ) -> None:
        super().__init__(env)
        self._quotas = dict(quotas or {})
        self._strict = strict

    def step(self, action: ActionSpec, **kwargs: Any) -> StateSnapshot:
        if self._quotas and "allocations" in action.parameters:
            allocations = action.parameters["allocations"]
            clamped: dict[Any, float] = {}
            for key, amount in allocations.items():
                quota = self._quotas.get(key) or self._quotas.get(int(key))
                if quota is not None and abs(float(amount)) > quota:
                    if self._strict:
                        raise ValueError(
                            f"Resource {key} allocation {amount} exceeds quota {quota}"
                        )
                    sign = 1.0 if float(amount) >= 0 else -1.0
                    clamped[key] = sign * quota
                    logger.debug(
                        "ResourceQuotaWrapper: clamping resource %s from %.2f to %.2f",
                        key,
                        amount,
                        clamped[key],
                    )
                else:
                    clamped[key] = float(amount)
            action = ActionSpec(
                name=action.name,
                parameters={**action.parameters, "allocations": clamped},
            )
        return self._env.step(action, **kwargs)
