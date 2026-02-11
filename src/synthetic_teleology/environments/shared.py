"""Multi-agent shared environment with per-agent observations.

``SharedEnvironment`` acts as a mediator between multiple agents and a
common world state.  It maintains a shared global state plus per-agent
local states, merging them into agent-specific observations.  Actions
from any agent can affect both the global and local state partitions.

Typical use: multi-agent scenarios where agents share a world but have
private local state and potentially different views of the shared state.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from synthetic_teleology.domain.values import ActionSpec, StateSnapshot
from synthetic_teleology.environments.base import BaseEnvironment
from synthetic_teleology.infrastructure.event_bus import EventBus

logger = logging.getLogger(__name__)


class SharedEnvironment(BaseEnvironment):
    """Multi-agent shared environment with global + local state partitions.

    The environment maintains:

    * **global_state** -- shared state visible to (and affected by) all agents.
    * **local_state[agent_id]** -- per-agent private state.

    An agent's observation is the concatenation of the global state and its
    own local state: ``observation = global_state + local_state[agent_id]``.

    Parameters
    ----------
    global_dimensions:
        Number of shared state dimensions.
    local_dimensions:
        Number of per-agent private state dimensions.
    initial_global_state:
        Starting global state.  If ``None``, defaults to zeros.
    observation_noise_std:
        Gaussian noise added to observations (simulates imperfect sensing).
    action_effect_global:
        Fraction of an action's effect applied to global state (vs local).
    event_bus:
        Event bus for domain events.
    env_id:
        Identifier for this environment.

    Example
    -------
    ::

        env = SharedEnvironment(global_dimensions=4, local_dimensions=2)
        env.register_agent("agent-1")
        env.register_agent("agent-2")
        state = env.reset()

        obs_1 = env.observe(agent_id="agent-1")
        action = ActionSpec(name="move", parameters={"delta_global": (0.1, 0, 0, 0)})
        new_state = env.step(action, agent_id="agent-1")
    """

    def __init__(
        self,
        global_dimensions: int = 4,
        local_dimensions: int = 2,
        initial_global_state: tuple[float, ...] | None = None,
        observation_noise_std: float = 0.0,
        action_effect_global: float = 0.5,
        event_bus: EventBus | None = None,
        env_id: str = "shared-env",
    ) -> None:
        super().__init__(event_bus=event_bus, env_id=env_id)

        if global_dimensions < 1:
            raise ValueError(f"global_dimensions must be >= 1, got {global_dimensions}")
        if local_dimensions < 0:
            raise ValueError(f"local_dimensions must be >= 0, got {local_dimensions}")

        self._global_dims = global_dimensions
        self._local_dims = local_dimensions
        self._obs_noise_std = observation_noise_std
        self._action_effect_global = max(0.0, min(1.0, action_effect_global))

        # Global state
        if initial_global_state is not None:
            self._initial_global = np.array(initial_global_state, dtype=np.float64)
            if self._initial_global.shape != (global_dimensions,):
                raise ValueError(
                    f"initial_global_state shape {self._initial_global.shape} "
                    f"does not match global_dimensions ({global_dimensions},)"
                )
        else:
            self._initial_global = np.zeros(global_dimensions, dtype=np.float64)

        self._global_state: NDArray[np.float64] = self._initial_global.copy()

        # Per-agent local states
        self._local_states: dict[str, NDArray[np.float64]] = {}
        self._initial_local_states: dict[str, NDArray[np.float64]] = {}

        # Agent registry
        self._registered_agents: set[str] = set()

        # Action history for conflict detection
        self._action_history: list[dict[str, Any]] = []

    # -- properties ---------------------------------------------------------

    @property
    def global_dimensions(self) -> int:
        """Number of shared state dimensions."""
        return self._global_dims

    @property
    def local_dimensions(self) -> int:
        """Number of per-agent private state dimensions."""
        return self._local_dims

    @property
    def total_observation_dimensions(self) -> int:
        """Total dimensions in an agent's observation (global + local)."""
        return self._global_dims + self._local_dims

    @property
    def global_state(self) -> NDArray[np.float64]:
        """Current global state (read-only copy)."""
        return self._global_state.copy()

    @property
    def registered_agents(self) -> frozenset[str]:
        """Set of registered agent IDs."""
        return frozenset(self._registered_agents)

    @property
    def num_agents(self) -> int:
        """Number of registered agents."""
        return len(self._registered_agents)

    # -- agent management ---------------------------------------------------

    def register_agent(
        self,
        agent_id: str,
        initial_local_state: tuple[float, ...] | None = None,
    ) -> None:
        """Register an agent with the shared environment.

        Parameters
        ----------
        agent_id:
            Unique identifier for the agent.
        initial_local_state:
            Starting local state for this agent.  If ``None``, defaults
            to zeros.

        Raises
        ------
        ValueError
            If the agent ID is already registered.
        """
        if agent_id in self._registered_agents:
            raise ValueError(f"Agent {agent_id!r} is already registered")

        if initial_local_state is not None:
            local = np.array(initial_local_state, dtype=np.float64)
            if local.shape != (self._local_dims,):
                raise ValueError(
                    f"initial_local_state shape {local.shape} does not match "
                    f"local_dimensions ({self._local_dims},)"
                )
        else:
            local = np.zeros(self._local_dims, dtype=np.float64)

        self._registered_agents.add(agent_id)
        self._local_states[agent_id] = local.copy()
        self._initial_local_states[agent_id] = local.copy()

        logger.info("SharedEnvironment %s: registered agent %s", self._env_id, agent_id)

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the environment.

        Parameters
        ----------
        agent_id:
            Agent to remove.

        Raises
        ------
        KeyError
            If the agent is not registered.
        """
        if agent_id not in self._registered_agents:
            raise KeyError(f"Agent {agent_id!r} is not registered")

        self._registered_agents.discard(agent_id)
        self._local_states.pop(agent_id, None)
        self._initial_local_states.pop(agent_id, None)

        logger.info("SharedEnvironment %s: unregistered agent %s", self._env_id, agent_id)

    def get_local_state(self, agent_id: str) -> NDArray[np.float64]:
        """Return a read-only copy of an agent's local state.

        Parameters
        ----------
        agent_id:
            The agent whose local state to retrieve.

        Raises
        ------
        KeyError
            If the agent is not registered.
        """
        if agent_id not in self._local_states:
            raise KeyError(f"Agent {agent_id!r} is not registered")
        return self._local_states[agent_id].copy()

    # -- BaseEnvironment implementation -------------------------------------

    def step(self, action: ActionSpec, **kwargs: Any) -> StateSnapshot:
        """Apply an agent's action to both global and local state.

        The ``agent_id`` **must** be passed as a keyword argument.

        The action's ``parameters`` dict may contain:

        * ``"delta_global"`` -- additive delta for the global state.
        * ``"delta_local"`` -- additive delta for the agent's local state.
        * ``"delta"`` -- split between global and local according to
          ``action_effect_global``.

        Parameters
        ----------
        action:
            Action specification.
        **kwargs:
            Must include ``agent_id: str``.

        Returns
        -------
        StateSnapshot
            The agent's observation after the action (global + local).

        Raises
        ------
        ValueError
            If ``agent_id`` is missing or not registered.
        """
        agent_id = kwargs.get("agent_id")
        if agent_id is None:
            raise ValueError(
                "SharedEnvironment.step() requires agent_id keyword argument"
            )
        if agent_id not in self._registered_agents:
            raise ValueError(f"Agent {agent_id!r} is not registered")

        # Parse deltas
        delta_global = self._parse_delta(
            action.parameters.get("delta_global"), self._global_dims
        )
        delta_local = self._parse_delta(
            action.parameters.get("delta_local"), self._local_dims
        )

        # Handle unified delta: split between global and local
        unified_delta = action.parameters.get("delta")
        if unified_delta is not None:
            total_dims = self._global_dims + self._local_dims
            full_delta = np.array(unified_delta, dtype=np.float64)
            if full_delta.shape == (total_dims,):
                g_part = full_delta[: self._global_dims] * self._action_effect_global
                l_part = full_delta[self._global_dims :] * (1.0 - self._action_effect_global)
                delta_global = delta_global + g_part
                delta_local = delta_local + l_part
            elif full_delta.shape == (self._global_dims,):
                delta_global = delta_global + full_delta * self._action_effect_global
            else:
                logger.warning(
                    "SharedEnvironment: unexpected delta shape %s, ignoring",
                    full_delta.shape,
                )

        # Apply global delta
        self._global_state = self._global_state + delta_global

        # Apply local delta
        self._local_states[agent_id] = self._local_states[agent_id] + delta_local

        # Record action
        self._action_history.append({
            "step": self._step_count + 1,
            "agent_id": agent_id,
            "action_name": action.name,
            "timestamp": time.time(),
        })

        # Build observation for the acting agent
        snapshot = self._build_observation(
            agent_id,
            metadata={
                "step": self._step_count + 1,
                "acting_agent": agent_id,
                "action_name": action.name,
            },
        )
        self._post_step(snapshot)
        return snapshot

    def observe(self, **kwargs: Any) -> StateSnapshot:
        """Return an agent-specific observation (global + local).

        If ``agent_id`` is not provided, returns only the global state.

        Parameters
        ----------
        **kwargs:
            Optional ``agent_id: str`` for agent-specific observations.

        Returns
        -------
        StateSnapshot
            Observation for the specified agent, or global-only if no
            agent is specified.
        """
        agent_id = kwargs.get("agent_id")

        if agent_id is not None:
            if agent_id not in self._registered_agents:
                raise ValueError(f"Agent {agent_id!r} is not registered")
            return self._build_observation(
                agent_id,
                metadata={
                    "step": self._step_count,
                    "observing_agent": agent_id,
                },
            )

        # No agent specified: return global state only
        return self._make_snapshot(
            values=tuple(float(v) for v in self._global_state),
            metadata={
                "step": self._step_count,
                "scope": "global_only",
                "num_agents": self.num_agents,
            },
        )

    def reset(self) -> StateSnapshot:
        """Reset global and all local states to their initial configurations.

        Returns
        -------
        StateSnapshot
            The initial global state.
        """
        self._global_state = self._initial_global.copy()
        for agent_id in self._registered_agents:
            if agent_id in self._initial_local_states:
                self._local_states[agent_id] = self._initial_local_states[agent_id].copy()
            else:
                self._local_states[agent_id] = np.zeros(self._local_dims, dtype=np.float64)
        self._action_history.clear()
        self._post_reset()

        return self._make_snapshot(
            values=tuple(float(v) for v in self._global_state),
            metadata={"step": 0, "reset": True, "num_agents": self.num_agents},
        )

    # -- perturbation -------------------------------------------------------

    def _apply_perturbation(self, perturbation: dict[str, Any]) -> None:
        """Apply shared-environment perturbations.

        Supported keys:

        * ``"global_shift"`` -- additive vector to global state.
        * ``"set_global"`` -- directly set global state.
        * ``"agent_shift"`` -- dict of ``{agent_id: delta_vector}`` for local states.
        * ``"disconnect_agent"`` -- agent_id to unregister.
        * ``"noise_std"`` -- update observation noise.

        Parameters
        ----------
        perturbation:
            Dictionary describing the perturbation.
        """
        if "global_shift" in perturbation:
            shift = np.array(perturbation["global_shift"], dtype=np.float64)
            if shift.shape == (self._global_dims,):
                self._global_state = self._global_state + shift

        if "set_global" in perturbation:
            new_global = np.array(perturbation["set_global"], dtype=np.float64)
            if new_global.shape == (self._global_dims,):
                self._global_state = new_global

        if "agent_shift" in perturbation:
            for agent_id, delta_raw in perturbation["agent_shift"].items():
                if agent_id in self._local_states:
                    delta = np.array(delta_raw, dtype=np.float64)
                    if delta.shape == (self._local_dims,):
                        self._local_states[agent_id] += delta

        if "disconnect_agent" in perturbation:
            target = perturbation["disconnect_agent"]
            if target in self._registered_agents:
                self.unregister_agent(target)

        if "noise_std" in perturbation:
            self._obs_noise_std = float(perturbation["noise_std"])

    # -- multi-agent queries ------------------------------------------------

    def get_all_observations(self) -> dict[str, StateSnapshot]:
        """Return observations for all registered agents.

        Returns
        -------
        dict[str, StateSnapshot]
            Mapping from agent_id to their observation snapshot.
        """
        return {
            agent_id: self._build_observation(
                agent_id,
                metadata={"step": self._step_count, "observing_agent": agent_id},
            )
            for agent_id in self._registered_agents
        }

    def get_agent_distances(self) -> dict[tuple[str, str], float]:
        """Compute pairwise L2 distances between agent local states.

        Returns
        -------
        dict[tuple[str, str], float]
            Mapping from ``(agent_a, agent_b)`` to their local-state distance.
        """
        agents = sorted(self._registered_agents)
        distances: dict[tuple[str, str], float] = {}
        for i, a in enumerate(agents):
            for b in agents[i + 1 :]:
                dist = float(
                    np.linalg.norm(self._local_states[a] - self._local_states[b])
                )
                distances[(a, b)] = dist
        return distances

    # -- internal helpers ---------------------------------------------------

    def _build_observation(
        self,
        agent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """Construct an observation for a specific agent.

        The observation is ``global_state ++ local_state[agent_id]`` plus
        optional Gaussian noise.
        """
        global_obs = self._global_state.copy()
        local_obs = self._local_states[agent_id].copy()

        # Add observation noise
        if self._obs_noise_std > 0.0:
            global_obs += np.random.normal(0.0, self._obs_noise_std, self._global_dims)
            if self._local_dims > 0:
                local_obs += np.random.normal(0.0, self._obs_noise_std, self._local_dims)

        if self._local_dims > 0:
            combined = np.concatenate([global_obs, local_obs])
        else:
            combined = global_obs

        return self._make_snapshot(
            values=tuple(float(v) for v in combined),
            metadata=metadata or {},
        )

    def _parse_delta(
        self,
        raw: Any,
        expected_dims: int,
    ) -> NDArray[np.float64]:
        """Parse a raw delta value into a numpy array of the expected shape.

        Returns a zero vector if ``raw`` is ``None`` or has wrong shape.
        """
        if raw is None:
            return np.zeros(expected_dims, dtype=np.float64)
        arr = np.array(raw, dtype=np.float64)
        if arr.shape != (expected_dims,):
            logger.warning(
                "SharedEnvironment: delta shape %s does not match expected (%d,), ignoring",
                arr.shape,
                expected_dims,
            )
            return np.zeros(expected_dims, dtype=np.float64)
        return arr

    def __repr__(self) -> str:
        return (
            f"SharedEnvironment("
            f"global_dims={self._global_dims}, "
            f"local_dims={self._local_dims}, "
            f"agents={self.num_agents}, "
            f"steps={self._step_count})"
        )
