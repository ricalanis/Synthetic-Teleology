"""Tests for SharedEnvironment."""

from __future__ import annotations

import numpy as np
import pytest

from synthetic_teleology.domain.values import ActionSpec, StateSnapshot
from synthetic_teleology.environments.shared import SharedEnvironment


class TestSharedEnvironment:
    """Test SharedEnvironment multi-agent state management."""

    def test_creation(self) -> None:
        env = SharedEnvironment(global_dimensions=4, local_dimensions=2)
        assert env.global_dimensions == 4
        assert env.local_dimensions == 2
        assert env.total_observation_dimensions == 6
        assert env.num_agents == 0

    def test_register_agent(self) -> None:
        env = SharedEnvironment(global_dimensions=2, local_dimensions=2)
        env.register_agent("a1")
        assert "a1" in env.registered_agents
        assert env.num_agents == 1

    def test_register_duplicate_raises(self) -> None:
        env = SharedEnvironment(global_dimensions=2, local_dimensions=1)
        env.register_agent("a1")
        with pytest.raises(ValueError, match="already registered"):
            env.register_agent("a1")

    def test_unregister_agent(self) -> None:
        env = SharedEnvironment(global_dimensions=2, local_dimensions=1)
        env.register_agent("a1")
        env.unregister_agent("a1")
        assert "a1" not in env.registered_agents
        assert env.num_agents == 0

    def test_unregister_missing_raises(self) -> None:
        env = SharedEnvironment(global_dimensions=2, local_dimensions=1)
        with pytest.raises(KeyError, match="not registered"):
            env.unregister_agent("missing")

    def test_observe_without_agent_id_returns_global(self) -> None:
        env = SharedEnvironment(
            global_dimensions=3,
            local_dimensions=2,
            initial_global_state=(1.0, 2.0, 3.0),
        )
        state = env.observe()
        assert len(state.values) == 3
        assert state.values == pytest.approx((1.0, 2.0, 3.0))

    def test_observe_with_agent_id_returns_combined(self) -> None:
        env = SharedEnvironment(
            global_dimensions=2,
            local_dimensions=2,
            initial_global_state=(1.0, 2.0),
        )
        env.register_agent("a1", initial_local_state=(3.0, 4.0))
        state = env.observe(agent_id="a1")
        assert len(state.values) == 4
        assert state.values == pytest.approx((1.0, 2.0, 3.0, 4.0))

    def test_observe_unregistered_agent_raises(self) -> None:
        env = SharedEnvironment(global_dimensions=2, local_dimensions=1)
        with pytest.raises(ValueError, match="not registered"):
            env.observe(agent_id="missing")

    def test_step_with_delta_global(self) -> None:
        env = SharedEnvironment(
            global_dimensions=2,
            local_dimensions=1,
            initial_global_state=(0.0, 0.0),
        )
        env.register_agent("a1")
        action = ActionSpec(name="global_move", parameters={"delta_global": (1.0, 2.0)})
        state = env.step(action, agent_id="a1")
        assert isinstance(state, StateSnapshot)
        # Global state should have changed
        np.testing.assert_array_almost_equal(env.global_state, [1.0, 2.0])

    def test_step_with_delta_local(self) -> None:
        env = SharedEnvironment(
            global_dimensions=2,
            local_dimensions=2,
            initial_global_state=(0.0, 0.0),
        )
        env.register_agent("a1")
        action = ActionSpec(name="local_move", parameters={"delta_local": (3.0, 4.0)})
        env.step(action, agent_id="a1")
        local_state = env.get_local_state("a1")
        np.testing.assert_array_almost_equal(local_state, [3.0, 4.0])

    def test_step_without_agent_id_raises(self) -> None:
        env = SharedEnvironment(global_dimensions=2, local_dimensions=1)
        env.register_agent("a1")
        action = ActionSpec(name="move", parameters={})
        with pytest.raises(ValueError, match="requires agent_id"):
            env.step(action)

    def test_step_with_unregistered_agent_raises(self) -> None:
        env = SharedEnvironment(global_dimensions=2, local_dimensions=1)
        action = ActionSpec(name="move", parameters={})
        with pytest.raises(ValueError, match="not registered"):
            env.step(action, agent_id="ghost")

    def test_agent_isolation(self) -> None:
        """Two agents' local states are independent."""
        env = SharedEnvironment(
            global_dimensions=1,
            local_dimensions=2,
            initial_global_state=(0.0,),
        )
        env.register_agent("a1", initial_local_state=(0.0, 0.0))
        env.register_agent("a2", initial_local_state=(0.0, 0.0))

        # Only a1 moves locally
        action = ActionSpec(name="local_move", parameters={"delta_local": (5.0, 5.0)})
        env.step(action, agent_id="a1")

        local_a1 = env.get_local_state("a1")
        local_a2 = env.get_local_state("a2")
        np.testing.assert_array_almost_equal(local_a1, [5.0, 5.0])
        np.testing.assert_array_almost_equal(local_a2, [0.0, 0.0])

    def test_global_state_shared(self) -> None:
        """Both agents see the same global state change."""
        env = SharedEnvironment(
            global_dimensions=2,
            local_dimensions=1,
            initial_global_state=(0.0, 0.0),
        )
        env.register_agent("a1")
        env.register_agent("a2")

        action = ActionSpec(name="global_move", parameters={"delta_global": (1.0, 1.0)})
        env.step(action, agent_id="a1")

        obs_a1 = env.observe(agent_id="a1")
        obs_a2 = env.observe(agent_id="a2")
        # Both should see the global change (first 2 dims)
        assert obs_a1.values[:2] == pytest.approx((1.0, 1.0))
        assert obs_a2.values[:2] == pytest.approx((1.0, 1.0))

    def test_reset(self) -> None:
        env = SharedEnvironment(
            global_dimensions=2,
            local_dimensions=1,
            initial_global_state=(1.0, 2.0),
        )
        env.register_agent("a1")
        env.step(ActionSpec(name="move", parameters={"delta_global": (5.0, 5.0)}), agent_id="a1")
        state = env.reset()
        assert state.values == pytest.approx((1.0, 2.0))
        assert env.step_count == 0

    def test_get_all_observations(self) -> None:
        env = SharedEnvironment(global_dimensions=2, local_dimensions=1)
        env.register_agent("a1")
        env.register_agent("a2")
        obs = env.get_all_observations()
        assert "a1" in obs
        assert "a2" in obs
        assert len(obs) == 2

    def test_get_agent_distances(self) -> None:
        env = SharedEnvironment(global_dimensions=1, local_dimensions=2)
        env.register_agent("a1", initial_local_state=(0.0, 0.0))
        env.register_agent("a2", initial_local_state=(3.0, 4.0))
        distances = env.get_agent_distances()
        assert ("a1", "a2") in distances
        assert distances[("a1", "a2")] == pytest.approx(5.0)

    def test_get_local_state_unregistered_raises(self) -> None:
        env = SharedEnvironment(global_dimensions=1, local_dimensions=1)
        with pytest.raises(KeyError, match="not registered"):
            env.get_local_state("missing")

    def test_perturbation_global_shift(self) -> None:
        env = SharedEnvironment(
            global_dimensions=2,
            local_dimensions=1,
            initial_global_state=(0.0, 0.0),
        )
        env.inject_perturbation({"global_shift": (10.0, 20.0)})
        np.testing.assert_array_almost_equal(env.global_state, [10.0, 20.0])

    def test_perturbation_set_global(self) -> None:
        env = SharedEnvironment(
            global_dimensions=2,
            local_dimensions=1,
            initial_global_state=(0.0, 0.0),
        )
        env.inject_perturbation({"set_global": (99.0, 88.0)})
        np.testing.assert_array_almost_equal(env.global_state, [99.0, 88.0])

    def test_perturbation_disconnect_agent(self) -> None:
        env = SharedEnvironment(global_dimensions=1, local_dimensions=1)
        env.register_agent("a1")
        env.inject_perturbation({"disconnect_agent": "a1"})
        assert "a1" not in env.registered_agents

    def test_invalid_global_dimensions_raises(self) -> None:
        with pytest.raises(ValueError, match="global_dimensions must be >= 1"):
            SharedEnvironment(global_dimensions=0, local_dimensions=1)

    def test_invalid_local_dimensions_raises(self) -> None:
        with pytest.raises(ValueError, match="local_dimensions must be >= 0"):
            SharedEnvironment(global_dimensions=1, local_dimensions=-1)

    def test_repr(self) -> None:
        env = SharedEnvironment(global_dimensions=3, local_dimensions=2)
        r = repr(env)
        assert "SharedEnvironment" in r
