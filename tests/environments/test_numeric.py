"""Tests for NumericEnvironment."""

from __future__ import annotations

import numpy as np
import pytest

from synthetic_teleology.domain.values import ActionSpec, StateSnapshot
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.infrastructure.event_bus import EventBus


class TestNumericEnvironment:
    """Test NumericEnvironment step, observe, reset, perturbation, bounds."""

    def test_default_creation(self) -> None:
        env = NumericEnvironment(dimensions=3)
        assert env.dimensions == 3
        assert env.step_count == 0
        assert env.noise_std == 0.0
        np.testing.assert_array_equal(env.state_array, np.zeros(3))

    def test_custom_initial_state(self) -> None:
        env = NumericEnvironment(dimensions=2, initial_state=(1.0, 2.0))
        np.testing.assert_array_almost_equal(env.state_array, [1.0, 2.0])

    def test_observe_returns_snapshot(self) -> None:
        env = NumericEnvironment(dimensions=2, initial_state=(3.0, 4.0))
        state = env.observe()
        assert isinstance(state, StateSnapshot)
        assert state.values == pytest.approx((3.0, 4.0))

    def test_step_applies_delta(self) -> None:
        env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
        action = ActionSpec(name="move", parameters={"delta": (1.0, -0.5)})
        state = env.step(action)
        assert state.values == pytest.approx((1.0, -0.5))
        assert env.step_count == 1

    def test_step_no_delta_is_noop(self) -> None:
        env = NumericEnvironment(dimensions=2, initial_state=(3.0, 4.0))
        action = ActionSpec(name="noop", parameters={})
        state = env.step(action)
        assert state.values == pytest.approx((3.0, 4.0))

    def test_step_wrong_delta_shape_raises(self) -> None:
        env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
        action = ActionSpec(name="bad", parameters={"delta": (1.0, 2.0, 3.0)})
        with pytest.raises(ValueError, match="delta shape"):
            env.step(action)

    def test_multiple_steps(self) -> None:
        env = NumericEnvironment(dimensions=1, initial_state=(0.0,))
        for _ in range(5):
            action = ActionSpec(name="step", parameters={"delta": (1.0,)})
            env.step(action)
        assert env.state_array == pytest.approx([5.0])
        assert env.step_count == 5

    def test_reset(self) -> None:
        env = NumericEnvironment(dimensions=2, initial_state=(1.0, 2.0))
        env.step(ActionSpec(name="move", parameters={"delta": (5.0, 5.0)}))
        assert env.step_count == 1
        state = env.reset()
        assert state.values == pytest.approx((1.0, 2.0))
        assert env.step_count == 0

    def test_bounds_clamping(self) -> None:
        env = NumericEnvironment(
            dimensions=2,
            initial_state=(5.0, 5.0),
            bounds=(0.0, 10.0),
        )
        # Push above upper bound
        action = ActionSpec(name="big", parameters={"delta": (20.0, 0.0)})
        state = env.step(action)
        assert state.values[0] == pytest.approx(10.0)
        assert state.values[1] == pytest.approx(5.0)

    def test_bounds_lower_clamping(self) -> None:
        env = NumericEnvironment(
            dimensions=2,
            initial_state=(1.0, 1.0),
            bounds=(0.0, 10.0),
        )
        action = ActionSpec(name="neg", parameters={"delta": (-5.0, -5.0)})
        state = env.step(action)
        assert state.values[0] == pytest.approx(0.0)
        assert state.values[1] == pytest.approx(0.0)

    def test_noise_introduces_randomness(self) -> None:
        np.random.seed(42)
        env = NumericEnvironment(
            dimensions=2,
            initial_state=(0.0, 0.0),
            noise_std=1.0,
        )
        action = ActionSpec(name="noop", parameters={"delta": (0.0, 0.0)})
        state = env.step(action)
        # With noise_std=1.0, state should not remain at exactly (0,0)
        assert state.values != pytest.approx((0.0, 0.0), abs=0.01)

    def test_custom_dynamics_fn(self) -> None:
        def double_dynamics(state, delta):
            return state + 2.0 * delta

        env = NumericEnvironment(
            dimensions=1,
            initial_state=(0.0,),
            dynamics_fn=double_dynamics,
        )
        action = ActionSpec(name="move", parameters={"delta": (1.0,)})
        state = env.step(action)
        assert state.values == pytest.approx((2.0,))

    def test_perturbation_shift(self) -> None:
        env = NumericEnvironment(dimensions=2, initial_state=(5.0, 5.0))
        env.inject_perturbation({"shift": (2.0, -1.0)})
        state = env.observe()
        assert state.values == pytest.approx((7.0, 4.0))

    def test_perturbation_set_state(self) -> None:
        env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
        env.inject_perturbation({"set_state": (10.0, 20.0)})
        state = env.observe()
        assert state.values == pytest.approx((10.0, 20.0))

    def test_perturbation_scale(self) -> None:
        env = NumericEnvironment(dimensions=2, initial_state=(5.0, 10.0))
        env.inject_perturbation({"scale": 2.0})
        state = env.observe()
        assert state.values == pytest.approx((10.0, 20.0))

    def test_perturbation_noise_std_update(self) -> None:
        env = NumericEnvironment(dimensions=2, noise_std=0.0)
        assert env.noise_std == 0.0
        env.inject_perturbation({"noise_std": 0.5})
        assert env.noise_std == 0.5

    def test_perturbation_history(self) -> None:
        env = NumericEnvironment(dimensions=1)
        env.inject_perturbation({"shift": (1.0,)})
        env.inject_perturbation({"scale": 2.0})
        assert len(env.perturbation_history) == 2

    def test_invalid_dimensions_raises(self) -> None:
        with pytest.raises(ValueError, match="dimensions must be >= 1"):
            NumericEnvironment(dimensions=0)

    def test_initial_state_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_state shape"):
            NumericEnvironment(dimensions=2, initial_state=(1.0, 2.0, 3.0))

    def test_state_array_returns_copy(self) -> None:
        env = NumericEnvironment(dimensions=2, initial_state=(1.0, 2.0))
        arr = env.state_array
        arr[0] = 999.0
        # Original should not be mutated
        assert env.state_array[0] == pytest.approx(1.0)

    def test_env_id(self) -> None:
        env = NumericEnvironment(dimensions=2, env_id="my-env")
        assert env.env_id == "my-env"

    def test_repr(self) -> None:
        env = NumericEnvironment(dimensions=3, noise_std=0.1)
        r = repr(env)
        assert "NumericEnvironment" in r
        assert "3" in r
