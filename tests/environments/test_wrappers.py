"""Tests for EnvironmentWrapper and concrete wrappers."""

from __future__ import annotations

import pytest

from synthetic_teleology.domain.values import ActionSpec
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.environments.resource import ResourceEnvironment
from synthetic_teleology.environments.wrappers import (
    EnvironmentWrapper,
    HistoryTrackingWrapper,
    NoisyObservationWrapper,
    ResourceQuotaWrapper,
)


class TestEnvironmentWrapper:
    def test_delegates_step(self) -> None:
        env = NumericEnvironment(dimensions=2, noise_std=0.0)
        wrapped = EnvironmentWrapper(env)
        action = ActionSpec(name="move", parameters={"delta": (1.0, 0.0)})

        snap = wrapped.step(action)
        assert snap.values[0] == pytest.approx(1.0)
        assert wrapped.step_count == 1

    def test_delegates_observe(self) -> None:
        env = NumericEnvironment(dimensions=2, initial_state=(3.0, 4.0), noise_std=0.0)
        wrapped = EnvironmentWrapper(env)
        snap = wrapped.observe()
        assert snap.values == pytest.approx((3.0, 4.0))

    def test_delegates_reset(self) -> None:
        env = NumericEnvironment(dimensions=2, noise_std=0.0)
        wrapped = EnvironmentWrapper(env)
        wrapped.step(ActionSpec(name="move", parameters={"delta": (1.0, 0.0)}))
        wrapped.reset()
        assert wrapped.step_count == 0

    def test_unwrapped_returns_inner(self) -> None:
        env = NumericEnvironment(dimensions=2)
        w1 = EnvironmentWrapper(env)
        w2 = EnvironmentWrapper(w1)
        assert w2.unwrapped is env

    def test_delegates_state_dict(self) -> None:
        env = NumericEnvironment(dimensions=2, noise_std=0.0)
        wrapped = EnvironmentWrapper(env)
        sd = wrapped.state_dict()
        assert "state" in sd

    def test_repr(self) -> None:
        env = NumericEnvironment(dimensions=2)
        wrapped = EnvironmentWrapper(env)
        assert "EnvironmentWrapper" in repr(wrapped)


class TestNoisyObservationWrapper:
    def test_adds_noise_to_observation(self) -> None:
        env = NumericEnvironment(
            dimensions=2, initial_state=(5.0, 5.0), noise_std=0.0
        )
        wrapped = NoisyObservationWrapper(env, noise_std=0.5)

        # Collect multiple observations; at least some should differ from (5, 5)
        observations = [wrapped.observe().values for _ in range(20)]
        values_dim0 = [obs[0] for obs in observations]
        # Not all should be exactly 5.0 due to noise
        assert not all(v == 5.0 for v in values_dim0)

    def test_zero_noise_passes_through(self) -> None:
        env = NumericEnvironment(
            dimensions=2, initial_state=(3.0, 4.0), noise_std=0.0
        )
        wrapped = NoisyObservationWrapper(env, noise_std=0.0)
        snap = wrapped.observe()
        assert snap.values == pytest.approx((3.0, 4.0))

    def test_step_not_affected(self) -> None:
        """Noise is only on observe, not on step."""
        env = NumericEnvironment(dimensions=1, noise_std=0.0)
        wrapped = NoisyObservationWrapper(env, noise_std=1.0)
        snap = wrapped.step(ActionSpec(name="move", parameters={"delta": (1.0,)}))
        assert snap.values[0] == pytest.approx(1.0)


class TestHistoryTrackingWrapper:
    def test_records_history(self) -> None:
        env = NumericEnvironment(dimensions=2, noise_std=0.0)
        wrapped = HistoryTrackingWrapper(env, max_history=100)

        a1 = ActionSpec(name="a1", parameters={"delta": (1.0, 0.0)})
        a2 = ActionSpec(name="a2", parameters={"delta": (0.0, 1.0)})
        wrapped.step(a1)
        wrapped.step(a2)

        assert len(wrapped.history) == 2
        assert wrapped.history[0][0].name == "a1"
        assert wrapped.history[1][0].name == "a2"

    def test_bounded_history(self) -> None:
        env = NumericEnvironment(dimensions=1, noise_std=0.0)
        wrapped = HistoryTrackingWrapper(env, max_history=3)

        for i in range(5):
            wrapped.step(ActionSpec(name=f"a{i}", parameters={"delta": (0.1,)}))

        assert len(wrapped.history) == 3
        # Should have the last 3
        assert wrapped.history[0][0].name == "a2"

    def test_reset_clears_history(self) -> None:
        env = NumericEnvironment(dimensions=1, noise_std=0.0)
        wrapped = HistoryTrackingWrapper(env)
        wrapped.step(ActionSpec(name="a", parameters={"delta": (1.0,)}))
        assert len(wrapped.history) == 1

        wrapped.reset()
        assert len(wrapped.history) == 0


class TestResourceQuotaWrapper:
    def test_clamps_allocation(self) -> None:
        env = ResourceEnvironment(num_resources=2, total_resources=100.0)
        wrapped = ResourceQuotaWrapper(env, quotas={0: 10.0, 1: 5.0})

        # Try to allocate 50 from resource 0 — should clamp to 10
        action = ActionSpec(name="alloc", parameters={"allocations": {0: 50.0, 1: 3.0}})
        wrapped.step(action)

        # Resource 0 should have consumed at most 10 + regen
        consumed = env.total_consumed
        assert consumed[0] <= 10.5  # slight tolerance for regen

    def test_strict_mode_raises(self) -> None:
        env = ResourceEnvironment(num_resources=1, total_resources=100.0)
        wrapped = ResourceQuotaWrapper(env, quotas={0: 5.0}, strict=True)

        action = ActionSpec(name="alloc", parameters={"allocations": {0: 20.0}})
        with pytest.raises(ValueError, match="exceeds quota"):
            wrapped.step(action)

    def test_no_quotas_passes_through(self) -> None:
        env = ResourceEnvironment(num_resources=1, total_resources=100.0)
        wrapped = ResourceQuotaWrapper(env)
        action = ActionSpec(name="alloc", parameters={"allocations": {0: 50.0}})
        wrapped.step(action)  # Should not raise


class TestComposition:
    def test_stacking_wrappers(self) -> None:
        env = NumericEnvironment(dimensions=2, noise_std=0.0)
        noisy = NoisyObservationWrapper(env, noise_std=0.1)
        tracked = HistoryTrackingWrapper(noisy, max_history=10)

        action = ActionSpec(name="move", parameters={"delta": (1.0, 0.0)})
        tracked.step(action)

        assert len(tracked.history) == 1
        assert tracked.unwrapped is env

    def test_state_dict_through_wrapper(self) -> None:
        env = NumericEnvironment(dimensions=2, initial_state=(1.0, 2.0), noise_std=0.0)
        wrapped = HistoryTrackingWrapper(env)

        sd = wrapped.state_dict()
        assert sd["state"] == pytest.approx([1.0, 2.0])
