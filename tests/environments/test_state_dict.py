"""Tests for state_dict() / load_state_dict() on all environments."""

from __future__ import annotations

import numpy as np
import pytest

from synthetic_teleology.domain.values import ActionSpec
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.environments.research import ResearchEnvironment
from synthetic_teleology.environments.resource import ResourceEnvironment
from synthetic_teleology.environments.shared import SharedEnvironment


class TestNumericEnvironmentStateDict:
    def test_round_trip(self) -> None:
        env = NumericEnvironment(dimensions=3, initial_state=(1.0, 2.0, 3.0), noise_std=0.0)
        env.step(ActionSpec(name="move", parameters={"delta": (0.5, -0.5, 0.1)}))

        sd = env.state_dict()
        assert sd["step_count"] == 1
        assert sd["dimensions"] == 3
        assert len(sd["state"]) == 3

        # Create a fresh env and restore
        env2 = NumericEnvironment(dimensions=3, noise_std=0.0)
        env2.load_state_dict(sd)

        assert env2.step_count == 1
        np.testing.assert_allclose(env2.state_array, env.state_array)

    def test_modify_then_restore(self) -> None:
        env = NumericEnvironment(dimensions=2, noise_std=0.0)
        env.reset()
        sd = env.state_dict()

        # Modify state
        env.step(ActionSpec(name="move", parameters={"delta": (10.0, 10.0)}))
        assert env.state_array[0] == pytest.approx(10.0)

        # Restore
        env.load_state_dict(sd)
        assert env.state_array[0] == pytest.approx(0.0)
        assert env.step_count == 0


class TestResourceEnvironmentStateDict:
    def test_round_trip(self) -> None:
        env = ResourceEnvironment(num_resources=2, total_resources=100.0)
        env.step(ActionSpec(name="consume", parameters={"allocations": {0: 30.0}}))

        sd = env.state_dict()
        assert sd["step_count"] == 1
        assert len(sd["levels"]) == 2
        assert sd["total_consumed"][0] > 0.0

        env2 = ResourceEnvironment(num_resources=2, total_resources=100.0)
        env2.load_state_dict(sd)

        np.testing.assert_allclose(env2.levels, env.levels)
        np.testing.assert_allclose(env2.total_consumed, env.total_consumed)

    def test_modify_then_restore(self) -> None:
        env = ResourceEnvironment(num_resources=2, total_resources=50.0)
        sd_before = env.state_dict()

        env.step(ActionSpec(
            name="drain",
            parameters={"consume_all": True, "consume_fraction": 0.5},
        ))
        assert env.levels[0] < 50.0

        env.load_state_dict(sd_before)
        np.testing.assert_allclose(env.levels, [50.0, 50.0], atol=0.1)


class TestResearchEnvironmentStateDict:
    def test_round_trip(self) -> None:
        env = ResearchEnvironment(topics=("ml", "bio"), noise_std=0.0)
        env.step(ActionSpec(name="research", parameters={"topic": 0, "effort": 1.0}))

        sd = env.state_dict()
        assert sd["step_count"] == 1
        assert len(sd["knowledge"]) == 2
        assert sd["knowledge"][0] > 0.0

        env2 = ResearchEnvironment(topics=("ml", "bio"), noise_std=0.0)
        env2.load_state_dict(sd)

        np.testing.assert_allclose(env2.knowledge_levels, env.knowledge_levels)
        assert env2.synthesis_quality == pytest.approx(env.synthesis_quality)
        assert env2.novelty == pytest.approx(env.novelty)

    def test_modify_then_restore(self) -> None:
        env = ResearchEnvironment(topics=("a", "b", "c"), noise_std=0.0)
        sd = env.state_dict()

        env.step(ActionSpec(name="explore", parameters={"effort": 2.0}))
        assert env.novelty > 0.0

        env.load_state_dict(sd)
        assert env.novelty == pytest.approx(0.0)


class TestSharedEnvironmentStateDict:
    def test_round_trip(self) -> None:
        env = SharedEnvironment(global_dimensions=3, local_dimensions=1)
        env.register_agent("a1", initial_local_state=(1.0,))
        env.register_agent("a2", initial_local_state=(2.0,))

        env.step(
            ActionSpec(name="move", parameters={"delta_global": (0.5, 0.0, 0.0)}),
            agent_id="a1",
        )

        sd = env.state_dict()
        assert sd["step_count"] == 1
        assert "a1" in sd["local_states"]
        assert "a2" in sd["local_states"]

        env2 = SharedEnvironment(global_dimensions=3, local_dimensions=1)
        env2.register_agent("a1")
        env2.register_agent("a2")
        env2.load_state_dict(sd)

        np.testing.assert_allclose(env2.global_state, env.global_state)
        np.testing.assert_allclose(
            env2.get_local_state("a1"),
            env.get_local_state("a1"),
        )


class TestBaseEnvironmentStateDictDefaults:
    """Verify base class defaults work even without subclass overrides."""

    def test_base_fields_in_all_envs(self) -> None:
        for env_cls, kwargs in [
            (NumericEnvironment, {"dimensions": 2}),
            (ResourceEnvironment, {"num_resources": 2}),
            (ResearchEnvironment, {}),
        ]:
            env = env_cls(**kwargs)
            sd = env.state_dict()
            assert "env_id" in sd
            assert "step_count" in sd
            assert "perturbation_history" in sd
