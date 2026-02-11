"""Distribution shift benchmark for the Synthetic Teleology framework.

Tests agent adaptivity when environment dynamics change mid-run.  The agent
operates in a :class:`NumericEnvironment`, pursues a target objective for
*N* steps, then a perturbation is injected (state shift, noise increase,
or dynamics change), and the agent must adapt for *N* additional steps.

Collected metrics: Goal Persistence (GP), Teleological Coherence (TC),
Adaptivity (AD), Lyapunov Stability (LS).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from synthetic_teleology.agents.factory import AgentFactory
from synthetic_teleology.agents.teleological import TeleologicalAgent
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.infrastructure.event_bus import EventBus
from synthetic_teleology.measurement.benchmarks.base import BaseBenchmark
from synthetic_teleology.measurement.collector import AgentLog, AgentLogEntry, EventCollector
from synthetic_teleology.measurement.engine import MetricsEngine
from synthetic_teleology.measurement.metrics.adaptivity import Adaptivity
from synthetic_teleology.measurement.metrics.goal_persistence import GoalPersistence
from synthetic_teleology.measurement.metrics.lyapunov_stability import LyapunovStability
from synthetic_teleology.measurement.metrics.teleological_coherence import TeleologicalCoherence
from synthetic_teleology.measurement.report import MetricsReport

logger = logging.getLogger(__name__)


class DistributionShiftBenchmark(BaseBenchmark):
    """Benchmark testing agent adaptivity under distribution shift.

    The scenario proceeds in two phases:

    1. **Pre-shift** (*steps_per_phase* steps): the agent pursues a target
       objective in a stable environment.
    2. **Post-shift** (*steps_per_phase* steps): a perturbation is injected
       at the midpoint, shifting the environment state, and the agent must
       adapt.

    Parameters
    ----------
    dimensions:
        Number of environment dimensions.
    steps_per_phase:
        Number of steps in each phase (pre- and post-shift).
    target_values:
        Target objective values.  If ``None``, defaults to all 5.0.
    perturbation_magnitude:
        Magnitude of the state shift perturbation.
    noise_std:
        Baseline noise standard deviation for the environment.
    post_shift_noise_std:
        Noise standard deviation after the distribution shift.
        If ``None``, defaults to ``noise_std * 3``.
    step_size:
        Agent action step size.
    """

    def __init__(
        self,
        dimensions: int = 3,
        steps_per_phase: int = 25,
        target_values: tuple[float, ...] | None = None,
        perturbation_magnitude: float = 5.0,
        noise_std: float = 0.05,
        post_shift_noise_std: float | None = None,
        step_size: float = 0.5,
        shift_mode: str = "sudden",
        transition_steps: int = 5,
        shift_type: str = "state",
    ) -> None:
        if shift_mode not in ("sudden", "gradual"):
            raise ValueError(f"shift_mode must be 'sudden' or 'gradual', got {shift_mode!r}")
        if shift_type not in ("state", "dynamics", "both"):
            raise ValueError(f"shift_type must be 'state', 'dynamics', or 'both', got {shift_type!r}")

        self._dimensions = dimensions
        self._steps_per_phase = steps_per_phase
        self._target_values = target_values or tuple(5.0 for _ in range(dimensions))
        self._perturbation_magnitude = perturbation_magnitude
        self._noise_std = noise_std
        self._post_shift_noise_std = (
            post_shift_noise_std
            if post_shift_noise_std is not None
            else noise_std * 3.0
        )
        self._step_size = step_size
        self._shift_mode = shift_mode
        self._transition_steps = transition_steps
        self._shift_type = shift_type

        # Initialized in setup()
        self._event_bus: EventBus | None = None
        self._collector: EventCollector | None = None
        self._env: NumericEnvironment | None = None
        self._agent: TeleologicalAgent | None = None
        self._engine: MetricsEngine | None = None

    def setup(self) -> None:
        """Create event bus, metrics engine with GP/TC/AD/LS metrics."""
        self._engine = MetricsEngine(
            metrics=[
                GoalPersistence(),
                TeleologicalCoherence(),
                Adaptivity(),
                LyapunovStability(),
            ]
        )

    def run_scenario(self, seed: int) -> AgentLog:
        """Run a distribution-shift scenario with the given seed.

        1. Create environment and agent with the seed.
        2. Run pre-shift phase (steps_per_phase steps).
        3. Inject perturbation at midpoint.
        4. Run post-shift phase (steps_per_phase steps).
        5. Return the agent log.
        """
        rng = np.random.default_rng(seed)
        np.random.seed(seed)

        # Fresh event bus and collector for each run
        event_bus = EventBus()
        collector = EventCollector(event_bus)

        # Create environment
        env = NumericEnvironment(
            dimensions=self._dimensions,
            initial_state=tuple(0.0 for _ in range(self._dimensions)),
            noise_std=self._noise_std,
            event_bus=event_bus,
            env_id=f"dist-shift-env-{seed}",
        )

        # Create agent with target objective
        agent = AgentFactory.create_simple_agent(
            agent_id=f"dist-shift-agent-{seed}",
            target_values=self._target_values,
            event_bus=event_bus,
            directions=tuple(Direction.APPROACH for _ in self._target_values),
            threshold=0.5,
            step_size=self._step_size,
        )

        env.reset()

        # Phase 1: pre-shift
        for step in range(self._steps_per_phase):
            action = agent.run_cycle(env)
            env.step(action)

        # Inject perturbation at midpoint
        shift_vector = tuple(
            float(rng.uniform(-1.0, 1.0) * self._perturbation_magnitude)
            for _ in range(self._dimensions)
        )
        env.inject_perturbation({
            "type": "distribution_shift",
            "shift": shift_vector,
            "noise_std": self._post_shift_noise_std,
            "magnitude": self._perturbation_magnitude,
            "affected_dimensions": tuple(range(self._dimensions)),
        })

        # Phase 2: post-shift
        for step in range(self._steps_per_phase):
            action = agent.run_cycle(env)
            env.step(action)

        # Retrieve or build the agent log
        agent_id = agent.id
        log = collector.get_log(agent_id)

        if log is None:
            # Build a synthetic log if the collector didn't capture events
            log = self._build_synthetic_log(agent_id, env, agent)

        return log

    def collect_metrics(self, log: AgentLog) -> MetricsReport:
        """Compute GP, TC, AD, LS metrics from the agent log."""
        assert self._engine is not None, "setup() must be called before collect_metrics()"
        return self._engine.build_report(log.agent_id, log)

    def teardown(self) -> None:
        """Release benchmark resources."""
        self._event_bus = None
        self._collector = None
        self._env = None
        self._agent = None

    def _build_synthetic_log(
        self,
        agent_id: str,
        env: NumericEnvironment,
        agent: TeleologicalAgent,
    ) -> AgentLog:
        """Build a synthetic AgentLog when the event collector did not
        capture data (fallback for environments without full event wiring).
        """
        log = AgentLog(agent_id=agent_id)
        total_steps = self._steps_per_phase * 2

        for step in range(total_steps):
            entry = AgentLogEntry(
                step=step,
                timestamp=time.time(),
                goal_id=agent.current_goal.goal_id,
                eval_score=agent.last_eval.score if agent.last_eval else 0.0,
                eval_confidence=agent.last_eval.confidence if agent.last_eval else 1.0,
                action_name=agent.last_policy.actions[0].name if (
                    agent.last_policy and agent.last_policy.actions
                ) else "noop",
                action_cost=0.0,
                state_values=tuple(float(v) for v in env.state_array),
                goal_revised=False,
                constraint_violated=False,
                reflection_triggered=step == self._steps_per_phase,
            )
            log.entries.append(entry)

        return log

    def __repr__(self) -> str:
        return (
            f"DistributionShiftBenchmark("
            f"dims={self._dimensions}, "
            f"steps_per_phase={self._steps_per_phase}, "
            f"perturbation={self._perturbation_magnitude})"
        )
