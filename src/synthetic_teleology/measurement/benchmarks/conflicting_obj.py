"""Conflicting objectives benchmark for the Synthetic Teleology framework.

Tests agent behaviour when the goal has multi-dimensional objectives that
pull in opposite directions.  For example, one dimension demands maximization
while another demands minimization, and both share overlapping state space
effects -- meaning progress on one objective regresses the other.

Collected metrics: Teleological Coherence (TC), Reflective Efficiency (RE),
Normative Fidelity (NF).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from synthetic_teleology.agents.factory import AgentFactory
from synthetic_teleology.agents.teleological import TeleologicalAgent
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import ObjectiveVector
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.infrastructure.event_bus import EventBus
from synthetic_teleology.measurement.benchmarks.base import BaseBenchmark
from synthetic_teleology.measurement.collector import AgentLog, AgentLogEntry, EventCollector
from synthetic_teleology.measurement.engine import MetricsEngine
from synthetic_teleology.measurement.metrics.innovation_yield import InnovationYield
from synthetic_teleology.measurement.metrics.normative_fidelity import NormativeFidelity
from synthetic_teleology.measurement.metrics.reflective_efficiency import ReflectiveEfficiency
from synthetic_teleology.measurement.metrics.teleological_coherence import TeleologicalCoherence
from synthetic_teleology.measurement.report import MetricsReport

logger = logging.getLogger(__name__)


class ConflictingObjectivesBenchmark(BaseBenchmark):
    """Benchmark testing agents with multi-dimensional conflicting objectives.

    The goal has objectives pulling in opposite directions on different
    dimensions.  For instance, dimension 0 targets a high value while
    dimension 1 targets a low value, but the available actions affect both
    dimensions simultaneously (coupled dynamics).

    Parameters
    ----------
    dimensions:
        Number of state/objective dimensions.  Must be >= 2.
    num_steps:
        Total number of loop iterations per scenario.
    conflict_strength:
        Controls how strongly dimensions are coupled.  Higher values
        make the conflict more severe: an action that helps one dimension
        hurts the opposing dimension proportionally.  Range [0, 1].
    noise_std:
        Environment noise standard deviation.
    step_size:
        Agent action step size.
    """

    def __init__(
        self,
        dimensions: int = 4,
        num_steps: int = 50,
        conflict_strength: float = 0.6,
        noise_std: float = 0.02,
        step_size: float = 0.5,
        tradeoff_step: int | None = None,
        tradeoff_multiplier: float = 1.5,
    ) -> None:
        if dimensions < 2:
            raise ValueError(f"dimensions must be >= 2, got {dimensions}")
        if not 0.0 <= conflict_strength <= 1.0:
            raise ValueError(
                f"conflict_strength must be in [0, 1], got {conflict_strength}"
            )

        self._dimensions = dimensions
        self._num_steps = num_steps
        self._conflict_strength = conflict_strength
        self._noise_std = noise_std
        self._step_size = step_size
        self._tradeoff_step = tradeoff_step
        self._tradeoff_multiplier = tradeoff_multiplier
        self._engine: MetricsEngine | None = None

    def _build_conflicting_objective(
        self,
        rng: np.random.Generator,
    ) -> ObjectiveVector:
        """Create an objective with conflicting directions.

        Even-indexed dimensions use MAXIMIZE with high target values;
        odd-indexed dimensions use MINIMIZE with low target values.
        This creates inherent tension when actions are coupled.
        """
        values: list[float] = []
        directions: list[Direction] = []

        for d in range(self._dimensions):
            if d % 2 == 0:
                # Maximize: target high value
                values.append(float(rng.uniform(5.0, 10.0)))
                directions.append(Direction.MAXIMIZE)
            else:
                # Minimize: target low value
                values.append(float(rng.uniform(-10.0, -5.0)))
                directions.append(Direction.MINIMIZE)

        return ObjectiveVector(
            values=tuple(values),
            directions=tuple(directions),
        )

    def _build_coupled_dynamics(
        self,
        rng: np.random.Generator,
    ):
        """Build a dynamics function where dimensions are coupled.

        When one dimension moves positively, coupled dimensions move
        negatively (and vice versa), scaled by conflict_strength.
        """
        # Build a coupling matrix: positive on diagonal, negative off-diagonal
        # for opposing dimension pairs
        coupling = np.eye(self._dimensions, dtype=np.float64)

        for d in range(self._dimensions):
            for d2 in range(self._dimensions):
                if d == d2:
                    continue
                # Opposing dimensions get negative coupling
                if (d % 2) != (d2 % 2):
                    coupling[d, d2] = -self._conflict_strength * float(
                        rng.uniform(0.3, 0.7)
                    )

        def dynamics_fn(state, delta):
            """Apply coupled dynamics: delta is transformed through coupling."""
            effective_delta = coupling @ delta
            return state + effective_delta

        return dynamics_fn

    def setup(self) -> None:
        """Create metrics engine with TC, RE, NF, IY metrics."""
        self._engine = MetricsEngine(
            metrics=[
                TeleologicalCoherence(),
                ReflectiveEfficiency(),
                NormativeFidelity(),
                InnovationYield(),
            ]
        )

    def run_scenario(self, seed: int) -> AgentLog:
        """Run a conflicting-objectives scenario.

        1. Create environment with coupled dynamics.
        2. Create agent with conflicting objective.
        3. Run for num_steps iterations.
        4. Return the agent log.
        """
        rng = np.random.default_rng(seed)
        np.random.seed(seed)

        event_bus = EventBus()
        collector = EventCollector(event_bus)

        # Build coupled dynamics
        dynamics_fn = self._build_coupled_dynamics(rng)

        # Create environment with coupled dynamics
        env = NumericEnvironment(
            dimensions=self._dimensions,
            initial_state=tuple(0.0 for _ in range(self._dimensions)),
            noise_std=self._noise_std,
            dynamics_fn=dynamics_fn,
            event_bus=event_bus,
            env_id=f"conflict-env-{seed}",
        )

        # Build conflicting objective
        objective = self._build_conflicting_objective(rng)

        # Create agent using factory with custom directions
        agent = AgentFactory.create_simple_agent(
            agent_id=f"conflict-agent-{seed}",
            target_values=objective.values,
            event_bus=event_bus,
            directions=objective.directions,
            threshold=0.3,
            step_size=self._step_size,
        )

        env.reset()

        # Run the scenario
        for step in range(self._num_steps):
            action = agent.run_cycle(env)
            env.step(action)

        # Retrieve or build the agent log
        agent_id = agent.id
        log = collector.get_log(agent_id)

        if log is None:
            log = self._build_synthetic_log(agent_id, env, agent)

        return log

    def collect_metrics(self, log: AgentLog) -> MetricsReport:
        """Compute TC, RE, NF metrics from the agent log."""
        assert self._engine is not None, "setup() must be called before collect_metrics()"
        return self._engine.build_report(log.agent_id, log)

    def teardown(self) -> None:
        """Release benchmark resources."""
        pass

    def _build_synthetic_log(
        self,
        agent_id: str,
        env: NumericEnvironment,
        agent: TeleologicalAgent,
    ) -> AgentLog:
        """Build a synthetic AgentLog as fallback."""
        log = AgentLog(agent_id=agent_id)

        for step in range(self._num_steps):
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
                reflection_triggered=False,
            )
            log.entries.append(entry)

        return log

    def __repr__(self) -> str:
        return (
            f"ConflictingObjectivesBenchmark("
            f"dims={self._dimensions}, "
            f"steps={self._num_steps}, "
            f"conflict_strength={self._conflict_strength})"
        )
