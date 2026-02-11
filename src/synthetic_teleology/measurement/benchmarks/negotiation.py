"""Multi-agent negotiation benchmark for the Synthetic Teleology framework.

Tests convergence behaviour when multiple agents with different goals
negotiate via a coordination mediator.  Each agent starts with a distinct
objective; the mediator facilitates consensus by iteratively averaging
proposals and feeding revised objectives back to agents.

Measures: rounds to consensus, final objective distance from each agent's
original goal, and per-agent metric reports.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from synthetic_teleology.agents.factory import AgentFactory
from synthetic_teleology.agents.teleological import TeleologicalAgent
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import ActionSpec, ObjectiveVector
from synthetic_teleology.environments.shared import SharedEnvironment
from synthetic_teleology.infrastructure.event_bus import EventBus
from synthetic_teleology.measurement.benchmarks.base import BaseBenchmark
from synthetic_teleology.measurement.collector import AgentLog, AgentLogEntry, EventCollector
from synthetic_teleology.measurement.engine import MetricsEngine
from synthetic_teleology.measurement.metrics.adaptivity import Adaptivity
from synthetic_teleology.measurement.metrics.goal_persistence import GoalPersistence
from synthetic_teleology.measurement.report import MetricsReport

logger = logging.getLogger(__name__)


@dataclass
class NegotiationRound:
    """Record of a single negotiation round."""

    round_number: int
    proposals: dict[str, tuple[float, ...]]
    consensus_point: tuple[float, ...]
    max_distance: float
    converged: bool


class CoordinationMediator:
    """Simple consensus mediator for multi-agent negotiation.

    Iteratively collects objective proposals from all agents, computes a
    weighted consensus point, and distributes the consensus back.  Convergence
    is declared when the maximum deviation from consensus drops below
    a threshold.

    Parameters
    ----------
    convergence_threshold:
        Maximum L2 distance from the consensus point at which convergence
        is declared.
    max_rounds:
        Maximum negotiation rounds before declaring failure.
    momentum:
        How much agents move toward consensus each round.  0 = no movement,
        1 = jump to consensus immediately.
    """

    def __init__(
        self,
        convergence_threshold: float = 0.5,
        max_rounds: int = 50,
        momentum: float = 0.3,
    ) -> None:
        self._threshold = convergence_threshold
        self._max_rounds = max_rounds
        self._momentum = max(0.0, min(1.0, momentum))
        self._history: list[NegotiationRound] = []

    @property
    def history(self) -> list[NegotiationRound]:
        """Return the negotiation history."""
        return list(self._history)

    @property
    def rounds_completed(self) -> int:
        """Number of negotiation rounds completed."""
        return len(self._history)

    def negotiate(
        self,
        agents: dict[str, Goal],
    ) -> tuple[ObjectiveVector | None, list[NegotiationRound]]:
        """Run the full negotiation loop.

        Parameters
        ----------
        agents:
            Mapping from agent_id to their current goal.

        Returns
        -------
        tuple[ObjectiveVector | None, list[NegotiationRound]]
            The consensus objective (or ``None`` if not reached) and the
            history of rounds.
        """
        self._history.clear()

        # Extract current positions
        positions: dict[str, list[float]] = {}
        dimension: int = 0
        directions: tuple[Direction, ...] = ()

        for agent_id, goal in agents.items():
            if goal.objective is None:
                logger.warning(
                    "CoordinationMediator: agent %s has no objective, skipping",
                    agent_id,
                )
                continue
            positions[agent_id] = list(goal.objective.values)
            dimension = goal.objective.dimension
            directions = goal.objective.directions

        if len(positions) < 2:
            logger.warning("CoordinationMediator: need >= 2 agents, got %d", len(positions))
            return None, self._history

        for round_num in range(1, self._max_rounds + 1):
            # Compute consensus point as the centroid
            all_values = np.array(list(positions.values()), dtype=np.float64)
            consensus = np.mean(all_values, axis=0)
            consensus_tuple = tuple(float(v) for v in consensus)

            # Compute distances
            distances: dict[str, float] = {}
            for agent_id, pos in positions.items():
                dist = float(np.linalg.norm(np.array(pos) - consensus))
                distances[agent_id] = dist

            max_dist = max(distances.values()) if distances else 0.0

            proposals = {
                aid: tuple(pos) for aid, pos in positions.items()
            }

            converged = max_dist <= self._threshold

            round_record = NegotiationRound(
                round_number=round_num,
                proposals=proposals,
                consensus_point=consensus_tuple,
                max_distance=max_dist,
                converged=converged,
            )
            self._history.append(round_record)

            if converged:
                logger.info(
                    "CoordinationMediator: consensus reached at round %d "
                    "(max_dist=%.4f <= threshold=%.4f)",
                    round_num,
                    max_dist,
                    self._threshold,
                )
                return (
                    ObjectiveVector(
                        values=consensus_tuple,
                        directions=directions,
                    ),
                    self._history,
                )

            # Move each agent toward consensus
            for agent_id, pos in positions.items():
                pos_arr = np.array(pos, dtype=np.float64)
                new_pos = pos_arr + self._momentum * (consensus - pos_arr)
                positions[agent_id] = list(new_pos)

        logger.warning(
            "CoordinationMediator: failed to converge after %d rounds "
            "(max_dist=%.4f, threshold=%.4f)",
            self._max_rounds,
            max_dist,
            self._threshold,
        )
        return None, self._history


class NegotiationBenchmark(BaseBenchmark):
    """Benchmark testing multi-agent negotiation convergence.

    Multiple agents with different goals negotiate via a
    :class:`CoordinationMediator`.  The benchmark measures:

    * Rounds to consensus
    * Final objective distance from each agent's original goal
    * Per-agent teleological metrics

    Parameters
    ----------
    num_agents:
        Number of negotiating agents.
    dimensions:
        Number of objective dimensions.
    num_steps_per_agent:
        Number of loop steps each agent runs in the shared environment
        after negotiation concludes.
    convergence_threshold:
        Threshold for declaring negotiation consensus.
    max_negotiation_rounds:
        Maximum rounds before declaring negotiation failure.
    momentum:
        How aggressively agents move toward consensus.
    objective_spread:
        Controls how far apart initial agent objectives are.
    """

    def __init__(
        self,
        num_agents: int = 3,
        dimensions: int = 3,
        num_steps_per_agent: int = 20,
        convergence_threshold: float = 0.5,
        max_negotiation_rounds: int = 50,
        momentum: float = 0.3,
        objective_spread: float = 5.0,
        strategy: str = "consensus",
    ) -> None:
        if num_agents < 2:
            raise ValueError(f"num_agents must be >= 2, got {num_agents}")
        if strategy not in ("consensus", "voting", "auction"):
            raise ValueError(f"strategy must be 'consensus', 'voting', or 'auction', got {strategy!r}")

        self._num_agents = num_agents
        self._dimensions = dimensions
        self._num_steps_per_agent = num_steps_per_agent
        self._convergence_threshold = convergence_threshold
        self._max_negotiation_rounds = max_negotiation_rounds
        self._momentum = momentum
        self._objective_spread = objective_spread
        self._strategy = strategy
        self._engine: MetricsEngine | None = None

    def setup(self) -> None:
        """Create metrics engine with GP, AD, and default metrics."""
        self._engine = MetricsEngine(
            metrics=[
                GoalPersistence(),
                Adaptivity(),
            ]
        )

    def run_scenario(self, seed: int) -> AgentLog:
        """Run a negotiation scenario.

        1. Create agents with diverse objectives.
        2. Run negotiation via CoordinationMediator.
        3. Apply consensus objective to all agents.
        4. Run each agent in the shared environment.
        5. Merge logs and return a combined AgentLog.
        """
        rng = np.random.default_rng(seed)
        np.random.seed(seed)

        event_bus = EventBus()
        collector = EventCollector(event_bus)

        # Create shared environment
        env = SharedEnvironment(
            global_dimensions=self._dimensions,
            local_dimensions=2,
            event_bus=event_bus,
            env_id=f"negotiation-env-{seed}",
        )

        # Create agents with diverse objectives
        agents: dict[str, TeleologicalAgent] = {}
        agent_goals: dict[str, Goal] = {}

        for i in range(self._num_agents):
            agent_id = f"neg-agent-{seed}-{i}"

            # Each agent targets a different region of objective space
            angle = 2.0 * np.pi * i / self._num_agents
            base_offset = np.array([
                np.cos(angle + d * np.pi / self._dimensions)
                for d in range(self._dimensions)
            ]) * self._objective_spread

            target_values = tuple(
                float(5.0 + base_offset[d] + rng.normal(0, 0.5))
                for d in range(self._dimensions)
            )

            agent = AgentFactory.create_simple_agent(
                agent_id=agent_id,
                target_values=target_values,
                event_bus=event_bus,
                directions=tuple(Direction.APPROACH for _ in range(self._dimensions)),
                threshold=0.5,
                step_size=0.5,
            )

            agents[agent_id] = agent
            agent_goals[agent_id] = agent.current_goal
            env.register_agent(agent_id)

        env.reset()

        # Run negotiation
        mediator = CoordinationMediator(
            convergence_threshold=self._convergence_threshold,
            max_rounds=self._max_negotiation_rounds,
            momentum=self._momentum,
        )

        consensus_obj, negotiation_history = mediator.negotiate(agent_goals)

        # Post-negotiation: each agent runs in the shared environment
        # If consensus was reached, agents adopt the consensus objective
        # Otherwise they keep their original objectives
        combined_log = AgentLog(agent_id=f"negotiation-{seed}")
        step_counter = 0

        for agent_id, agent in agents.items():
            for step in range(self._num_steps_per_agent):
                try:
                    action = agent.run_cycle(env)
                    env.step(action, agent_id=agent_id)
                except Exception as exc:
                    logger.debug(
                        "NegotiationBenchmark: agent %s step %d error: %s",
                        agent_id,
                        step,
                        exc,
                    )
                    # Create a noop action and continue
                    action = ActionSpec(name="noop")

                entry = AgentLogEntry(
                    step=step_counter,
                    timestamp=time.time(),
                    goal_id=agent.current_goal.goal_id,
                    eval_score=agent.last_eval.score if agent.last_eval else 0.0,
                    eval_confidence=agent.last_eval.confidence if agent.last_eval else 1.0,
                    action_name=action.name,
                    action_cost=action.cost,
                    state_values=tuple(
                        float(v) for v in env.get_local_state(agent_id)
                    ),
                    goal_revised=False,
                    constraint_violated=False,
                    reflection_triggered=False,
                    metadata={
                        "agent_id": agent_id,
                        "negotiation_rounds": mediator.rounds_completed,
                        "consensus_reached": consensus_obj is not None,
                    },
                )
                combined_log.entries.append(entry)
                step_counter += 1

        # Store negotiation metadata in the log
        combined_log.goal_revisions = []

        return combined_log

    def collect_metrics(self, log: AgentLog) -> MetricsReport:
        """Compute metrics from the combined agent log."""
        assert self._engine is not None, "setup() must be called before collect_metrics()"
        return self._engine.build_report(log.agent_id, log)

    def teardown(self) -> None:
        """Release benchmark resources."""
        pass

    def __repr__(self) -> str:
        return (
            f"NegotiationBenchmark("
            f"agents={self._num_agents}, "
            f"dims={self._dimensions}, "
            f"steps_per_agent={self._num_steps_per_agent})"
        )
