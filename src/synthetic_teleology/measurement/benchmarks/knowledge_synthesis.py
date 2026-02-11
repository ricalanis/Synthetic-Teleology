"""Knowledge synthesis benchmark for the Synthetic Teleology framework.

Tests agent performance in a :class:`ResearchEnvironment` where the agent
must accumulate knowledge across topics and synthesize it.  The benchmark
evaluates the agent's ability to balance breadth (exploring multiple topics)
with depth (specializing in key areas) while maintaining novelty.

Collected metrics: Innovation Yield (IY), Reflective Efficiency (RE),
Teleological Coherence (TC).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from synthetic_teleology.agents.teleological import TeleologicalAgent
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import (
    ActionSpec,
    ObjectiveVector,
    PolicySpec,
    StateSnapshot,
)
from synthetic_teleology.environments.research import ResearchEnvironment
from synthetic_teleology.infrastructure.event_bus import EventBus
from synthetic_teleology.measurement.benchmarks.base import BaseBenchmark
from synthetic_teleology.measurement.collector import AgentLog, AgentLogEntry, EventCollector
from synthetic_teleology.measurement.engine import MetricsEngine
from synthetic_teleology.measurement.metrics.innovation_yield import InnovationYield
from synthetic_teleology.measurement.metrics.reflective_efficiency import ReflectiveEfficiency
from synthetic_teleology.measurement.metrics.teleological_coherence import TeleologicalCoherence
from synthetic_teleology.measurement.report import MetricsReport
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.goal_revision import ThresholdUpdater
from synthetic_teleology.services.planning import BasePlanner

logger = logging.getLogger(__name__)


class ResearchPlanner(BasePlanner):
    """Simple planner for research environments.

    Selects among "research", "synthesize", and "explore" actions based
    on the agent's current state relative to its goal.  Heuristically
    balances knowledge acquisition, synthesis, and exploration.

    Parameters
    ----------
    num_topics:
        Number of knowledge topics in the environment.
    exploration_rate:
        Probability of choosing "explore" at each step.
    synthesis_threshold:
        Minimum average knowledge level before synthesis is attempted.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_topics: int = 3,
        exploration_rate: float = 0.15,
        synthesis_threshold: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self._num_topics = num_topics
        self._exploration_rate = exploration_rate
        self._synthesis_threshold = synthesis_threshold
        self._rng = np.random.default_rng(seed)
        self._action_counter: dict[str, int] = {
            "research": 0,
            "synthesize": 0,
            "explore": 0,
        }

    def plan(self, goal: Goal, state: StateSnapshot) -> PolicySpec:
        """Generate a research policy based on state and goal.

        Strategy:
        1. With probability ``exploration_rate``, explore for novelty.
        2. If average knowledge is below ``synthesis_threshold``, research
           the least-known topic.
        3. Otherwise, synthesize.
        """
        if state.dimension < self._num_topics + 2:
            # State too small; fall back to research topic 0
            action = ActionSpec(
                name="research",
                parameters={"topic": 0, "effort": 0.5},
            )
            return PolicySpec(actions=(action,))

        # Parse state: [knowledge_per_topic..., synthesis_quality, novelty]
        knowledge = state.values[:self._num_topics]
        synthesis_quality = state.values[self._num_topics] if len(state.values) > self._num_topics else 0.0
        novelty = state.values[self._num_topics + 1] if len(state.values) > self._num_topics + 1 else 0.0

        avg_knowledge = sum(knowledge) / max(1, len(knowledge))

        # Decision logic
        roll = float(self._rng.random())

        if roll < self._exploration_rate:
            # Explore
            action = ActionSpec(
                name="explore",
                parameters={"effort": float(self._rng.uniform(0.3, 0.8))},
            )
            self._action_counter["explore"] += 1
        elif avg_knowledge < self._synthesis_threshold:
            # Research: pick the least-known topic
            min_topic = int(np.argmin(knowledge))
            action = ActionSpec(
                name="research",
                parameters={
                    "topic": min_topic,
                    "effort": float(self._rng.uniform(0.4, 0.9)),
                },
            )
            self._action_counter["research"] += 1
        else:
            # Synthesize
            action = ActionSpec(
                name="synthesize",
                parameters={"effort": float(self._rng.uniform(0.5, 1.0))},
            )
            self._action_counter["synthesize"] += 1

        return PolicySpec(
            actions=(action,),
            metadata={
                "planner": "ResearchPlanner",
                "avg_knowledge": avg_knowledge,
                "synthesis_quality": synthesis_quality,
                "novelty": novelty,
                "action_counts": dict(self._action_counter),
            },
        )


class KnowledgeSynthesisBenchmark(BaseBenchmark):
    """Benchmark testing agent performance in knowledge synthesis.

    The agent operates in a :class:`ResearchEnvironment` and must balance
    research (knowledge acquisition), synthesis (combining knowledge), and
    exploration (maintaining novelty).

    Parameters
    ----------
    topics:
        Tuple of topic names for the research environment.
    num_steps:
        Total number of loop iterations per scenario.
    target_knowledge:
        Target knowledge level per topic in the objective.
    target_synthesis:
        Target synthesis quality in the objective.
    target_novelty:
        Target novelty level in the objective.
    exploration_rate:
        Probability of the planner choosing exploration.
    synthesis_threshold:
        Minimum average knowledge before synthesis is attempted.
    knowledge_decay:
        Per-step knowledge decay rate.
    novelty_decay:
        Per-step novelty decay rate.
    """

    def __init__(
        self,
        topics: tuple[str, ...] = ("machine_learning", "neuroscience", "philosophy"),
        num_steps: int = 60,
        target_knowledge: float = 5.0,
        target_synthesis: float = 3.0,
        target_novelty: float = 2.0,
        exploration_rate: float = 0.15,
        synthesis_threshold: float = 1.0,
        knowledge_decay: float = 0.01,
        novelty_decay: float = 0.05,
    ) -> None:
        self._topics = topics
        self._num_topics = len(topics)
        self._num_steps = num_steps
        self._target_knowledge = target_knowledge
        self._target_synthesis = target_synthesis
        self._target_novelty = target_novelty
        self._exploration_rate = exploration_rate
        self._synthesis_threshold = synthesis_threshold
        self._knowledge_decay = knowledge_decay
        self._novelty_decay = novelty_decay
        self._engine: MetricsEngine | None = None

    def _build_objective(self) -> ObjectiveVector:
        """Build the knowledge synthesis objective vector.

        State layout: [knowledge_per_topic..., synthesis_quality, novelty]
        """
        values = (
            tuple(self._target_knowledge for _ in range(self._num_topics))
            + (self._target_synthesis, self._target_novelty)
        )
        directions = (
            tuple(Direction.MAXIMIZE for _ in range(self._num_topics))
            + (Direction.MAXIMIZE, Direction.MAXIMIZE)
        )
        return ObjectiveVector(values=values, directions=directions)

    def setup(self) -> None:
        """Create metrics engine with IY, RE, TC metrics."""
        self._engine = MetricsEngine(
            metrics=[
                InnovationYield(),
                ReflectiveEfficiency(),
                TeleologicalCoherence(),
            ]
        )

    def run_scenario(self, seed: int) -> AgentLog:
        """Run a knowledge synthesis scenario.

        1. Create research environment and teleological agent.
        2. Run the agent for num_steps iterations.
        3. Return the agent log.
        """
        rng = np.random.default_rng(seed)
        np.random.seed(seed)

        event_bus = EventBus()
        collector = EventCollector(event_bus)

        # Create research environment
        env = ResearchEnvironment(
            topics=self._topics,
            knowledge_decay=self._knowledge_decay,
            novelty_decay=self._novelty_decay,
            noise_std=0.02,
            event_bus=event_bus,
            env_id=f"research-env-{seed}",
        )

        # Build objective and goal
        objective = self._build_objective()
        goal = Goal(
            name=f"knowledge-synthesis-{seed}",
            description=(
                f"Accumulate knowledge across {self._num_topics} topics, "
                f"synthesize findings, and maintain novelty"
            ),
            objective=objective,
        )

        # Create agent with research planner
        evaluator = NumericEvaluator(max_distance=15.0)
        updater = ThresholdUpdater(threshold=0.6, learning_rate=0.08)
        planner = ResearchPlanner(
            num_topics=self._num_topics,
            exploration_rate=self._exploration_rate,
            synthesis_threshold=self._synthesis_threshold,
            seed=seed,
        )

        agent = TeleologicalAgent(
            agent_id=f"knowledge-agent-{seed}",
            initial_goal=goal,
            event_bus=event_bus,
            evaluator=evaluator,
            updater=updater,
            planner=planner,
        )

        env.reset()

        # Run the scenario
        agent_log = AgentLog(agent_id=agent.id)

        for step in range(self._num_steps):
            # Perceive (manually, since ResearchEnvironment returns
            # a different state shape than agent might expect)
            state = env.observe()

            # Run the agent's full cycle, but drive env.step manually
            try:
                action = agent.run_cycle(env)
            except Exception:
                # If the cycle fails (e.g., dimension mismatch), use planner directly
                policy = planner.plan(goal, state)
                action = policy.actions[0] if policy.actions else ActionSpec(
                    name="research", parameters={"topic": 0, "effort": 0.5}
                )

            new_state = env.step(action)

            entry = AgentLogEntry(
                step=step,
                timestamp=time.time(),
                goal_id=agent.current_goal.goal_id,
                eval_score=agent.last_eval.score if agent.last_eval else 0.0,
                eval_confidence=agent.last_eval.confidence if agent.last_eval else 1.0,
                action_name=action.name,
                action_cost=action.cost,
                state_values=new_state.values,
                goal_revised=False,
                constraint_violated=False,
                reflection_triggered=False,
                metadata={
                    "knowledge_levels": new_state.values[:self._num_topics],
                    "synthesis_quality": new_state.values[self._num_topics]
                    if len(new_state.values) > self._num_topics
                    else 0.0,
                    "novelty": new_state.values[self._num_topics + 1]
                    if len(new_state.values) > self._num_topics + 1
                    else 0.0,
                },
            )
            agent_log.entries.append(entry)

        # Try to get log from collector; fall back to our manually built log
        collected_log = collector.get_log(agent.id)
        if collected_log is not None and collected_log.num_steps > 0:
            return collected_log

        return agent_log

    def collect_metrics(self, log: AgentLog) -> MetricsReport:
        """Compute IY, RE, TC metrics from the agent log."""
        assert self._engine is not None, "setup() must be called before collect_metrics()"
        return self._engine.build_report(log.agent_id, log)

    def teardown(self) -> None:
        """Release benchmark resources."""
        pass

    def __repr__(self) -> str:
        return (
            f"KnowledgeSynthesisBenchmark("
            f"topics={self._num_topics}, "
            f"steps={self._num_steps}, "
            f"target_knowledge={self._target_knowledge})"
        )
