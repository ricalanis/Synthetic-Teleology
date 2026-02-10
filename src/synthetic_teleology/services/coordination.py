"""Multi-agent coordination service for the Synthetic Teleology framework.

Implements the Mediator pattern for coordinating multiple teleological agents
that must negotiate a shared objective.  Three negotiation strategies are
provided:

* :class:`ConsensusNegotiator` -- iterative averaging until convergence.
* :class:`VotingNegotiator`    -- Borda-count social choice.
* :class:`AuctionNegotiator`   -- bid-based dimension allocation.

The :class:`CoordinationMediator` wires agents together, selects a strategy,
and publishes domain events (``NegotiationStarted``, ``ConsensusReached``)
through the event bus.

Classes
-------
BaseNegotiator
    Abstract base class for negotiation strategies.
ConsensusNegotiator
    Iterative averaging with configurable tolerance and max rounds.
VotingNegotiator
    Borda-count ranking over agent objectives.
AuctionNegotiator
    Bid-based allocation across objective dimensions.
CoordinationMediator
    Mediator managing the agent network and orchestrating negotiation.
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Any, Mapping

import numpy as np

from synthetic_teleology.agents.base import BaseAgent
from synthetic_teleology.domain.enums import NegotiationStrategy
from synthetic_teleology.domain.events import ConsensusReached, NegotiationStarted
from synthetic_teleology.domain.exceptions import NegotiationDeadlock
from synthetic_teleology.domain.values import ObjectiveVector, StateSnapshot
from synthetic_teleology.infrastructure.event_bus import EventBus

logger = logging.getLogger(__name__)


# ===================================================================== #
#  Base Negotiator (ABC)                                                  #
# ===================================================================== #


class BaseNegotiator(ABC):
    """Abstract base class for multi-agent negotiation strategies.

    A negotiator takes a list of agents that each hold a current goal with
    an :class:`ObjectiveVector`, plus a shared state snapshot, and returns
    a single :class:`ObjectiveVector` that represents the group agreement.

    Subclasses must implement :meth:`negotiate`.
    """

    @abstractmethod
    def negotiate(
        self,
        agents: list[BaseAgent],
        shared_state: StateSnapshot,
    ) -> ObjectiveVector:
        """Negotiate a shared objective among agents.

        Parameters
        ----------
        agents:
            The participating agents.  Each agent's ``current_goal.objective``
            is used as that agent's preferred objective.
        shared_state:
            The current shared environment state, available for
            context-sensitive negotiation.

        Returns
        -------
        ObjectiveVector
            The agreed-upon shared objective after negotiation.

        Raises
        ------
        NegotiationDeadlock
            If the negotiation fails to reach agreement.
        ValueError
            If any agent lacks a valid objective or objectives have
            mismatched dimensionality.
        """


# ===================================================================== #
#  Consensus Negotiator                                                   #
# ===================================================================== #


class ConsensusNegotiator(BaseNegotiator):
    """Iterative averaging negotiation until convergence.

    In each round, every agent's objective moves toward the group mean.
    Convergence is declared when the maximum pairwise distance among all
    agent objectives falls below ``tolerance``.

    Parameters
    ----------
    max_rounds:
        Maximum number of averaging rounds before declaring deadlock.
        Defaults to 100.
    tolerance:
        Convergence threshold: maximum pairwise distance for consensus.
        Defaults to 0.01.
    blending_rate:
        How far each agent's objective moves toward the group mean each
        round, in (0, 1].  1.0 jumps directly to the mean; smaller values
        allow smoother convergence.  Defaults to 0.5.
    """

    def __init__(
        self,
        max_rounds: int = 100,
        tolerance: float = 0.01,
        blending_rate: float = 0.5,
    ) -> None:
        if max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
        if not 0.0 < blending_rate <= 1.0:
            raise ValueError(f"blending_rate must be in (0, 1], got {blending_rate}")
        self._max_rounds = max_rounds
        self._tolerance = tolerance
        self._blending_rate = blending_rate

    @property
    def max_rounds(self) -> int:
        """Maximum number of negotiation rounds."""
        return self._max_rounds

    @property
    def tolerance(self) -> float:
        """Convergence threshold for pairwise distance."""
        return self._tolerance

    def negotiate(
        self,
        agents: list[BaseAgent],
        shared_state: StateSnapshot,
    ) -> ObjectiveVector:
        """Negotiate via iterative averaging.

        Each round, the group mean is computed and every agent's working
        objective is blended toward that mean.  The process converges when
        the maximum pairwise distance drops below ``tolerance``.

        Returns
        -------
        ObjectiveVector
            The converged group objective (the mean of all objectives
            at convergence).

        Raises
        ------
        NegotiationDeadlock
            If convergence is not reached within ``max_rounds``.
        ValueError
            If fewer than two agents, or objectives have mismatched dimensions.
        """
        objectives = self._extract_objectives(agents)
        if len(objectives) < 2:
            # Single agent: its objective is the consensus
            return objectives[0]

        dimension = objectives[0].dimension
        directions = objectives[0].directions
        weights = objectives[0].weights

        # Initialize working values as numpy arrays for efficient computation
        working: list[np.ndarray] = [
            np.array(obj.values, dtype=np.float64) for obj in objectives
        ]

        for round_num in range(1, self._max_rounds + 1):
            # Compute group mean
            group_mean = np.mean(working, axis=0)

            # Blend each agent's objective toward the mean
            for i in range(len(working)):
                working[i] = (
                    working[i] + self._blending_rate * (group_mean - working[i])
                )

            # Check convergence: max pairwise distance
            max_dist = self._max_pairwise_distance(working)

            logger.debug(
                "ConsensusNegotiator round %d/%d: max_pairwise_dist=%.6f, "
                "tolerance=%.6f",
                round_num,
                self._max_rounds,
                max_dist,
                self._tolerance,
            )

            if max_dist < self._tolerance:
                # Consensus reached: return the current mean
                final_mean = np.mean(working, axis=0)
                final_values = tuple(float(v) for v in final_mean)
                logger.info(
                    "ConsensusNegotiator: consensus reached after %d rounds "
                    "(max_dist=%.6f)",
                    round_num,
                    max_dist,
                )
                return ObjectiveVector(
                    values=final_values,
                    directions=directions,
                    weights=weights,
                )

        # Deadlock: did not converge within max_rounds
        agent_ids = tuple(a.id for a in agents)
        raise NegotiationDeadlock(
            message=(
                f"ConsensusNegotiator failed to converge after "
                f"{self._max_rounds} rounds (max_dist={max_dist:.6f}, "
                f"tolerance={self._tolerance})"
            ),
            participant_ids=agent_ids,
            rounds_completed=self._max_rounds,
        )

    @staticmethod
    def _max_pairwise_distance(vectors: list[np.ndarray]) -> float:
        """Compute the maximum Euclidean distance between any pair of vectors."""
        max_dist = 0.0
        for i, j in combinations(range(len(vectors)), 2):
            dist = float(np.linalg.norm(vectors[i] - vectors[j]))
            if dist > max_dist:
                max_dist = dist
        return max_dist

    @staticmethod
    def _extract_objectives(agents: list[BaseAgent]) -> list[ObjectiveVector]:
        """Extract and validate objectives from all agents.

        Raises
        ------
        ValueError
            If any agent lacks an objective or dimensions do not match.
        """
        if not agents:
            raise ValueError("Cannot negotiate with zero agents")

        objectives: list[ObjectiveVector] = []
        for agent in agents:
            obj = agent.current_goal.objective
            if obj is None:
                raise ValueError(
                    f"Agent {agent.id!r} has no objective on its current goal"
                )
            objectives.append(obj)

        # Validate dimension consistency
        ref_dim = objectives[0].dimension
        for i, obj in enumerate(objectives[1:], start=1):
            if obj.dimension != ref_dim:
                raise ValueError(
                    f"Dimension mismatch: agent 0 has dimension {ref_dim}, "
                    f"agent {i} has dimension {obj.dimension}"
                )

        return objectives


# ===================================================================== #
#  Voting Negotiator                                                      #
# ===================================================================== #


class VotingNegotiator(BaseNegotiator):
    """Social choice negotiation using Borda count scoring.

    Each agent ranks all agents' objectives (including its own) by
    proximity to its own preferred objective.  Rankings are converted
    to Borda scores (rank 1 = highest score = N-1 points, rank N =
    0 points, where N is the number of agents).  The objective with
    the highest total Borda score is selected as the group agreement.
    """

    def negotiate(
        self,
        agents: list[BaseAgent],
        shared_state: StateSnapshot,
    ) -> ObjectiveVector:
        """Negotiate via Borda-count voting.

        Each agent ranks all candidate objectives by distance to its own
        preferred objective.  The objective with the highest total Borda
        score wins.

        Returns
        -------
        ObjectiveVector
            The winning objective (the one with the highest Borda score).

        Raises
        ------
        ValueError
            If agents have no objectives or dimensions are mismatched.
        """
        objectives = ConsensusNegotiator._extract_objectives(agents)
        n = len(objectives)

        if n == 1:
            return objectives[0]

        # Borda scores: one entry per candidate objective
        borda_scores = np.zeros(n, dtype=np.float64)

        for voter_idx in range(n):
            voter_obj = objectives[voter_idx]

            # Compute distances from voter's objective to every candidate
            distances: list[tuple[float, int]] = []
            for candidate_idx in range(n):
                dist = voter_obj.distance_to(objectives[candidate_idx])
                distances.append((dist, candidate_idx))

            # Sort by distance ascending (closest first = highest rank)
            distances.sort(key=lambda x: x[0])

            # Assign Borda points: closest gets (n-1), farthest gets 0
            for rank, (_, candidate_idx) in enumerate(distances):
                borda_scores[candidate_idx] += (n - 1 - rank)

        # Select the winner (highest Borda score)
        winner_idx = int(np.argmax(borda_scores))

        logger.info(
            "VotingNegotiator: winner is agent %d's objective "
            "(borda_score=%.1f, scores=%s)",
            winner_idx,
            borda_scores[winner_idx],
            [float(s) for s in borda_scores],
        )

        return objectives[winner_idx]


# ===================================================================== #
#  Auction Negotiator                                                     #
# ===================================================================== #


class AuctionNegotiator(BaseNegotiator):
    """Bid-based dimension allocation negotiation.

    Each agent is given a budget to distribute across the dimensions of
    the objective vector.  The bid for each dimension is proportional to
    how far that dimension is from the agent's desired value (the agent
    bids more on dimensions it cares most about).

    For each dimension, the agent with the highest bid wins, and the
    winning agent's preferred value for that dimension is used in the
    final agreed-upon objective.

    Parameters
    ----------
    budget_per_agent:
        The total budget each agent can allocate across dimensions.
        Defaults to 100.0.
    """

    def __init__(self, budget_per_agent: float = 100.0) -> None:
        if budget_per_agent <= 0:
            raise ValueError(
                f"budget_per_agent must be positive, got {budget_per_agent}"
            )
        self._budget_per_agent = budget_per_agent

    @property
    def budget_per_agent(self) -> float:
        """The total budget each agent can allocate."""
        return self._budget_per_agent

    def negotiate(
        self,
        agents: list[BaseAgent],
        shared_state: StateSnapshot,
    ) -> ObjectiveVector:
        """Negotiate via bid-based dimension auction.

        Each agent distributes its budget proportionally to its deviation
        from the shared state on each dimension.  The highest bidder per
        dimension determines that dimension's value in the final objective.

        Returns
        -------
        ObjectiveVector
            The assembled objective where each dimension comes from the
            highest bidder for that dimension.

        Raises
        ------
        ValueError
            If agents have no objectives or dimensions are mismatched.
        """
        objectives = ConsensusNegotiator._extract_objectives(agents)
        n = len(objectives)
        dimension = objectives[0].dimension
        directions = objectives[0].directions
        weights = objectives[0].weights

        if n == 1:
            return objectives[0]

        # Use shared state values as the reference point for bid calculation
        state_arr = np.array(shared_state.values[:dimension], dtype=np.float64)
        if len(shared_state.values) < dimension:
            # Pad with zeros if state has fewer dimensions
            state_arr = np.zeros(dimension, dtype=np.float64)
            state_arr[: len(shared_state.values)] = shared_state.values

        # Each agent computes bids per dimension
        # bids[agent_idx][dim_idx] = agent's bid for that dimension
        bids = np.zeros((n, dimension), dtype=np.float64)

        for agent_idx, obj in enumerate(objectives):
            obj_arr = np.array(obj.values, dtype=np.float64)
            deviations = np.abs(obj_arr - state_arr)

            total_deviation = deviations.sum()
            if total_deviation > 0:
                # Distribute budget proportionally to deviation
                bids[agent_idx] = (
                    self._budget_per_agent * deviations / total_deviation
                )
            else:
                # Agent is at the state already: spread budget evenly
                bids[agent_idx] = self._budget_per_agent / dimension

        # For each dimension, the highest bidder wins
        final_values: list[float] = []
        winning_agents: list[int] = []

        for dim in range(dimension):
            dim_bids = bids[:, dim]
            winner_idx = int(np.argmax(dim_bids))
            winning_agents.append(winner_idx)
            final_values.append(objectives[winner_idx].values[dim])

        logger.info(
            "AuctionNegotiator: dimension winners=%s, bids_summary=%s",
            winning_agents,
            {
                f"agent_{i}": [f"{b:.2f}" for b in bids[i]]
                for i in range(n)
            },
        )

        return ObjectiveVector(
            values=tuple(final_values),
            directions=directions,
            weights=weights,
        )


# ===================================================================== #
#  Coordination Mediator                                                  #
# ===================================================================== #


_STRATEGY_MAP: dict[NegotiationStrategy, type[BaseNegotiator]] = {
    NegotiationStrategy.CONSENSUS: ConsensusNegotiator,
    NegotiationStrategy.VOTING: VotingNegotiator,
    NegotiationStrategy.AUCTION: AuctionNegotiator,
}


class CoordinationMediator:
    """Mediator that manages an agent network and orchestrates negotiation.

    The mediator maintains a registry of agents, selects and delegates to
    a negotiation strategy, and publishes domain events through the event
    bus.

    Parameters
    ----------
    negotiator:
        The negotiation strategy to use for resolving shared objectives.
    event_bus:
        Event bus for publishing ``NegotiationStarted`` and
        ``ConsensusReached`` events.
    mediator_id:
        Optional identifier for this mediator instance.

    Example
    -------
    ::

        mediator = CoordinationMediator(
            negotiator=ConsensusNegotiator(tolerance=0.01),
            event_bus=bus,
        )
        mediator.register_agent(agent_a)
        mediator.register_agent(agent_b)
        shared_objective = mediator.negotiate_shared_goal(current_state)
    """

    def __init__(
        self,
        negotiator: BaseNegotiator,
        event_bus: EventBus,
        mediator_id: str = "",
    ) -> None:
        self._negotiator = negotiator
        self._event_bus = event_bus
        self._mediator_id = mediator_id or str(uuid.uuid4())[:8]
        self._agents: dict[str, BaseAgent] = {}
        self._message_log: list[dict[str, Any]] = []

    # -- properties ---------------------------------------------------------

    @property
    def mediator_id(self) -> str:
        """Unique identifier for this mediator."""
        return self._mediator_id

    @property
    def negotiator(self) -> BaseNegotiator:
        """The current negotiation strategy."""
        return self._negotiator

    @negotiator.setter
    def negotiator(self, value: BaseNegotiator) -> None:
        """Replace the negotiation strategy."""
        self._negotiator = value

    @property
    def event_bus(self) -> EventBus:
        """The event bus used by this mediator."""
        return self._event_bus

    @property
    def agent_ids(self) -> list[str]:
        """Sorted list of registered agent identifiers."""
        return sorted(self._agents.keys())

    @property
    def agent_count(self) -> int:
        """Number of currently registered agents."""
        return len(self._agents)

    @property
    def message_log(self) -> list[dict[str, Any]]:
        """Chronological log of all broadcast messages."""
        return list(self._message_log)

    # -- agent management ---------------------------------------------------

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the mediator.

        If an agent with the same id is already registered, it is replaced.

        Parameters
        ----------
        agent:
            The agent to register.
        """
        self._agents[agent.id] = agent
        logger.info(
            "CoordinationMediator %s: registered agent %s (total=%d)",
            self._mediator_id,
            agent.id,
            len(self._agents),
        )

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the mediator's registry.

        Parameters
        ----------
        agent_id:
            Identifier of the agent to remove.

        Returns
        -------
        bool
            ``True`` if the agent was found and removed, ``False`` otherwise.
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            logger.info(
                "CoordinationMediator %s: removed agent %s (total=%d)",
                self._mediator_id,
                agent_id,
                len(self._agents),
            )
            return True
        logger.warning(
            "CoordinationMediator %s: agent %s not found for removal",
            self._mediator_id,
            agent_id,
        )
        return False

    def get_agent(self, agent_id: str) -> BaseAgent | None:
        """Retrieve a registered agent by id, or ``None`` if not found."""
        return self._agents.get(agent_id)

    def get_agents(self) -> list[BaseAgent]:
        """Return all registered agents in id-sorted order."""
        return [self._agents[aid] for aid in sorted(self._agents)]

    # -- negotiation --------------------------------------------------------

    def negotiate_shared_goal(
        self,
        shared_state: StateSnapshot,
    ) -> ObjectiveVector:
        """Orchestrate a negotiation round among all registered agents.

        Publishes a ``NegotiationStarted`` event before negotiation and a
        ``ConsensusReached`` event upon successful completion.

        Parameters
        ----------
        shared_state:
            The current shared environment state provided as context to
            the negotiation strategy.

        Returns
        -------
        ObjectiveVector
            The agreed-upon shared objective.

        Raises
        ------
        ValueError
            If no agents are registered.
        NegotiationDeadlock
            If the negotiation strategy fails to reach agreement.
        """
        if not self._agents:
            raise ValueError(
                "CoordinationMediator: cannot negotiate with zero agents"
            )

        agents = self.get_agents()
        participant_ids = tuple(a.id for a in agents)
        negotiation_id = str(uuid.uuid4())[:8]

        # Determine strategy enum for the event
        strategy_enum = self._resolve_strategy_enum()

        # Emit NegotiationStarted
        contested_goal_ids = tuple(
            a.current_goal.goal_id for a in agents
        )
        self._event_bus.publish(
            NegotiationStarted(
                source_id=self._mediator_id,
                negotiation_id=negotiation_id,
                participant_ids=participant_ids,
                strategy=strategy_enum,
                contested_goal_ids=contested_goal_ids,
                timestamp=time.time(),
            )
        )

        logger.info(
            "CoordinationMediator %s: starting negotiation %s with %d agents "
            "(strategy=%s)",
            self._mediator_id,
            negotiation_id,
            len(agents),
            strategy_enum.value,
        )

        # Delegate to the negotiator
        t0 = time.monotonic()
        agreed_objective = self._negotiator.negotiate(agents, shared_state)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        # Emit ConsensusReached
        self._event_bus.publish(
            ConsensusReached(
                source_id=self._mediator_id,
                negotiation_id=negotiation_id,
                agreed_objective=agreed_objective,
                participant_ids=participant_ids,
                rounds=self._estimate_rounds(),
                timestamp=time.time(),
            )
        )

        logger.info(
            "CoordinationMediator %s: negotiation %s completed in %.1f ms, "
            "agreed_objective=%s",
            self._mediator_id,
            negotiation_id,
            elapsed_ms,
            agreed_objective.values,
        )

        return agreed_objective

    # -- messaging ----------------------------------------------------------

    def broadcast_message(
        self,
        sender_id: str,
        content: Mapping[str, Any],
        *,
        exclude: set[str] | None = None,
    ) -> int:
        """Broadcast a message from one agent to all other registered agents.

        The message is recorded in the mediator's internal log.  This is a
        lightweight coordination channel for agents that need to share
        information outside of formal negotiation rounds.

        Parameters
        ----------
        sender_id:
            Identifier of the sending agent.
        content:
            Arbitrary message content (must be serializable).
        exclude:
            Optional set of agent ids to exclude from receiving the message.

        Returns
        -------
        int
            Number of agents that received the message.
        """
        exclude_set = exclude or set()
        recipients = [
            aid for aid in self._agents
            if aid != sender_id and aid not in exclude_set
        ]

        message_record: dict[str, Any] = {
            "timestamp": time.time(),
            "sender_id": sender_id,
            "content": dict(content),
            "recipient_ids": recipients,
        }
        self._message_log.append(message_record)

        logger.debug(
            "CoordinationMediator %s: broadcast from %s to %d agents",
            self._mediator_id,
            sender_id,
            len(recipients),
        )

        return len(recipients)

    # -- internal helpers ---------------------------------------------------

    def _resolve_strategy_enum(self) -> NegotiationStrategy:
        """Map the current negotiator instance to a NegotiationStrategy enum."""
        if isinstance(self._negotiator, ConsensusNegotiator):
            return NegotiationStrategy.CONSENSUS
        if isinstance(self._negotiator, VotingNegotiator):
            return NegotiationStrategy.VOTING
        if isinstance(self._negotiator, AuctionNegotiator):
            return NegotiationStrategy.AUCTION
        # Fallback for custom negotiators
        return NegotiationStrategy.CONSENSUS

    def _estimate_rounds(self) -> int:
        """Estimate the number of rounds used in the last negotiation.

        For ConsensusNegotiator this is meaningful; for others, return 1
        as they are single-round protocols.
        """
        if isinstance(self._negotiator, ConsensusNegotiator):
            # The exact round count is not exposed by the negotiator,
            # so we return the max_rounds as an upper bound.  A production
            # system would track this internally.
            return self._negotiator.max_rounds
        return 1

    def __repr__(self) -> str:
        return (
            f"CoordinationMediator("
            f"id={self._mediator_id!r}, "
            f"agents={len(self._agents)}, "
            f"negotiator={type(self._negotiator).__name__})"
        )
