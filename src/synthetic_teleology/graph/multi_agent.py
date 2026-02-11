"""Multi-agent coordination as a LangGraph.

Builds a coordinator graph that runs per-agent teleological subgraphs
and negotiates shared objectives using the existing negotiation strategies.

Note: No ``from __future__ import annotations`` — LangGraph needs runtime
type resolution for TypedDict schemas.
"""

import logging
import operator
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import ObjectiveVector
from synthetic_teleology.services.coordination import BaseNegotiator, ConsensusNegotiator

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for one agent in a multi-agent graph."""

    agent_id: str
    goal: Goal
    perceive_fn: Callable
    transition_fn: Callable | None = None
    act_fn: Callable | None = None
    max_steps_per_round: int = 10


class MultiAgentState(TypedDict, total=False):
    """State for the multi-agent coordination graph."""

    # Shared strategies (injected once)
    evaluator: Any
    goal_updater: Any
    planner: Any
    constraint_pipeline: Any
    policy_filter: Any
    goal_achieved_threshold: float

    # Coordination
    agent_results: dict[str, Any]
    shared_objective: ObjectiveVector
    negotiation_round: int
    max_rounds: int

    # Accumulation
    events: Annotated[list, operator.add]


def _make_agent_node(
    config: AgentConfig,
    agent_key: str,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a node function that runs a teleological subgraph for one agent."""

    def agent_node(state: dict[str, Any]) -> dict[str, Any]:
        from synthetic_teleology.graph.graph import build_teleological_graph

        shared_objective = state.get("shared_objective")
        agent_goal = config.goal

        # Apply shared objective if available from negotiation
        if shared_objective is not None and agent_goal.objective is not None:
            agent_goal = Goal(
                name=agent_goal.name,
                objective=shared_objective,
            )

        sub_app = build_teleological_graph()
        sub_state = {
            "step": 0,
            "max_steps": config.max_steps_per_round,
            "goal_achieved_threshold": state.get("goal_achieved_threshold", 0.9),
            "goal": agent_goal,
            "evaluator": state["evaluator"],
            "goal_updater": state["goal_updater"],
            "planner": state["planner"],
            "constraint_pipeline": state["constraint_pipeline"],
            "policy_filter": state["policy_filter"],
            "perceive_fn": config.perceive_fn,
            "act_fn": config.act_fn,
            "transition_fn": config.transition_fn,
            "events": [],
            "goal_history": [],
            "eval_history": [],
            "action_history": [],
            "metadata": {},
        }

        result = sub_app.invoke(sub_state)

        # Store agent result
        agent_results = dict(state.get("agent_results", {}))
        agent_results[config.agent_id] = {
            "final_goal": result.get("goal"),
            "eval_signal": result.get("eval_signal"),
            "steps": result.get("step", 0),
            "stop_reason": result.get("stop_reason"),
            "events": result.get("events", []),
        }

        event = {
            "type": "agent_round_completed",
            "agent_id": config.agent_id,
            "steps": result.get("step", 0),
            "eval_score": result["eval_signal"].score if result.get("eval_signal") else 0.0,
            "timestamp": time.time(),
        }

        return {
            "agent_results": agent_results,
            "events": [event],
        }

    agent_node.__name__ = f"agent_{agent_key}"
    return agent_node


def _negotiate_node(negotiator: BaseNegotiator) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a negotiation node using the provided strategy."""

    def negotiate(state: dict[str, Any]) -> dict[str, Any]:
        agent_results = state.get("agent_results", {})
        if not agent_results:
            return {}

        objectives = []
        for _agent_id, result in agent_results.items():
            goal = result.get("final_goal")
            if goal is not None and goal.objective is not None:
                objectives.append(goal.objective)

        if len(objectives) < 2:
            if objectives:
                return {"shared_objective": objectives[0]}
            return {}

        # Use simple averaging for graph-based negotiation (no agent objects needed)
        import numpy as np

        values_list = [np.array(obj.values) for obj in objectives]
        mean_values = np.mean(values_list, axis=0)
        ref = objectives[0]

        shared = ObjectiveVector(
            values=tuple(float(v) for v in mean_values),
            directions=ref.directions,
            weights=ref.weights,
        )

        round_num = state.get("negotiation_round", 0) + 1
        event = {
            "type": "negotiation_completed",
            "round": round_num,
            "shared_objective": shared.values,
            "timestamp": time.time(),
        }

        return {
            "shared_objective": shared,
            "negotiation_round": round_num,
            "events": [event],
        }

    return negotiate


def _should_continue_multi(state: dict[str, Any]) -> str:
    """Check if multi-agent loop should continue."""
    round_num = state.get("negotiation_round", 0)
    max_rounds = state.get("max_rounds", 5)

    if round_num >= max_rounds:
        return "__end__"

    # Check if any agent achieved its goal
    agent_results = state.get("agent_results", {})
    all_achieved = all(
        r.get("stop_reason") == "goal_achieved"
        for r in agent_results.values()
    ) if agent_results else False

    if all_achieved:
        return "__end__"

    return "negotiate"


def build_multi_agent_graph(
    agent_configs: list[AgentConfig],
    negotiation_strategy: BaseNegotiator | None = None,
    max_rounds: int = 5,
    checkpointer: Any = None,
) -> Any:
    """Build a multi-agent coordination graph.

    Each agent runs its own teleological subgraph, then a negotiation
    step produces a shared objective for the next round.

    Parameters
    ----------
    agent_configs:
        List of per-agent configurations.
    negotiation_strategy:
        Negotiation strategy. Defaults to ConsensusNegotiator.
    max_rounds:
        Maximum negotiation rounds.
    checkpointer:
        Optional LangGraph checkpointer.

    Returns
    -------
    CompiledStateGraph
        A compiled multi-agent graph.
    """
    negotiator = negotiation_strategy or ConsensusNegotiator()

    graph = StateGraph(MultiAgentState)

    # Add agent nodes
    agent_keys = []
    for i, config in enumerate(agent_configs):
        key = f"agent_{i}"
        agent_keys.append(key)
        node_fn = _make_agent_node(config, key)
        graph.add_node(key, node_fn)

    # Add negotiation node
    graph.add_node("negotiate", _negotiate_node(negotiator))

    # Wire: START -> first agent -> second agent -> ... -> check -> negotiate -> loop
    graph.add_edge(START, agent_keys[0])
    for i in range(len(agent_keys) - 1):
        graph.add_edge(agent_keys[i], agent_keys[i + 1])

    # Last agent -> conditional: continue or end
    graph.add_conditional_edges(
        agent_keys[-1],
        _should_continue_multi,
        {"negotiate": "negotiate", "__end__": END},
    )

    # Negotiate -> first agent (loop back)
    graph.add_edge("negotiate", agent_keys[0])

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)
