"""Multi-agent coordination as a LangGraph.

Builds a coordinator graph that runs per-agent teleological subgraphs
and negotiates shared objectives using either numeric averaging or
LLM-powered negotiation.

Supports parallel execution via LangGraph ``Send`` API for fan-out/fan-in.

Note: No ``from __future__ import annotations`` — LangGraph needs runtime
type resolution for TypedDict schemas.
"""

import logging
import operator
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import ObjectiveVector
from synthetic_teleology.services.coordination import BaseNegotiator, ConsensusNegotiator

logger = logging.getLogger(__name__)


def _merge_agent_results(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    """Custom reducer for merging parallel agent results."""
    merged = dict(left)
    merged.update(right)
    return merged


@dataclass
class AgentConfig:
    """Configuration for one agent in a multi-agent graph.

    In LLM mode, set ``goal`` as a text description and provide ``model``,
    ``tools``, ``criteria``, and ``constraints``.
    In numeric mode, set ``goal`` as a ``Goal`` entity with an objective vector.
    """

    agent_id: str
    goal: Goal | str = ""
    model: Any = None
    tools: list[Any] = field(default_factory=list)
    criteria: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    perceive_fn: Callable | None = None
    transition_fn: Callable | None = None
    act_fn: Callable | None = None
    max_steps_per_round: int = 10
    bdi_agent: Any = None


class MultiAgentState(TypedDict, total=False):
    """State for the multi-agent coordination graph."""

    # Shared strategies (injected once)
    evaluator: Any
    goal_updater: Any
    planner: Any
    constraint_pipeline: Any
    policy_filter: Any
    goal_achieved_threshold: float

    # LLM configuration
    model: Any
    tools: list[Any]

    # Coordination
    agent_results: Annotated[dict[str, Any], _merge_agent_results]
    shared_objective: ObjectiveVector | None
    shared_direction: str
    negotiation_round: int
    max_rounds: int

    # Accumulation
    events: Annotated[list, operator.add]
    reasoning_trace: Annotated[list, operator.add]


def _make_agent_node(
    config: AgentConfig,
    agent_key: str,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a node function that runs a teleological subgraph for one agent."""

    def agent_node(state: dict[str, Any]) -> dict[str, Any]:
        # Determine if this agent should run in LLM mode
        agent_model = config.model or state.get("model")
        is_llm = agent_model is not None and isinstance(config.goal, str)

        if is_llm:
            return _run_llm_agent(config, agent_key, state, agent_model)
        return _run_numeric_agent(config, agent_key, state)

    agent_node.__name__ = f"agent_{agent_key}"
    return agent_node


def _run_llm_agent(
    config: AgentConfig,
    agent_key: str,
    state: dict[str, Any],
    model: Any,
) -> dict[str, Any]:
    """Run an LLM-mode agent subgraph."""
    from synthetic_teleology.graph.builder import GraphBuilder

    goal_text = config.goal if isinstance(config.goal, str) else config.goal.description

    # Prepend shared direction if available from negotiation
    shared_direction = state.get("shared_direction", "")
    if shared_direction:
        goal_text = f"{goal_text}\n\nShared direction from negotiation: {shared_direction}"

    builder = (
        GraphBuilder(config.agent_id)
        .with_model(model)
        .with_goal(goal_text, criteria=config.criteria)
        .with_max_steps(config.max_steps_per_round)
        .with_goal_achieved_threshold(state.get("goal_achieved_threshold", 0.9))
    )

    if config.tools:
        builder.with_tools(*config.tools)
    if config.constraints:
        builder.with_constraints(*config.constraints)
    if config.perceive_fn is not None:
        builder.with_environment(
            perceive_fn=config.perceive_fn,
            act_fn=config.act_fn,
            transition_fn=config.transition_fn,
        )

    sub_app, sub_state = builder.build()
    result = sub_app.invoke(sub_state)

    return _collect_agent_result(config, state, result)


def _run_numeric_agent(
    config: AgentConfig,
    agent_key: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Run a numeric-mode agent subgraph."""
    from synthetic_teleology.graph.graph import build_teleological_graph

    shared_objective = state.get("shared_objective")
    agent_goal = config.goal if isinstance(config.goal, Goal) else Goal(
        name=config.agent_id,
        description=str(config.goal),
    )

    # Apply shared objective if available from negotiation
    if shared_objective is not None and agent_goal.objective is not None:
        agent_goal = Goal(
            name=agent_goal.name,
            objective=shared_objective,
        )

    sub_app = build_teleological_graph(
        evaluator=state.get("evaluator"),
        goal_updater=state.get("goal_updater"),
        planner=state.get("planner"),
        constraint_pipeline=state.get("constraint_pipeline"),
        policy_filter=state.get("policy_filter"),
    )
    sub_state: dict[str, Any] = {
        "step": 0,
        "max_steps": config.max_steps_per_round,
        "goal_achieved_threshold": state.get("goal_achieved_threshold", 0.9),
        "goal": agent_goal,
        "perceive_fn": config.perceive_fn,
        "act_fn": config.act_fn,
        "transition_fn": config.transition_fn,
        "events": [],
        "goal_history": [],
        "eval_history": [],
        "action_history": [],
        "reasoning_trace": [],
        "metadata": {},
    }

    result = sub_app.invoke(sub_state)
    return _collect_agent_result(config, state, result)


def _collect_agent_result(
    config: AgentConfig,
    state: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Collect agent results into the multi-agent state."""
    agent_results = {config.agent_id: {
        "final_goal": result.get("goal"),
        "eval_signal": result.get("eval_signal"),
        "steps": result.get("step", 0),
        "stop_reason": result.get("stop_reason"),
        "events": result.get("events", []),
        "reasoning_trace": result.get("reasoning_trace", []),
    }}

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


def _negotiate_node(
    negotiator: BaseNegotiator,
    negotiation_model: Any = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a negotiation node using the provided strategy."""

    def negotiate(state: dict[str, Any]) -> dict[str, Any]:
        agent_results = state.get("agent_results", {})
        if not agent_results:
            return {}

        # Try LLM negotiation if model provided and agents have text goals
        if negotiation_model is not None:
            return _llm_negotiate(state, agent_results, negotiation_model)

        return _numeric_negotiate(state, agent_results)

    return negotiate


def _llm_negotiate(
    state: dict[str, Any],
    agent_results: dict[str, Any],
    model: Any,
) -> dict[str, Any]:
    """LLM-powered negotiation."""
    from synthetic_teleology.services.llm_negotiation import LLMNegotiator

    negotiator = LLMNegotiator(
        model=model,
        max_dialogue_rounds=state.get("max_dialogue_rounds", 3),
    )

    try:
        consensus = negotiator.negotiate(agent_results)
        round_num = state.get("negotiation_round", 0) + 1

        event = {
            "type": "llm_negotiation_completed",
            "round": round_num,
            "shared_direction": consensus.shared_direction[:200],
            "confidence": consensus.confidence,
            "timestamp": time.time(),
        }

        result: dict[str, Any] = {
            "shared_direction": consensus.shared_direction,
            "negotiation_round": round_num,
            "events": [event],
            "reasoning_trace": [
                {
                    "node": "negotiate",
                    "round": round_num,
                    "reasoning": consensus.reasoning,
                    "timestamp": time.time(),
                }
            ],
        }

        return result
    except Exception as exc:
        logger.warning("LLM negotiation failed, falling back to numeric: %s", exc)
        return _numeric_negotiate(state, agent_results)


def _numeric_negotiate(
    state: dict[str, Any],
    agent_results: dict[str, Any],
) -> dict[str, Any]:
    """Numeric averaging negotiation (backward compat)."""
    objectives = []
    for _agent_id, result in agent_results.items():
        goal = result.get("final_goal")
        if goal is not None and goal.objective is not None:
            objectives.append(goal.objective)

    if len(objectives) < 2:
        if objectives:
            return {"shared_objective": objectives[0]}
        return {}

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
    negotiation_model: Any = None,
    max_dialogue_rounds: int = 3,
    parallel: bool = False,
) -> Any:
    """Build a multi-agent coordination graph.

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
    negotiation_model:
        Optional LLM model for dialogue-based negotiation.
    max_dialogue_rounds:
        Rounds within each LLM negotiation dialogue.
    parallel:
        If True, run agents in parallel using LangGraph Send API.
        If False, run agents sequentially (backward compat).

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
    graph.add_node("negotiate", _negotiate_node(negotiator, negotiation_model))

    if parallel and len(agent_keys) > 1:
        # Parallel: fan-out to all agents, then fan-in to collect_results
        # Use a dispatch node that sends to all agents
        def dispatch_node(state: dict[str, Any]) -> list[Any]:
            """Fan-out: dispatch to all agent nodes."""
            from langgraph.types import Send
            return [Send(key, state) for key in agent_keys]

        graph.add_node("dispatch", dispatch_node)

        # Collect results node (identity — results already merged by reducer)
        def collect_results(state: dict[str, Any]) -> dict[str, Any]:
            return {}

        graph.add_node("collect_results", collect_results)

        graph.add_edge(START, "dispatch")
        # Each agent edge handled by Send
        for key in agent_keys:
            graph.add_edge(key, "collect_results")

        graph.add_conditional_edges(
            "collect_results",
            _should_continue_multi,
            {"negotiate": "negotiate", "__end__": END},
        )
        graph.add_edge("negotiate", "dispatch")
    else:
        # Sequential: START -> agent_0 -> agent_1 -> ... -> negotiate -> loop
        graph.add_edge(START, agent_keys[0])
        for i in range(len(agent_keys) - 1):
            graph.add_edge(agent_keys[i], agent_keys[i + 1])

        graph.add_conditional_edges(
            agent_keys[-1],
            _should_continue_multi,
            {"negotiate": "negotiate", "__end__": END},
        )
        graph.add_edge("negotiate", agent_keys[0])

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)
