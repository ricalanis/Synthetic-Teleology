"""High-level one-liner constructors for common teleological agents.

Each function returns a ``(compiled_graph, initial_state)`` tuple ready
for ``app.invoke(initial_state)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.graph.builder import GraphBuilder
from synthetic_teleology.services.evaluation import BaseEvaluator
from synthetic_teleology.services.goal_revision import BaseGoalUpdater
from synthetic_teleology.services.planning import BasePlanner


def create_teleological_agent(
    target_values: tuple[float, ...],
    perceive_fn: Callable,
    transition_fn: Callable | None = None,
    act_fn: Callable | None = None,
    *,
    directions: tuple[Direction, ...] | None = None,
    evaluator: BaseEvaluator | None = None,
    goal_updater: BaseGoalUpdater | None = None,
    planner: BasePlanner | None = None,
    max_steps: int = 100,
    goal_achieved_threshold: float = 0.9,
    step_size: float = 0.5,
    checkpointer: Any | None = None,
    agent_id: str = "agent",
) -> tuple[Any, dict[str, Any]]:
    """Create a numeric teleological agent as a compiled LangGraph.

    Parameters
    ----------
    target_values:
        Target values for the objective vector.
    perceive_fn:
        Callable returning a StateSnapshot.
    transition_fn:
        Callable taking an ActionSpec to transition the environment.
    act_fn:
        Callable selecting an action from policy and state.
    directions:
        Per-dimension optimization directions. Defaults to APPROACH.
    evaluator:
        Evaluation strategy. Defaults to NumericEvaluator.
    goal_updater:
        Goal revision strategy. Defaults to ThresholdUpdater.
    planner:
        Planning strategy. Defaults to GreedyPlanner with auto action space.
    max_steps:
        Maximum loop iterations.
    goal_achieved_threshold:
        Score threshold for early stopping.
    step_size:
        Step size for default action space.
    checkpointer:
        Optional LangGraph checkpointer.
    agent_id:
        Agent identifier.

    Returns
    -------
    tuple[CompiledStateGraph, dict]
        The compiled graph and initial state.
    """
    builder = (
        GraphBuilder(agent_id)
        .with_objective(target_values, directions=directions)
        .with_max_steps(max_steps)
        .with_goal_achieved_threshold(goal_achieved_threshold)
        .with_action_step_size(step_size)
        .with_environment(
            perceive_fn=perceive_fn,
            act_fn=act_fn,
            transition_fn=transition_fn,
        )
    )

    if evaluator is not None:
        builder.with_evaluator(evaluator)
    if goal_updater is not None:
        builder.with_goal_updater(goal_updater)
    if planner is not None:
        builder.with_planner(planner)
    if checkpointer is not None:
        builder.with_checkpointer(checkpointer)

    return builder.build()


def create_llm_teleological_agent(
    model: Any,
    target_values: tuple[float, ...],
    perceive_fn: Callable,
    transition_fn: Callable | None = None,
    *,
    directions: tuple[Direction, ...] | None = None,
    max_steps: int = 50,
    goal_achieved_threshold: float = 0.9,
    checkpointer: Any | None = None,
    agent_id: str = "llm-agent",
) -> tuple[Any, dict[str, Any]]:
    """Create an LLM-powered teleological agent.

    This is a convenience constructor that uses the standard numeric
    evaluation/planning but stores the LLM model reference in metadata
    for nodes that need it.

    Parameters
    ----------
    model:
        A LangChain chat model (e.g. ``ChatAnthropic``, ``ChatOpenAI``).
    target_values:
        Target objective values.
    perceive_fn:
        Environment observation function.
    transition_fn:
        Environment transition function.
    directions:
        Per-dimension directions.
    max_steps:
        Maximum iterations.
    goal_achieved_threshold:
        Score threshold.
    checkpointer:
        Optional checkpointer.
    agent_id:
        Agent ID.

    Returns
    -------
    tuple[CompiledStateGraph, dict]
        The compiled graph and initial state.
    """
    builder = (
        GraphBuilder(agent_id)
        .with_objective(target_values, directions=directions)
        .with_max_steps(max_steps)
        .with_goal_achieved_threshold(goal_achieved_threshold)
        .with_environment(perceive_fn=perceive_fn, transition_fn=transition_fn)
        .with_metadata(llm_model=model)
    )

    if checkpointer is not None:
        builder.with_checkpointer(checkpointer)

    return builder.build()


def create_react_teleological_agent(
    model: Any,
    tools: list[Any],
    goal_description: str,
    perceive_fn: Callable,
    transition_fn: Callable | None = None,
    *,
    target_values: tuple[float, ...] = (1.0,),
    directions: tuple[Direction, ...] | None = None,
    max_steps: int = 30,
    goal_achieved_threshold: float = 0.9,
    checkpointer: Any | None = None,
    agent_id: str = "react-agent",
) -> tuple[Any, dict[str, Any]]:
    """Create a ReAct agent wrapped in the teleological loop.

    Stores the LLM model and tools in metadata for custom node
    implementations to use.

    Parameters
    ----------
    model:
        A LangChain chat model.
    tools:
        List of LangChain tools for the ReAct agent.
    goal_description:
        Human-readable description of the goal.
    perceive_fn:
        Environment observation function.
    transition_fn:
        Environment transition function.
    target_values:
        Numeric target for evaluation.
    directions:
        Per-dimension directions.
    max_steps:
        Maximum iterations.
    goal_achieved_threshold:
        Score threshold.
    checkpointer:
        Optional checkpointer.
    agent_id:
        Agent ID.

    Returns
    -------
    tuple[CompiledStateGraph, dict]
        The compiled graph and initial state.
    """
    builder = (
        GraphBuilder(agent_id)
        .with_objective(target_values, directions=directions, goal_name=goal_description)
        .with_max_steps(max_steps)
        .with_goal_achieved_threshold(goal_achieved_threshold)
        .with_environment(perceive_fn=perceive_fn, transition_fn=transition_fn)
        .with_metadata(llm_model=model, tools=tools, goal_description=goal_description)
    )

    if checkpointer is not None:
        builder.with_checkpointer(checkpointer)

    return builder.build()
