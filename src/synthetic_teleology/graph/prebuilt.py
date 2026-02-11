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

# -- LLM Mode (new in v1.0) -------------------------------------------------


def create_llm_agent(
    model: Any,
    goal: str,
    tools: list[Any] | None = None,
    criteria: list[str] | None = None,
    constraints: list[str] | None = None,
    *,
    perceive_fn: Callable | None = None,
    transition_fn: Callable | None = None,
    act_fn: Callable | None = None,
    max_steps: int = 20,
    num_hypotheses: int = 3,
    temperature: float = 0.7,
    goal_achieved_threshold: float = 0.9,
    checkpointer: Any | None = None,
    agent_id: str = "llm-agent",
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Create an LLM-powered teleological agent.

    One-liner to create an agent where all reasoning (evaluation, planning,
    revision, constraint checking) is done by an LLM.

    Parameters
    ----------
    model:
        A LangChain chat model (e.g. ``ChatAnthropic``, ``ChatOpenAI``).
    goal:
        Natural language goal description.
    tools:
        Optional LangChain tools the agent can use.
    criteria:
        Success criteria for evaluation.
    constraints:
        Natural language constraints.
    perceive_fn:
        Optional environment observation function. If not provided,
        a default stub is used.
    transition_fn:
        Optional environment transition function.
    act_fn:
        Optional action selection function.
    max_steps:
        Maximum loop iterations (default 20).
    num_hypotheses:
        Number of plan candidates (default 3).
    temperature:
        LLM sampling temperature (default 0.7).
    goal_achieved_threshold:
        Score threshold for early stopping (default 0.9).
    checkpointer:
        Optional LangGraph checkpointer.
    agent_id:
        Agent identifier.

    Returns
    -------
    tuple[CompiledStateGraph, dict]
        The compiled graph and initial state.

    Example
    -------
    ::

        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        app, state = create_llm_agent(
            model=model,
            goal="Write a comprehensive market analysis report",
            criteria=["Covers at least 5 competitors", "Includes revenue data"],
            constraints=["Use only public data sources"],
            max_steps=10,
        )
        result = app.invoke(state)
    """
    builder = (
        GraphBuilder(agent_id)
        .with_model(model)
        .with_goal(goal, criteria=criteria)
        .with_max_steps(max_steps)
        .with_goal_achieved_threshold(goal_achieved_threshold)
        .with_num_hypotheses(num_hypotheses)
        .with_temperature(temperature)
    )

    if tools:
        builder.with_tools(*tools)
    if constraints:
        builder.with_constraints(*constraints)
    if perceive_fn is not None or transition_fn is not None or act_fn is not None:
        builder.with_environment(
            perceive_fn=perceive_fn,
            act_fn=act_fn,
            transition_fn=transition_fn,
        )
    if checkpointer is not None:
        builder.with_checkpointer(checkpointer)
    if interrupt_before or interrupt_after:
        builder.with_human_approval(before=interrupt_before, after=interrupt_after)

    return builder.build()


# -- Numeric Mode (backward compatible) -------------------------------------


def create_numeric_agent(
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
    """Create a numeric teleological agent (backward compatible).

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


# -- Legacy aliases ----------------------------------------------------------

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
    """Create a numeric teleological agent (legacy alias for ``create_numeric_agent``)."""
    return create_numeric_agent(
        target_values=target_values,
        perceive_fn=perceive_fn,
        transition_fn=transition_fn,
        act_fn=act_fn,
        directions=directions,
        evaluator=evaluator,
        goal_updater=goal_updater,
        planner=planner,
        max_steps=max_steps,
        goal_achieved_threshold=goal_achieved_threshold,
        step_size=step_size,
        checkpointer=checkpointer,
        agent_id=agent_id,
    )


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
    """Create an LLM-powered teleological agent (legacy — prefers ``create_llm_agent``)."""
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
    """Create a ReAct agent wrapped in the teleological loop (legacy)."""
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
