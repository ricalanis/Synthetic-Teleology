"""LangGraph-native teleological loop.

Public API
----------
build_teleological_graph
    Build and compile the core Perceive-Evaluate-Revise-Plan-Filter-Act-Reflect graph.
TeleologicalState
    The TypedDict state flowing through the graph.
GraphBuilder
    Fluent builder producing compiled graphs.
build_multi_agent_graph
    Build a multi-agent coordination graph.

Prebuilt constructors:
    create_teleological_agent, create_llm_teleological_agent, create_react_teleological_agent

Node functions (for advanced customisation):
    perceive_node, evaluate_node, revise_node, check_constraints_node,
    plan_node, filter_policy_node, act_node, reflect_node

Edge functions:
    should_continue, should_revise
"""

from synthetic_teleology.graph.bdi_bridge import (
    build_bdi_teleological_graph,
    make_bdi_perceive_node,
    make_bdi_plan_node,
    make_bdi_revise_node,
)
from synthetic_teleology.graph.builder import GraphBuilder
from synthetic_teleology.graph.edges import should_continue, should_revise
from synthetic_teleology.graph.graph import build_teleological_graph
from synthetic_teleology.graph.intentional_bridge import (
    build_intentional_teleological_graph,
    make_intentional_perceive_node,
    make_intentional_plan_node,
    make_intentional_revise_node,
)
from synthetic_teleology.graph.multi_agent import AgentConfig, build_multi_agent_graph
from synthetic_teleology.graph.nodes import (
    act_node,
    check_constraints_node,
    evaluate_node,
    filter_policy_node,
    perceive_node,
    plan_node,
    reflect_node,
    revise_node,
)
from synthetic_teleology.graph.prebuilt import (
    create_llm_agent,
    create_llm_teleological_agent,
    create_numeric_agent,
    create_react_teleological_agent,
    create_teleological_agent,
)
from synthetic_teleology.graph.state import TeleologicalState
from synthetic_teleology.graph.streaming import (
    collect_stream_events,
    format_stream_events,
    stream_to_agent_log_entries,
)
from synthetic_teleology.graph.working_memory import WorkingMemory

__all__ = [
    "TeleologicalState",
    "build_teleological_graph",
    "GraphBuilder",
    "build_multi_agent_graph",
    "AgentConfig",
    # Prebuilt (v1.0)
    "create_llm_agent",
    "create_numeric_agent",
    # Prebuilt (legacy)
    "create_teleological_agent",
    "create_llm_teleological_agent",
    "create_react_teleological_agent",
    # Nodes
    "perceive_node",
    "evaluate_node",
    "revise_node",
    "check_constraints_node",
    "plan_node",
    "filter_policy_node",
    "act_node",
    "reflect_node",
    # Intentional State Mapping
    "build_intentional_teleological_graph",
    "make_intentional_perceive_node",
    "make_intentional_revise_node",
    "make_intentional_plan_node",
    # Intentional State Mapping (deprecated BDI aliases)
    "build_bdi_teleological_graph",
    "make_bdi_perceive_node",
    "make_bdi_revise_node",
    "make_bdi_plan_node",
    # Edges
    "should_continue",
    "should_revise",
    # Streaming
    "format_stream_events",
    "collect_stream_events",
    "stream_to_agent_log_entries",
    # Working memory
    "WorkingMemory",
]
