"""Synthetic Teleology Framework.

LangGraph toolkit for goal-directed agents implementing Haidemariam (2026)
"From the logic of coordination to goal-directed reasoning" —
the theory of Synthetic Teleology in Agentic AI.
"""

__version__ = "1.5.0"

from synthetic_teleology.graph import (
    GraphBuilder,
    TeleologicalState,
    build_teleological_graph,
)

__all__ = [
    "build_teleological_graph",
    "GraphBuilder",
    "TeleologicalState",
]
