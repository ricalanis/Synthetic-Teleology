"""Backward-compatibility shim for the BDI-LangGraph bridge.

.. deprecated::
    BDI bridge functions have been renamed to use "intentional" naming.
    Import from ``synthetic_teleology.graph.intentional_bridge`` instead.
"""

from synthetic_teleology.graph.intentional_bridge import (
    build_intentional_teleological_graph,
    make_intentional_perceive_node,
    make_intentional_plan_node,
    make_intentional_revise_node,
)

# Deprecated aliases
make_bdi_perceive_node = make_intentional_perceive_node
"""Deprecated alias for :func:`make_intentional_perceive_node`."""

make_bdi_revise_node = make_intentional_revise_node
"""Deprecated alias for :func:`make_intentional_revise_node`."""

make_bdi_plan_node = make_intentional_plan_node
"""Deprecated alias for :func:`make_intentional_plan_node`."""

build_bdi_teleological_graph = build_intentional_teleological_graph
"""Deprecated alias for :func:`build_intentional_teleological_graph`."""

__all__ = [
    "make_bdi_perceive_node",
    "make_bdi_revise_node",
    "make_bdi_plan_node",
    "build_bdi_teleological_graph",
]
