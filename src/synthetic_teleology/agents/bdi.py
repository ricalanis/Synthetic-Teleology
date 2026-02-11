"""Backward-compatibility shim for the BDI agent.

.. deprecated::
    ``BDIAgent`` has been renamed to :class:`IntentionalStateAgent`.
    Import from ``synthetic_teleology.agents.intentional`` instead.
"""

from synthetic_teleology.agents.intentional import IntentionalStateAgent

BDIAgent = IntentionalStateAgent
"""Deprecated alias for :class:`IntentionalStateAgent`."""

__all__ = ["BDIAgent"]
