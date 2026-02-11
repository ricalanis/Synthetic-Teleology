"""Agent layer for the Synthetic Teleology framework.

Re-exports public agent types for convenient top-level access::

    from synthetic_teleology.agents import (
        BaseAgent,
        TeleologicalAgent,
        IntentionalStateAgent,
        BDIAgent,  # deprecated alias
        LLMAgent,
        AgentFactory,
        AgentBuilder,
    )
"""

from synthetic_teleology.agents.base import BaseAgent
from synthetic_teleology.agents.bdi import BDIAgent
from synthetic_teleology.agents.factory import AgentBuilder, AgentFactory
from synthetic_teleology.agents.intentional import IntentionalStateAgent
from synthetic_teleology.agents.llm import LLMAgent
from synthetic_teleology.agents.teleological import TeleologicalAgent

__all__ = [
    "BaseAgent",
    "TeleologicalAgent",
    "IntentionalStateAgent",
    "BDIAgent",  # deprecated alias
    "LLMAgent",
    "AgentFactory",
    "AgentBuilder",
]
