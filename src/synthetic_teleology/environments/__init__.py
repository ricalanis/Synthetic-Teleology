"""Environment layer for the Synthetic Teleology framework.

Re-exports public environment types for convenient top-level access::

    from synthetic_teleology.environments import (
        BaseEnvironment,
        NumericEnvironment,
        ResourceEnvironment,
        ResearchEnvironment,
        SharedEnvironment,
    )
"""

from synthetic_teleology.environments.base import BaseEnvironment
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.environments.research import ResearchEnvironment
from synthetic_teleology.environments.resource import ResourceEnvironment
from synthetic_teleology.environments.shared import SharedEnvironment

__all__ = [
    "BaseEnvironment",
    "NumericEnvironment",
    "ResourceEnvironment",
    "ResearchEnvironment",
    "SharedEnvironment",
]
