"""Infrastructure layer for the Synthetic Teleology framework.

Re-exports the public API surface for convenience::

    from synthetic_teleology.infrastructure import (
        EventBus, AsyncEventBus, EventStore,
        ComponentRegistry, registry,
        LoopConfig, AgentConfig, BenchmarkConfig, EnvironmentConfig,
    )
"""

from synthetic_teleology.infrastructure.config import (
    AgentConfig,
    BenchmarkConfig,
    EnvironmentConfig,
    LoopConfig,
    load_config_from_json,
)
from synthetic_teleology.infrastructure.event_bus import (
    AsyncEventBus,
    EventBus,
    EventStore,
)
from synthetic_teleology.infrastructure.llm import (
    LLMConfig,
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMResponse,
)
from synthetic_teleology.infrastructure.registry import (
    ComponentRegistry,
    registry,
)
from synthetic_teleology.infrastructure.serialization import (
    deserialize,
    from_json,
    from_yaml,
    serialize,
    to_json,
    to_yaml,
    yaml_available,
)

__all__ = [
    # Event bus
    "EventBus",
    "AsyncEventBus",
    "EventStore",
    # Registry
    "ComponentRegistry",
    "registry",
    # Configuration
    "LoopConfig",
    "AgentConfig",
    "BenchmarkConfig",
    "EnvironmentConfig",
    "load_config_from_json",
    # Serialization
    "serialize",
    "deserialize",
    "to_json",
    "from_json",
    "to_yaml",
    "from_yaml",
    "yaml_available",
    # LLM
    "LLMConfig",
    "LLMError",
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
]
