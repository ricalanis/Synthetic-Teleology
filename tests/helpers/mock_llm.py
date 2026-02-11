"""Mock LLM for testing — re-exports from ``synthetic_teleology.testing``."""

from synthetic_teleology.testing.mock_llm import (
    MockStructuredChatModel,
    _StructuredOutputRunnable,
)

__all__ = ["MockStructuredChatModel", "_StructuredOutputRunnable"]
