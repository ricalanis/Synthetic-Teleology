"""Public testing utilities for Synthetic Teleology.

Provides mock LLM models for writing self-contained examples and tests
without requiring API keys.
"""

from synthetic_teleology.testing.mock_llm import MockStructuredChatModel

__all__ = ["MockStructuredChatModel"]
