"""LLM integration layer for the Synthetic Teleology framework.

This sub-package provides a **provider-agnostic** abstraction over large
language model backends (Anthropic, OpenAI, HuggingFace, generic HTTP).

Public API
----------
LLMProvider
    Abstract base class that every concrete provider must implement.
LLMResponse
    Structured response returned by all providers.
LLMConfig
    Per-call configuration (model, temperature, max_tokens, etc.).
LLMError
    Base exception for all LLM-related failures.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =========================================================================== #
#  Exceptions                                                                  #
# =========================================================================== #

class LLMError(Exception):
    """Base exception for LLM provider errors."""


class LLMConnectionError(LLMError):
    """Raised when the provider cannot be reached."""


class LLMRateLimitError(LLMError):
    """Raised when the provider returns a rate-limit / quota error."""


class LLMResponseError(LLMError):
    """Raised when the provider returns an unparseable or invalid response."""


# =========================================================================== #
#  Data structures                                                             #
# =========================================================================== #

@dataclass(frozen=True)
class LLMConfig:
    """Per-call configuration for an LLM request.

    Attributes
    ----------
    model:
        Model identifier (e.g. ``"claude-opus-4-6"``, ``"gpt-4o"``).
    temperature:
        Sampling temperature.
    max_tokens:
        Maximum number of tokens in the response.
    stop_sequences:
        Optional sequences that cause the model to stop generating.
    system_prompt:
        System-level prompt prepended to the conversation.
    extra:
        Provider-specific parameters passed through verbatim.
    """

    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    stop_sequences: tuple[str, ...] = ()
    system_prompt: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.model:
            raise ValueError("LLMConfig.model must not be empty")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(
                f"temperature must be in [0, 2], got {self.temperature}"
            )
        if self.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be >= 1, got {self.max_tokens}"
            )


@dataclass(frozen=True)
class LLMMessage:
    """A single message in a conversation.

    Attributes
    ----------
    role:
        One of ``"system"``, ``"user"``, ``"assistant"``.
    content:
        The text content of the message.
    """

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class LLMResponse:
    """Structured response from an LLM provider.

    Attributes
    ----------
    text:
        The generated text content.
    model:
        The model that actually produced the response (may differ from the
        requested model if the provider aliases or falls back).
    usage:
        Token usage statistics (provider-dependent keys).
    finish_reason:
        Why generation stopped (e.g. ``"stop"``, ``"max_tokens"``).
    raw:
        The full raw response object from the provider, for debugging.
    """

    text: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


# =========================================================================== #
#  Abstract provider                                                           #
# =========================================================================== #

class LLMProvider(ABC):
    """Abstract base class for LLM backend integrations.

    .. deprecated:: 1.0.0
        Use ``langchain_core.language_models.BaseChatModel`` directly instead.
        See :class:`synthetic_teleology.infrastructure.llm.langchain_bridge.LLMProviderToChatModel`
        for a bridge from legacy providers to LangChain.

    Usage::

        provider = AnthropicProvider(api_key="...")
        config = LLMConfig(model="claude-opus-4-6", temperature=0.3)
        response = provider.generate(messages, config)
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        import warnings

        super().__init_subclass__(**kwargs)
        warnings.warn(
            f"{cls.__name__} inherits from LLMProvider which is deprecated in v1.0. "
            "Use langchain_core.language_models.BaseChatModel instead. "
            "See synthetic_teleology.infrastructure.llm.langchain_bridge for migration.",
            DeprecationWarning,
            stacklevel=2,
        )

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier (e.g. ``"anthropic"``)."""
        ...

    @abstractmethod
    def generate(
        self,
        messages: Sequence[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Synchronously generate a response.

        Parameters
        ----------
        messages:
            Conversation history as a sequence of :class:`LLMMessage`.
        config:
            Per-call configuration.

        Returns
        -------
        LLMResponse

        Raises
        ------
        LLMError
            On any provider-level failure.
        """
        ...

    @abstractmethod
    async def generate_async(
        self,
        messages: Sequence[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Asynchronously generate a response.

        Same semantics as :meth:`generate` but for async contexts.
        """
        ...

    def validate_config(self, config: LLMConfig) -> None:
        """Optional hook for provider-specific config validation.

        The default implementation delegates to ``config.validate()``.
        Subclasses can override to add additional checks.
        """
        config.validate()


# =========================================================================== #
#  Public API                                                                  #
# =========================================================================== #

__all__ = [
    # Exceptions
    "LLMError",
    "LLMConnectionError",
    "LLMRateLimitError",
    "LLMResponseError",
    # Data
    "LLMConfig",
    "LLMMessage",
    "LLMResponse",
    # Abstract provider
    "LLMProvider",
]


# ---------------------------------------------------------------------------
# Lazy imports for concrete providers (avoids hard dependencies)
# ---------------------------------------------------------------------------

def __getattr__(name: str):  # noqa: N807
    """Lazy-load concrete providers on attribute access.

    This allows ``from synthetic_teleology.infrastructure.llm import AnthropicProvider``
    to work without forcing all optional dependencies to be installed.
    """
    _lazy_map = {
        "AnthropicProvider": "synthetic_teleology.infrastructure.llm.anthropic",
        "OpenAIProvider": "synthetic_teleology.infrastructure.llm.openai_provider",
        "GenericOpenAPIProvider": "synthetic_teleology.infrastructure.llm.openapi",
        "HuggingFaceLocalProvider": "synthetic_teleology.infrastructure.llm.huggingface",
        "ProviderRouter": "synthetic_teleology.infrastructure.llm.router",
        "LLMProviderFactory": "synthetic_teleology.infrastructure.llm.factory",
    }

    if name in _lazy_map:
        import importlib
        module = importlib.import_module(_lazy_map[name])
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
