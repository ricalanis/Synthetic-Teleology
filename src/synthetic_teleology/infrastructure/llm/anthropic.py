"""Anthropic Claude provider for the Synthetic Teleology LLM layer.

Wraps the ``anthropic`` Python SDK to implement the :class:`LLMProvider`
interface.  Handles rate limiting with exponential backoff, connection
errors, and response parsing.

Requires the ``anthropic`` package (``pip install anthropic``).  If the
package is not installed, a clear error is raised at instantiation time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Sequence

from synthetic_teleology.infrastructure.llm import (
    LLMConfig,
    LLMConnectionError,
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    LLMResponseError,
)

logger = logging.getLogger(__name__)

# Attempt import at module level for type checking; actual use is guarded
try:
    import anthropic as _anthropic_module

    _HAS_ANTHROPIC = True
except ImportError:
    _anthropic_module = None  # type: ignore[assignment]
    _HAS_ANTHROPIC = False


def _check_anthropic_available() -> None:
    """Raise a clear error if the anthropic package is not installed."""
    if not _HAS_ANTHROPIC:
        raise ImportError(
            "The 'anthropic' package is required for AnthropicProvider. "
            "Install it with: pip install anthropic"
        )


class AnthropicProvider(LLMProvider):
    """LLM provider for Anthropic Claude models.

    Parameters
    ----------
    api_key:
        Anthropic API key.  If ``None``, the ``ANTHROPIC_API_KEY``
        environment variable is used.
    default_model:
        Default model identifier.  Defaults to ``"claude-sonnet-4-20250514"``.
    max_retries:
        Maximum number of retries on rate-limit errors.  Defaults to 3.
    base_retry_delay:
        Base delay in seconds for exponential backoff.  Defaults to 1.0.
    timeout:
        Request timeout in seconds.  Defaults to 60.0.

    Raises
    ------
    ImportError
        If the ``anthropic`` package is not installed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        base_retry_delay: float = 1.0,
        timeout: float = 60.0,
    ) -> None:
        _check_anthropic_available()

        self._default_model = default_model
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay
        self._timeout = timeout

        # Create sync and async clients
        client_kwargs: dict = {"timeout": timeout}
        if api_key is not None:
            client_kwargs["api_key"] = api_key

        self._client = _anthropic_module.Anthropic(**client_kwargs)
        self._async_client = _anthropic_module.AsyncAnthropic(**client_kwargs)

    @property
    def provider_name(self) -> str:
        """Return ``'anthropic'``."""
        return "anthropic"

    def validate_config(self, config: LLMConfig) -> None:
        """Validate config with Anthropic-specific checks."""
        config.validate()
        # Anthropic models require max_tokens to be explicitly set
        if config.max_tokens < 1:
            raise ValueError("Anthropic API requires max_tokens >= 1")

    def generate(
        self,
        messages: Sequence[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Synchronously generate a response via the Anthropic Messages API.

        Implements exponential backoff on rate-limit errors.

        Parameters
        ----------
        messages:
            Conversation history.
        config:
            Per-call configuration.

        Returns
        -------
        LLMResponse

        Raises
        ------
        LLMRateLimitError
            After exhausting all retries on rate-limit responses.
        LLMConnectionError
            On network or connection failures.
        LLMResponseError
            On unparseable or invalid responses.
        """
        self.validate_config(config)

        model = config.model or self._default_model
        anthropic_messages = self._build_messages(messages)
        system_prompt = config.system_prompt or None

        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                kwargs: dict = {
                    "model": model,
                    "messages": anthropic_messages,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                }

                if system_prompt:
                    kwargs["system"] = system_prompt

                if config.stop_sequences:
                    kwargs["stop_sequences"] = list(config.stop_sequences)

                # Pass through extra parameters
                for key, value in config.extra.items():
                    kwargs[key] = value

                response = self._client.messages.create(**kwargs)

                return self._parse_response(response)

            except _anthropic_module.RateLimitError as exc:
                last_error = exc
                delay = self._base_retry_delay * (2 ** attempt)
                logger.warning(
                    "AnthropicProvider: rate limited (attempt %d/%d), "
                    "retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    delay,
                    exc,
                )
                if attempt < self._max_retries:
                    time.sleep(delay)
                continue

            except _anthropic_module.APIConnectionError as exc:
                raise LLMConnectionError(
                    f"Anthropic API connection failed: {exc}"
                ) from exc

            except _anthropic_module.APIStatusError as exc:
                raise LLMResponseError(
                    f"Anthropic API error (status {exc.status_code}): {exc.message}"
                ) from exc

            except Exception as exc:
                raise LLMError(
                    f"Unexpected error calling Anthropic API: {exc}"
                ) from exc

        raise LLMRateLimitError(
            f"Anthropic API rate limit exceeded after {self._max_retries + 1} attempts: "
            f"{last_error}"
        )

    async def generate_async(
        self,
        messages: Sequence[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Asynchronously generate a response via the Anthropic Messages API.

        Same semantics and retry logic as :meth:`generate` but for async
        contexts.
        """
        self.validate_config(config)

        model = config.model or self._default_model
        anthropic_messages = self._build_messages(messages)
        system_prompt = config.system_prompt or None

        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                kwargs: dict = {
                    "model": model,
                    "messages": anthropic_messages,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                }

                if system_prompt:
                    kwargs["system"] = system_prompt

                if config.stop_sequences:
                    kwargs["stop_sequences"] = list(config.stop_sequences)

                for key, value in config.extra.items():
                    kwargs[key] = value

                response = await self._async_client.messages.create(**kwargs)

                return self._parse_response(response)

            except _anthropic_module.RateLimitError as exc:
                last_error = exc
                delay = self._base_retry_delay * (2 ** attempt)
                logger.warning(
                    "AnthropicProvider (async): rate limited (attempt %d/%d), "
                    "retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    delay,
                    exc,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(delay)
                continue

            except _anthropic_module.APIConnectionError as exc:
                raise LLMConnectionError(
                    f"Anthropic API connection failed: {exc}"
                ) from exc

            except _anthropic_module.APIStatusError as exc:
                raise LLMResponseError(
                    f"Anthropic API error (status {exc.status_code}): {exc.message}"
                ) from exc

            except Exception as exc:
                raise LLMError(
                    f"Unexpected error calling Anthropic API: {exc}"
                ) from exc

        raise LLMRateLimitError(
            f"Anthropic API rate limit exceeded after {self._max_retries + 1} attempts: "
            f"{last_error}"
        )

    # -- internal helpers -----------------------------------------------------

    def _build_messages(
        self,
        messages: Sequence[LLMMessage],
    ) -> list[dict[str, str]]:
        """Convert LLMMessage sequence to Anthropic messages format.

        System messages are filtered out (they go in the ``system``
        parameter instead).
        """
        result: list[dict[str, str]] = []
        for msg in messages:
            if msg.role == "system":
                # System messages handled separately via the system parameter
                continue
            result.append({
                "role": msg.role,
                "content": msg.content,
            })
        return result

    def _parse_response(self, response: object) -> LLMResponse:
        """Parse an Anthropic API response into an LLMResponse."""
        try:
            # Extract text from content blocks
            text_parts: list[str] = []
            for block in response.content:  # type: ignore[attr-defined]
                if hasattr(block, "text"):
                    text_parts.append(block.text)

            text = "".join(text_parts)

            usage = {}
            if hasattr(response, "usage") and response.usage is not None:  # type: ignore[attr-defined]
                usage_obj = response.usage  # type: ignore[attr-defined]
                if hasattr(usage_obj, "input_tokens"):
                    usage["input_tokens"] = usage_obj.input_tokens
                if hasattr(usage_obj, "output_tokens"):
                    usage["output_tokens"] = usage_obj.output_tokens

            return LLMResponse(
                text=text,
                model=getattr(response, "model", ""),  # type: ignore[attr-defined]
                usage=usage,
                finish_reason=getattr(response, "stop_reason", ""),  # type: ignore[attr-defined]
                raw={"id": getattr(response, "id", "")},  # type: ignore[attr-defined]
            )

        except Exception as exc:
            raise LLMResponseError(
                f"Failed to parse Anthropic response: {exc}"
            ) from exc

    def __repr__(self) -> str:
        return (
            f"AnthropicProvider("
            f"model={self._default_model!r}, "
            f"max_retries={self._max_retries})"
        )
