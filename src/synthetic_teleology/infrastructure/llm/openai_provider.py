"""OpenAI provider for the Synthetic Teleology LLM layer.

Wraps the ``openai`` Python SDK to implement the :class:`LLMProvider`
interface.  Handles rate limiting with exponential backoff, connection
errors, and response parsing.

Requires the ``openai`` package (``pip install openai``).  If the
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

# Attempt import at module level
try:
    import openai as _openai_module

    _HAS_OPENAI = True
except ImportError:
    _openai_module = None  # type: ignore[assignment]
    _HAS_OPENAI = False


def _check_openai_available() -> None:
    """Raise a clear error if the openai package is not installed."""
    if not _HAS_OPENAI:
        raise ImportError(
            "The 'openai' package is required for OpenAIProvider. "
            "Install it with: pip install openai"
        )


class OpenAIProvider(LLMProvider):
    """LLM provider for OpenAI models (GPT-4, GPT-4o, o1, etc.).

    Parameters
    ----------
    api_key:
        OpenAI API key.  If ``None``, the ``OPENAI_API_KEY`` environment
        variable is used.
    organization:
        Optional OpenAI organization ID.
    default_model:
        Default model identifier.  Defaults to ``"gpt-4o"``.
    max_retries:
        Maximum number of retries on rate-limit errors.  Defaults to 3.
    base_retry_delay:
        Base delay in seconds for exponential backoff.  Defaults to 1.0.
    timeout:
        Request timeout in seconds.  Defaults to 60.0.
    base_url:
        Optional base URL for the API (for proxies or custom endpoints).

    Raises
    ------
    ImportError
        If the ``openai`` package is not installed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        organization: str | None = None,
        default_model: str = "gpt-4o",
        max_retries: int = 3,
        base_retry_delay: float = 1.0,
        timeout: float = 60.0,
        base_url: str | None = None,
    ) -> None:
        _check_openai_available()

        self._default_model = default_model
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay

        client_kwargs: dict = {"timeout": timeout}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if organization is not None:
            client_kwargs["organization"] = organization
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        self._client = _openai_module.OpenAI(**client_kwargs)
        self._async_client = _openai_module.AsyncOpenAI(**client_kwargs)

    @property
    def provider_name(self) -> str:
        """Return ``'openai'``."""
        return "openai"

    def validate_config(self, config: LLMConfig) -> None:
        """Validate config with OpenAI-specific checks."""
        config.validate()

    def generate(
        self,
        messages: Sequence[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Synchronously generate a response via the OpenAI Chat Completions API.

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
            After exhausting all retries.
        LLMConnectionError
            On network failures.
        LLMResponseError
            On invalid responses.
        """
        self.validate_config(config)

        model = config.model or self._default_model
        openai_messages = self._build_messages(messages, config.system_prompt)

        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                kwargs: dict = {
                    "model": model,
                    "messages": openai_messages,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                }

                if config.stop_sequences:
                    kwargs["stop"] = list(config.stop_sequences)

                # Pass through extra parameters
                for key, value in config.extra.items():
                    kwargs[key] = value

                response = self._client.chat.completions.create(**kwargs)

                return self._parse_response(response)

            except _openai_module.RateLimitError as exc:
                last_error = exc
                delay = self._base_retry_delay * (2 ** attempt)
                logger.warning(
                    "OpenAIProvider: rate limited (attempt %d/%d), "
                    "retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    delay,
                    exc,
                )
                if attempt < self._max_retries:
                    time.sleep(delay)
                continue

            except _openai_module.APIConnectionError as exc:
                raise LLMConnectionError(
                    f"OpenAI API connection failed: {exc}"
                ) from exc

            except _openai_module.APIStatusError as exc:
                raise LLMResponseError(
                    f"OpenAI API error (status {exc.status_code}): {exc.message}"
                ) from exc

            except Exception as exc:
                raise LLMError(
                    f"Unexpected error calling OpenAI API: {exc}"
                ) from exc

        raise LLMRateLimitError(
            f"OpenAI API rate limit exceeded after {self._max_retries + 1} attempts: "
            f"{last_error}"
        )

    async def generate_async(
        self,
        messages: Sequence[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Asynchronously generate a response via the OpenAI Chat Completions API."""
        self.validate_config(config)

        model = config.model or self._default_model
        openai_messages = self._build_messages(messages, config.system_prompt)

        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                kwargs: dict = {
                    "model": model,
                    "messages": openai_messages,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                }

                if config.stop_sequences:
                    kwargs["stop"] = list(config.stop_sequences)

                for key, value in config.extra.items():
                    kwargs[key] = value

                response = await self._async_client.chat.completions.create(**kwargs)

                return self._parse_response(response)

            except _openai_module.RateLimitError as exc:
                last_error = exc
                delay = self._base_retry_delay * (2 ** attempt)
                logger.warning(
                    "OpenAIProvider (async): rate limited (attempt %d/%d), "
                    "retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    delay,
                    exc,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(delay)
                continue

            except _openai_module.APIConnectionError as exc:
                raise LLMConnectionError(
                    f"OpenAI API connection failed: {exc}"
                ) from exc

            except _openai_module.APIStatusError as exc:
                raise LLMResponseError(
                    f"OpenAI API error (status {exc.status_code}): {exc.message}"
                ) from exc

            except Exception as exc:
                raise LLMError(
                    f"Unexpected error calling OpenAI API: {exc}"
                ) from exc

        raise LLMRateLimitError(
            f"OpenAI API rate limit exceeded after {self._max_retries + 1} attempts: "
            f"{last_error}"
        )

    # -- internal helpers -----------------------------------------------------

    def _build_messages(
        self,
        messages: Sequence[LLMMessage],
        system_prompt: str = "",
    ) -> list[dict[str, str]]:
        """Convert LLMMessage sequence to OpenAI chat format.

        If a system_prompt is provided and no system message exists in
        the conversation, it is prepended.
        """
        result: list[dict[str, str]] = []

        # Prepend system prompt if provided and not already present
        has_system = any(m.role == "system" for m in messages)
        if system_prompt and not has_system:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            result.append({"role": msg.role, "content": msg.content})

        return result

    def _parse_response(self, response: object) -> LLMResponse:
        """Parse an OpenAI ChatCompletion response into an LLMResponse."""
        try:
            choices = getattr(response, "choices", [])
            if not choices:
                raise LLMResponseError("OpenAI response contains no choices")

            choice = choices[0]
            message = getattr(choice, "message", None)
            text = getattr(message, "content", "") or "" if message else ""
            finish_reason = getattr(choice, "finish_reason", "") or ""

            usage = {}
            usage_obj = getattr(response, "usage", None)
            if usage_obj is not None:
                if hasattr(usage_obj, "prompt_tokens"):
                    usage["prompt_tokens"] = usage_obj.prompt_tokens
                if hasattr(usage_obj, "completion_tokens"):
                    usage["completion_tokens"] = usage_obj.completion_tokens
                if hasattr(usage_obj, "total_tokens"):
                    usage["total_tokens"] = usage_obj.total_tokens

            return LLMResponse(
                text=text,
                model=getattr(response, "model", ""),
                usage=usage,
                finish_reason=finish_reason,
                raw={"id": getattr(response, "id", "")},
            )

        except LLMResponseError:
            raise
        except Exception as exc:
            raise LLMResponseError(
                f"Failed to parse OpenAI response: {exc}"
            ) from exc

    def __repr__(self) -> str:
        return (
            f"OpenAIProvider("
            f"model={self._default_model!r}, "
            f"max_retries={self._max_retries})"
        )
