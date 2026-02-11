"""Generic OpenAI-compatible API provider for the Synthetic Teleology LLM layer.

Uses ``httpx`` to make raw HTTP requests to any endpoint that implements
the OpenAI chat completions API format.  This supports providers like
Ollama, LM Studio, vLLM, Together AI, Fireworks, and any other service
that exposes a ``/v1/chat/completions`` endpoint.

Requires the ``httpx`` package (``pip install httpx``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from typing import Any

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

try:
    import httpx as _httpx

    _HAS_HTTPX = True
except ImportError:
    _httpx = None  # type: ignore[assignment]
    _HAS_HTTPX = False


def _check_httpx_available() -> None:
    """Raise a clear error if httpx is not installed."""
    if not _HAS_HTTPX:
        raise ImportError(
            "The 'httpx' package is required for GenericOpenAPIProvider. "
            "Install it with: pip install httpx"
        )


class GenericOpenAPIProvider(LLMProvider):
    """LLM provider for any OpenAI-compatible HTTP API endpoint.

    Constructs raw HTTP POST requests to ``{base_url}/chat/completions``
    (or a custom path) with the OpenAI chat completions request format.

    Parameters
    ----------
    base_url:
        The base URL of the API endpoint (e.g. ``"http://localhost:11434/v1"``
        for Ollama, ``"https://api.together.xyz/v1"`` for Together AI).
    api_key:
        Optional API key.  Sent in the ``Authorization: Bearer`` header.
    default_model:
        Default model identifier.  Defaults to ``"default"``.
    completions_path:
        Path appended to base_url for chat completions.
        Defaults to ``"/chat/completions"``.
    max_retries:
        Maximum number of retries on rate-limit errors.
    base_retry_delay:
        Base delay for exponential backoff.
    timeout:
        Request timeout in seconds.
    extra_headers:
        Additional headers to send with every request.

    Raises
    ------
    ImportError
        If ``httpx`` is not installed.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        default_model: str = "default",
        completions_path: str = "/chat/completions",
        max_retries: int = 3,
        base_retry_delay: float = 1.0,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        _check_httpx_available()

        # Normalize base URL: strip trailing slash
        self._base_url = base_url.rstrip("/")
        self._completions_path = completions_path
        self._api_key = api_key
        self._default_model = default_model
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay
        self._timeout = timeout

        # Build headers
        self._headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            self._headers.update(extra_headers)

        # Create httpx clients
        self._client = _httpx.Client(
            timeout=_httpx.Timeout(timeout),
            headers=self._headers,
        )
        self._async_client = _httpx.AsyncClient(
            timeout=_httpx.Timeout(timeout),
            headers=self._headers,
        )

    @property
    def provider_name(self) -> str:
        """Return ``'openapi'``."""
        return "openapi"

    @property
    def endpoint_url(self) -> str:
        """The full URL for chat completions."""
        return f"{self._base_url}{self._completions_path}"

    def generate(
        self,
        messages: Sequence[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Synchronously generate a response via HTTP POST.

        Parameters
        ----------
        messages:
            Conversation history.
        config:
            Per-call configuration.

        Returns
        -------
        LLMResponse
        """
        self.validate_config(config)

        payload = self._build_payload(messages, config)
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(
                    self.endpoint_url,
                    json=payload,
                )

                if response.status_code == 429:
                    last_error = LLMRateLimitError(
                        f"Rate limited (HTTP 429): {response.text}"
                    )
                    delay = self._base_retry_delay * (2 ** attempt)
                    logger.warning(
                        "GenericOpenAPIProvider: rate limited (attempt %d/%d), "
                        "retrying in %.1fs",
                        attempt + 1,
                        self._max_retries + 1,
                        delay,
                    )
                    if attempt < self._max_retries:
                        time.sleep(delay)
                    continue

                if response.status_code >= 500:
                    raise LLMConnectionError(
                        f"Server error (HTTP {response.status_code}): {response.text}"
                    )

                if response.status_code >= 400:
                    raise LLMResponseError(
                        f"Client error (HTTP {response.status_code}): {response.text}"
                    )

                return self._parse_response(response.json())

            except (LLMRateLimitError, LLMConnectionError, LLMResponseError):
                raise
            except _httpx.ConnectError as exc:
                raise LLMConnectionError(
                    f"Failed to connect to {self.endpoint_url}: {exc}"
                ) from exc
            except _httpx.TimeoutException as exc:
                raise LLMConnectionError(
                    f"Request to {self.endpoint_url} timed out: {exc}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise LLMResponseError(
                    f"Invalid JSON response from {self.endpoint_url}: {exc}"
                ) from exc
            except Exception as exc:
                if isinstance(exc, LLMError):
                    raise
                raise LLMError(
                    f"Unexpected error calling {self.endpoint_url}: {exc}"
                ) from exc

        raise LLMRateLimitError(
            f"Rate limit exceeded after {self._max_retries + 1} attempts: {last_error}"
        )

    async def generate_async(
        self,
        messages: Sequence[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Asynchronously generate a response via HTTP POST."""
        self.validate_config(config)

        payload = self._build_payload(messages, config)
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._async_client.post(
                    self.endpoint_url,
                    json=payload,
                )

                if response.status_code == 429:
                    last_error = LLMRateLimitError(
                        f"Rate limited (HTTP 429): {response.text}"
                    )
                    delay = self._base_retry_delay * (2 ** attempt)
                    logger.warning(
                        "GenericOpenAPIProvider (async): rate limited (attempt %d/%d), "
                        "retrying in %.1fs",
                        attempt + 1,
                        self._max_retries + 1,
                        delay,
                    )
                    if attempt < self._max_retries:
                        await asyncio.sleep(delay)
                    continue

                if response.status_code >= 500:
                    raise LLMConnectionError(
                        f"Server error (HTTP {response.status_code}): {response.text}"
                    )

                if response.status_code >= 400:
                    raise LLMResponseError(
                        f"Client error (HTTP {response.status_code}): {response.text}"
                    )

                return self._parse_response(response.json())

            except (LLMRateLimitError, LLMConnectionError, LLMResponseError):
                raise
            except _httpx.ConnectError as exc:
                raise LLMConnectionError(
                    f"Failed to connect to {self.endpoint_url}: {exc}"
                ) from exc
            except _httpx.TimeoutException as exc:
                raise LLMConnectionError(
                    f"Request to {self.endpoint_url} timed out: {exc}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise LLMResponseError(
                    f"Invalid JSON response from {self.endpoint_url}: {exc}"
                ) from exc
            except Exception as exc:
                if isinstance(exc, LLMError):
                    raise
                raise LLMError(
                    f"Unexpected error calling {self.endpoint_url}: {exc}"
                ) from exc

        raise LLMRateLimitError(
            f"Rate limit exceeded after {self._max_retries + 1} attempts: {last_error}"
        )

    def close(self) -> None:
        """Close the underlying HTTP clients."""
        self._client.close()

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        await self._async_client.aclose()

    # -- internal helpers -----------------------------------------------------

    def _build_payload(
        self,
        messages: Sequence[LLMMessage],
        config: LLMConfig,
    ) -> dict[str, Any]:
        """Build the request payload in OpenAI chat completions format."""
        model = config.model or self._default_model

        openai_messages: list[dict[str, str]] = []

        # Add system prompt if present
        has_system = any(m.role == "system" for m in messages)
        if config.system_prompt and not has_system:
            openai_messages.append({
                "role": "system",
                "content": config.system_prompt,
            })

        for msg in messages:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        payload: dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }

        if config.stop_sequences:
            payload["stop"] = list(config.stop_sequences)

        # Pass through extra parameters
        for key, value in config.extra.items():
            payload[key] = value

        return payload

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parse a JSON response dict into an LLMResponse.

        Expects the standard OpenAI chat completions response format.
        """
        try:
            choices = data.get("choices", [])
            if not choices:
                raise LLMResponseError(
                    f"Response contains no choices: {data}"
                )

            choice = choices[0]
            message = choice.get("message", {})
            text = message.get("content", "") or ""
            finish_reason = choice.get("finish_reason", "") or ""

            usage: dict[str, int] = {}
            usage_data = data.get("usage", {})
            if usage_data:
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    if key in usage_data:
                        usage[key] = int(usage_data[key])

            return LLMResponse(
                text=text,
                model=data.get("model", ""),
                usage=usage,
                finish_reason=finish_reason,
                raw={"id": data.get("id", "")},
            )

        except LLMResponseError:
            raise
        except Exception as exc:
            raise LLMResponseError(
                f"Failed to parse response: {exc}"
            ) from exc

    def __repr__(self) -> str:
        return (
            f"GenericOpenAPIProvider("
            f"base_url={self._base_url!r}, "
            f"model={self._default_model!r})"
        )

    def __del__(self) -> None:
        """Best-effort cleanup of HTTP clients."""
        try:
            self._client.close()
        except Exception:
            pass
