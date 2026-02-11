"""Provider router for the Synthetic Teleology LLM layer.

Implements the **Strategy + Fallback Chain** pattern: routes LLM requests
to the highest-priority healthy provider, automatically falling back to
lower-priority alternatives on failure.

Usage::

    router = ProviderRouter([
        (anthropic_provider, 10),
        (openai_provider, 5),
        (local_provider, 1),
    ])
    response = router.generate(messages, config)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from synthetic_teleology.infrastructure.llm import (
    LLMConfig,
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMResponse,
)

logger = logging.getLogger(__name__)


class _ProviderEntry:
    """Internal entry tracking a provider's priority and health status."""

    __slots__ = (
        "provider",
        "priority",
        "healthy",
        "consecutive_failures",
        "last_failure_time",
        "total_requests",
        "total_failures",
    )

    def __init__(self, provider: LLMProvider, priority: int) -> None:
        self.provider = provider
        self.priority = priority
        self.healthy: bool = True
        self.consecutive_failures: int = 0
        self.last_failure_time: float = 0.0
        self.total_requests: int = 0
        self.total_failures: int = 0

    def record_success(self) -> None:
        """Mark the provider as healthy after a successful request."""
        self.healthy = True
        self.consecutive_failures = 0
        self.total_requests += 1

    def record_failure(self) -> None:
        """Record a failure and potentially mark the provider as unhealthy."""
        self.consecutive_failures += 1
        self.total_failures += 1
        self.total_requests += 1
        self.last_failure_time = time.monotonic()

    def __repr__(self) -> str:
        status = "healthy" if self.healthy else "unhealthy"
        return (
            f"ProviderEntry(name={self.provider.provider_name!r}, "
            f"priority={self.priority}, status={status}, "
            f"failures={self.consecutive_failures})"
        )


class ProviderRouter(LLMProvider):
    """Strategy + Fallback Chain router for multiple LLM providers.

    Routes requests to providers in priority order (highest first).
    On failure, automatically falls back to the next healthy provider.
    Providers that exceed the failure threshold are temporarily marked
    unhealthy and excluded from routing until the cooldown period expires.

    Parameters
    ----------
    providers:
        List of ``(provider, priority)`` tuples.  Higher priority values
        are tried first.
    failure_threshold:
        Number of consecutive failures before a provider is marked
        unhealthy.  Defaults to 3.
    cooldown_seconds:
        How long an unhealthy provider is excluded before retrying.
        Defaults to 60.0.

    Raises
    ------
    ValueError
        If no providers are given.
    """

    def __init__(
        self,
        providers: Sequence[tuple[LLMProvider, int]],
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
    ) -> None:
        if not providers:
            raise ValueError("ProviderRouter requires at least one provider")

        self._entries: list[_ProviderEntry] = [
            _ProviderEntry(provider, priority)
            for provider, priority in providers
        ]
        # Sort by priority descending (highest first)
        self._entries.sort(key=lambda e: e.priority, reverse=True)

        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds

    @property
    def provider_name(self) -> str:
        """Return ``'router'``."""
        return "router"

    @property
    def providers(self) -> list[tuple[LLMProvider, int, bool]]:
        """Return a list of ``(provider, priority, is_healthy)`` tuples."""
        return [
            (e.provider, e.priority, e.healthy)
            for e in self._entries
        ]

    @property
    def num_providers(self) -> int:
        """Total number of registered providers."""
        return len(self._entries)

    @property
    def num_healthy(self) -> int:
        """Number of currently healthy providers."""
        self._check_cooldowns()
        return sum(1 for e in self._entries if e.healthy)

    def generate(
        self,
        messages: Sequence[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Route the request to the highest-priority healthy provider.

        Falls back to lower-priority providers on failure.

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
        LLMError
            If all providers fail.
        """
        self._check_cooldowns()

        errors: list[tuple[str, Exception]] = []

        for entry in self._entries:
            if not entry.healthy:
                continue

            try:
                logger.debug(
                    "ProviderRouter: trying %s (priority=%d)",
                    entry.provider.provider_name,
                    entry.priority,
                )

                response = entry.provider.generate(messages, config)
                entry.record_success()

                logger.debug(
                    "ProviderRouter: %s succeeded",
                    entry.provider.provider_name,
                )
                return response

            except Exception as exc:
                entry.record_failure()
                errors.append((entry.provider.provider_name, exc))

                logger.warning(
                    "ProviderRouter: %s failed (consecutive=%d): %s",
                    entry.provider.provider_name,
                    entry.consecutive_failures,
                    exc,
                )

                # Mark unhealthy if threshold exceeded
                if entry.consecutive_failures >= self._failure_threshold:
                    entry.healthy = False
                    logger.warning(
                        "ProviderRouter: %s marked unhealthy after %d consecutive failures",
                        entry.provider.provider_name,
                        entry.consecutive_failures,
                    )

        # All providers failed
        error_summary = "; ".join(
            f"{name}: {exc}" for name, exc in errors
        )
        raise LLMError(
            f"All {len(self._entries)} providers failed. Errors: {error_summary}"
        )

    async def generate_async(
        self,
        messages: Sequence[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Asynchronously route the request to the highest-priority healthy provider.

        Same fallback logic as :meth:`generate` but uses async provider methods.
        """
        self._check_cooldowns()

        errors: list[tuple[str, Exception]] = []

        for entry in self._entries:
            if not entry.healthy:
                continue

            try:
                logger.debug(
                    "ProviderRouter (async): trying %s (priority=%d)",
                    entry.provider.provider_name,
                    entry.priority,
                )

                response = await entry.provider.generate_async(messages, config)
                entry.record_success()

                logger.debug(
                    "ProviderRouter (async): %s succeeded",
                    entry.provider.provider_name,
                )
                return response

            except Exception as exc:
                entry.record_failure()
                errors.append((entry.provider.provider_name, exc))

                logger.warning(
                    "ProviderRouter (async): %s failed (consecutive=%d): %s",
                    entry.provider.provider_name,
                    entry.consecutive_failures,
                    exc,
                )

                if entry.consecutive_failures >= self._failure_threshold:
                    entry.healthy = False
                    logger.warning(
                        "ProviderRouter (async): %s marked unhealthy after %d "
                        "consecutive failures",
                        entry.provider.provider_name,
                        entry.consecutive_failures,
                    )

        error_summary = "; ".join(
            f"{name}: {exc}" for name, exc in errors
        )
        raise LLMError(
            f"All {len(self._entries)} providers failed. Errors: {error_summary}"
        )

    def health_check(self) -> dict[str, dict]:
        """Check the health status of all providers.

        Returns
        -------
        dict[str, dict]
            Mapping from provider name to a status dictionary containing:
            ``healthy``, ``priority``, ``consecutive_failures``,
            ``total_requests``, ``total_failures``, ``last_failure_time``.
        """
        self._check_cooldowns()
        result: dict[str, dict] = {}

        for entry in self._entries:
            name = entry.provider.provider_name
            # Handle duplicate names by appending priority
            key = name if name not in result else f"{name}_p{entry.priority}"
            result[key] = {
                "healthy": entry.healthy,
                "priority": entry.priority,
                "consecutive_failures": entry.consecutive_failures,
                "total_requests": entry.total_requests,
                "total_failures": entry.total_failures,
                "last_failure_time": entry.last_failure_time,
                "provider_type": type(entry.provider).__name__,
            }

        return result

    def reset_health(self, provider_name: str | None = None) -> int:
        """Reset health status for one or all providers.

        Parameters
        ----------
        provider_name:
            If given, reset only the specified provider.  If ``None``,
            reset all providers.

        Returns
        -------
        int
            Number of providers reset.
        """
        count = 0
        for entry in self._entries:
            if provider_name is None or entry.provider.provider_name == provider_name:
                entry.healthy = True
                entry.consecutive_failures = 0
                entry.last_failure_time = 0.0
                count += 1

        logger.info("ProviderRouter: reset health for %d providers", count)
        return count

    def _check_cooldowns(self) -> None:
        """Re-enable unhealthy providers whose cooldown has expired."""
        now = time.monotonic()
        for entry in self._entries:
            if not entry.healthy and entry.last_failure_time > 0:
                elapsed = now - entry.last_failure_time
                if elapsed >= self._cooldown_seconds:
                    entry.healthy = True
                    entry.consecutive_failures = 0
                    logger.info(
                        "ProviderRouter: %s re-enabled after %.1fs cooldown",
                        entry.provider.provider_name,
                        elapsed,
                    )

    def __repr__(self) -> str:
        provider_info = ", ".join(
            f"{e.provider.provider_name}(p={e.priority}, h={e.healthy})"
            for e in self._entries
        )
        return f"ProviderRouter([{provider_info}])"
