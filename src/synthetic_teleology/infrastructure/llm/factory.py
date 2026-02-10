"""LLM provider factory for the Synthetic Teleology framework.

Registry-based factory pattern with auto-discovery of available providers.
Creates concrete :class:`LLMProvider` instances by name, handling optional
dependencies gracefully.

Usage::

    factory = LLMProviderFactory()
    provider = factory.create("anthropic", api_key="sk-...")
    provider = factory.create("openai", api_key="sk-...")
    provider = factory.create("openapi", base_url="http://localhost:11434/v1")
    provider = factory.create("huggingface", model_name="microsoft/phi-2")
    provider = factory.create("router", providers=[...])
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from synthetic_teleology.infrastructure.llm import LLMProvider

logger = logging.getLogger(__name__)


# Type for provider constructor functions
ProviderConstructor = Callable[..., LLMProvider]


class LLMProviderFactory:
    """Registry-based factory for creating LLM provider instances.

    Maintains a registry mapping provider names to constructor functions.
    Constructors for built-in providers (Anthropic, OpenAI, OpenAPI,
    HuggingFace, Router) are pre-registered.  Custom providers can be
    registered via :meth:`register`.

    Parameters
    ----------
    auto_discover:
        If ``True`` (default), pre-register all built-in providers.
    """

    def __init__(self, auto_discover: bool = True) -> None:
        self._registry: dict[str, ProviderConstructor] = {}

        if auto_discover:
            self._discover_builtin_providers()

    # -- registration ---------------------------------------------------------

    def register(
        self,
        name: str,
        constructor: ProviderConstructor,
        overwrite: bool = False,
    ) -> None:
        """Register a provider constructor under the given *name*.

        Parameters
        ----------
        name:
            The provider name (e.g. ``"my_custom_provider"``).
        constructor:
            A callable that takes keyword arguments and returns an
            :class:`LLMProvider` instance.
        overwrite:
            If ``False`` (default), raises ``ValueError`` when *name*
            is already registered.  If ``True``, silently replaces.

        Raises
        ------
        ValueError
            If the name is already registered and ``overwrite`` is ``False``.
        """
        if name in self._registry and not overwrite:
            raise ValueError(
                f"Provider {name!r} is already registered. "
                f"Use overwrite=True to replace it."
            )
        self._registry[name] = constructor
        logger.debug("LLMProviderFactory: registered provider %r", name)

    def unregister(self, name: str) -> bool:
        """Remove a provider from the registry.

        Returns ``True`` if the provider was found and removed.
        """
        if name in self._registry:
            del self._registry[name]
            return True
        return False

    # -- creation -------------------------------------------------------------

    def create(self, provider_name: str, **kwargs: Any) -> LLMProvider:
        """Create an LLM provider instance by name.

        Parameters
        ----------
        provider_name:
            The registered name of the provider to create.
        **kwargs:
            Arguments passed to the provider's constructor.

        Returns
        -------
        LLMProvider
            A configured provider instance.

        Raises
        ------
        ValueError
            If the provider name is not registered.
        ImportError
            If the provider's required dependencies are not installed.
        """
        constructor = self._registry.get(provider_name)
        if constructor is None:
            available = ", ".join(sorted(self._registry.keys()))
            raise ValueError(
                f"Unknown provider {provider_name!r}. "
                f"Available providers: {available}"
            )

        logger.info(
            "LLMProviderFactory: creating provider %r with kwargs %s",
            provider_name,
            list(kwargs.keys()),
        )

        return constructor(**kwargs)

    # -- query ----------------------------------------------------------------

    @property
    def registered_providers(self) -> list[str]:
        """Return a sorted list of registered provider names."""
        return sorted(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Check whether a provider name is registered."""
        return name in self._registry

    def available_providers(self) -> dict[str, bool]:
        """Check which providers can actually be instantiated.

        Tests each registered provider by checking if its required
        dependencies are importable.

        Returns
        -------
        dict[str, bool]
            Mapping from provider name to availability.
        """
        result: dict[str, bool] = {}
        for name in self._registry:
            result[name] = self._check_availability(name)
        return result

    # -- auto-discovery -------------------------------------------------------

    def _discover_builtin_providers(self) -> None:
        """Register constructors for all built-in providers.

        Each provider is wrapped in a lazy constructor that only imports
        the provider module when actually called, so missing dependencies
        do not prevent the factory from being used.
        """
        self._registry["anthropic"] = self._create_anthropic
        self._registry["openai"] = self._create_openai
        self._registry["openapi"] = self._create_openapi
        self._registry["huggingface"] = self._create_huggingface
        self._registry["router"] = self._create_router

    @staticmethod
    def _create_anthropic(**kwargs: Any) -> LLMProvider:
        """Lazy constructor for AnthropicProvider."""
        from synthetic_teleology.infrastructure.llm.anthropic import AnthropicProvider

        return AnthropicProvider(**kwargs)

    @staticmethod
    def _create_openai(**kwargs: Any) -> LLMProvider:
        """Lazy constructor for OpenAIProvider."""
        from synthetic_teleology.infrastructure.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(**kwargs)

    @staticmethod
    def _create_openapi(**kwargs: Any) -> LLMProvider:
        """Lazy constructor for GenericOpenAPIProvider."""
        from synthetic_teleology.infrastructure.llm.openapi import GenericOpenAPIProvider

        return GenericOpenAPIProvider(**kwargs)

    @staticmethod
    def _create_huggingface(**kwargs: Any) -> LLMProvider:
        """Lazy constructor for HuggingFaceLocalProvider."""
        from synthetic_teleology.infrastructure.llm.huggingface import HuggingFaceLocalProvider

        return HuggingFaceLocalProvider(**kwargs)

    @staticmethod
    def _create_router(**kwargs: Any) -> LLMProvider:
        """Lazy constructor for ProviderRouter.

        Accepts either ``providers`` as a list of ``(provider, priority)``
        tuples, or ``provider_configs`` as a list of dicts for declarative
        configuration.
        """
        from synthetic_teleology.infrastructure.llm.router import ProviderRouter

        if "providers" in kwargs:
            return ProviderRouter(**kwargs)

        # Support declarative config: list of dicts
        provider_configs = kwargs.pop("provider_configs", None)
        if provider_configs is not None:
            factory = LLMProviderFactory()
            provider_tuples: list[tuple[LLMProvider, int]] = []

            for config_dict in provider_configs:
                name = config_dict.pop("provider_name")
                priority = config_dict.pop("priority", 1)
                provider = factory.create(name, **config_dict)
                provider_tuples.append((provider, priority))

            return ProviderRouter(providers=provider_tuples, **kwargs)

        raise ValueError(
            "ProviderRouter requires either 'providers' (list of tuples) "
            "or 'provider_configs' (list of dicts)"
        )

    @staticmethod
    def _check_availability(name: str) -> bool:
        """Check if a provider's dependencies are available."""
        dependency_map = {
            "anthropic": "anthropic",
            "openai": "openai",
            "openapi": "httpx",
            "huggingface": "transformers",
            "router": None,  # No external dependency
        }

        module_name = dependency_map.get(name)
        if module_name is None:
            return True

        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    # -- dunder helpers -------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LLMProviderFactory(providers={sorted(self._registry.keys())})"
        )

    def __contains__(self, name: str) -> bool:
        return name in self._registry
