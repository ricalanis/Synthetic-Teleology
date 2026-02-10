"""Component registry for the Synthetic Teleology framework.

Provides a service-locator / component-registry that lets evaluators, updaters,
planners, environments, and any other pluggable piece register themselves by
*category* and *name* -- either via the ``@registry.register(...)`` decorator
or the imperative ``registry.register_instance(...)`` API.

A **global singleton** ``registry`` is provided for convenience; most of the
framework wires through it.  Tests or isolated runs can instantiate their own
``ComponentRegistry`` to avoid cross-contamination.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ComponentRegistry:
    """Hierarchical service locator keyed by ``(category, name)`` pairs.

    Categories are arbitrary strings (e.g. ``"evaluator"``, ``"planner"``,
    ``"environment"``).  Names are unique **within** a category.

    Usage -- decorator style::

        @registry.register("evaluator", "numeric")
        class NumericEvaluator:
            ...

    Usage -- imperative style::

        registry.register_instance("planner", "greedy", GreedyPlanner())
    """

    def __init__(self) -> None:
        # category -> name -> component (class or instance)
        self._components: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    #  Registration                                                       #
    # ------------------------------------------------------------------ #

    def register(
        self,
        category: str,
        name: str,
        *,
        overwrite: bool = False,
    ) -> Callable[[type[T]], type[T]]:
        """Decorator that registers the decorated **class** under
        ``(category, name)``.

        Parameters
        ----------
        category:
            Logical group (e.g. ``"evaluator"``).
        name:
            Unique identifier within the category.
        overwrite:
            If ``True``, silently replace an existing registration.
            Otherwise raise ``ValueError`` on duplicates.
        """

        def decorator(cls: type[T]) -> type[T]:
            self._set(category, name, cls, overwrite=overwrite)
            return cls

        return decorator

    def register_instance(
        self,
        category: str,
        name: str,
        instance: Any,
        *,
        overwrite: bool = False,
    ) -> None:
        """Imperatively register a pre-built *instance* (or class)."""
        self._set(category, name, instance, overwrite=overwrite)

    # ------------------------------------------------------------------ #
    #  Lookup                                                              #
    # ------------------------------------------------------------------ #

    def get(self, category: str, name: str) -> Any:
        """Return the component registered under ``(category, name)``.

        Raises ``KeyError`` if not found.
        """
        try:
            return self._components[category][name]
        except KeyError:
            available = self.list_category(category)
            raise KeyError(
                f"Component '{category}/{name}' not registered. "
                f"Available in '{category}': {available}"
            ) from None

    def get_or_none(self, category: str, name: str) -> Any | None:
        """Return the component or ``None`` if not found."""
        return self._components.get(category, {}).get(name)

    def has(self, category: str, name: str) -> bool:
        """Return ``True`` if ``(category, name)`` is registered."""
        return name in self._components.get(category, {})

    # ------------------------------------------------------------------ #
    #  Introspection                                                       #
    # ------------------------------------------------------------------ #

    def list_category(self, category: str) -> list[str]:
        """Return the names registered under *category*."""
        return list(self._components.get(category, {}).keys())

    def list_categories(self) -> list[str]:
        """Return all category names that have at least one registration."""
        return list(self._components.keys())

    def all_entries(self) -> dict[str, dict[str, Any]]:
        """Return a shallow copy of the full registry tree."""
        return {cat: dict(entries) for cat, entries in self._components.items()}

    def count(self, category: str | None = None) -> int:
        """Count registered components.

        If *category* is ``None``, return the total across all categories.
        """
        if category is not None:
            return len(self._components.get(category, {}))
        return sum(len(entries) for entries in self._components.values())

    # ------------------------------------------------------------------ #
    #  Removal / lifecycle                                                 #
    # ------------------------------------------------------------------ #

    def unregister(self, category: str, name: str) -> Any:
        """Remove and return the component. Raises ``KeyError`` if missing."""
        try:
            return self._components[category].pop(name)
        except KeyError:
            raise KeyError(
                f"Cannot unregister '{category}/{name}': not found."
            ) from None

    def clear(self, category: str | None = None) -> None:
        """Clear registrations.

        If *category* is given, clear only that category; otherwise clear
        everything.
        """
        if category is not None:
            self._components.pop(category, None)
        else:
            self._components.clear()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _set(
        self,
        category: str,
        name: str,
        component: Any,
        *,
        overwrite: bool = False,
    ) -> None:
        bucket = self._components.setdefault(category, {})
        if not overwrite and name in bucket:
            raise ValueError(
                f"Component '{category}/{name}' is already registered as "
                f"{bucket[name]!r}. Pass overwrite=True to replace."
            )
        bucket[name] = component
        logger.debug("Registered %s/%s: %r", category, name, component)

    # ------------------------------------------------------------------ #
    #  Dunder helpers                                                      #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        parts = [f"{cat}({len(ns)})" for cat, ns in self._components.items()]
        return f"<ComponentRegistry [{', '.join(parts)}]>"

    def __contains__(self, key: tuple[str, str]) -> bool:
        """Support ``("evaluator", "numeric") in registry``."""
        if not isinstance(key, tuple) or len(key) != 2:
            return False
        category, name = key
        return self.has(category, name)


# ===================================================================== #
#  Global singleton                                                      #
# ===================================================================== #

registry = ComponentRegistry()
"""Module-level global registry.  Import and use directly::

    from synthetic_teleology.infrastructure.registry import registry

    @registry.register("evaluator", "numeric")
    class NumericEvaluator:
        ...
"""
