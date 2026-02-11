"""Working memory for LLM agents without external environments.

``WorkingMemory`` closes the perception-action loop by accumulating
action results and making them visible to ``perceive_node`` on subsequent
iterations.  It provides ``perceive`` and ``record`` callbacks suitable for
``GraphBuilder.with_environment(perceive_fn=..., transition_fn=...)``.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from synthetic_teleology.domain.values import StateSnapshot


class WorkingMemory:
    """Accumulates agent actions as a perception source.

    Usage::

        memory = WorkingMemory("Initial project context: ...")
        app, init = (
            GraphBuilder("agent")
            .with_model(model)
            .with_goal("...")
            .with_environment(
                perceive_fn=memory.perceive,
                transition_fn=memory.record,
            )
            .build()
        )

    Parameters
    ----------
    initial_context:
        Text returned by the first ``perceive()`` call before any actions.
    max_entries:
        FIFO cap on stored entries.
    """

    def __init__(self, initial_context: str = "", max_entries: int = 20) -> None:
        self._initial_context = initial_context
        self._max_entries = max_entries
        self._entries: deque[dict[str, Any]] = deque(maxlen=max_entries)
        self._step = 0

    # -- Callbacks -----------------------------------------------------------

    def perceive(self) -> StateSnapshot:
        """Return a ``StateSnapshot`` with accumulated working memory.

        Suitable for ``perceive_fn``.
        """
        if not self._entries:
            observation = self._initial_context or "No observations yet."
        else:
            lines = [self._initial_context] if self._initial_context else []
            lines.append(f"\nWorking memory ({len(self._entries)} entries):")
            for entry in self._entries:
                action_name = entry.get("action", "?")
                result = str(entry.get("result", ""))[:300]
                lines.append(f"  [step {entry.get('step', '?')}] {action_name}: {result}")
            observation = "\n".join(lines)

        return StateSnapshot(
            timestamp=time.time(),
            observation=observation,
        )

    def record(self, action: Any, constraints_context: Any = None) -> None:
        """Record an executed action into working memory.

        Accepts 1 or 2 arguments so ``act_node``'s ``inspect.signature``
        detection routes correctly for both plain and constraint-conditioned
        transition functions.

        Suitable for ``transition_fn``.
        """
        self._step += 1
        entry: dict[str, Any] = {
            "action": getattr(action, "name", str(action)),
            "result": getattr(action, "description", None) or str(action),
            "step": self._step,
            "timestamp": time.time(),
        }
        if constraints_context is not None:
            entry["constraints_context"] = constraints_context
        self._entries.append(entry)

    # -- Accessors -----------------------------------------------------------

    @property
    def entries(self) -> list[dict[str, Any]]:
        """Return a copy of all stored entries."""
        return list(self._entries)

    @property
    def step(self) -> int:
        """Current step counter."""
        return self._step

    def clear(self) -> None:
        """Reset working memory and step counter."""
        self._entries.clear()
        self._step = 0
