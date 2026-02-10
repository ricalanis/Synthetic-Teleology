"""Event collector -- subscribes to domain events and builds structured logs.

The :class:`EventCollector` implements the **Repository pattern** for
measurement data.  It listens to the :class:`EventBus` and accumulates
per-agent :class:`AgentLog` instances that downstream metrics, reports,
and benchmarks consume.

Design notes
~~~~~~~~~~~~
* Events arrive in arbitrary order within a single loop step.  The collector
  buffers partial data in ``_pending_entries`` and flushes when
  :class:`LoopStepCompleted` is received.
* The collector is purely passive -- it never publishes events itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from synthetic_teleology.domain.events import (
    ActionExecuted,
    ConstraintViolated,
    DomainEvent,
    EvaluationCompleted,
    GoalRevised,
    LoopStepCompleted,
    ReflectionTriggered,
)
from synthetic_teleology.infrastructure.event_bus import EventBus


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentLogEntry:
    """Single timestep log entry capturing everything that happened in one
    iteration of the teleological loop."""

    step: int
    timestamp: float
    goal_id: str = ""
    eval_score: float = 0.0
    eval_confidence: float = 1.0
    action_name: str = ""
    action_cost: float = 0.0
    state_values: tuple[float, ...] = ()
    goal_revised: bool = False
    constraint_violated: bool = False
    reflection_triggered: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentLog:
    """Complete log for one agent's run.

    Stores both the sequential timestep entries and dedicated lists of
    significant domain events (revisions, constraint violations) for quick
    lookup by metric implementations.
    """

    agent_id: str
    entries: list[AgentLogEntry] = field(default_factory=list)
    goal_revisions: list[GoalRevised] = field(default_factory=list)
    constraint_violations: list[ConstraintViolated] = field(default_factory=list)

    # -- derived properties ---------------------------------------------------

    @property
    def num_steps(self) -> int:
        """Total number of completed loop steps."""
        return len(self.entries)

    @property
    def revision_count(self) -> int:
        """How many goal revisions occurred."""
        return len(self.goal_revisions)

    def get_scores(self) -> list[float]:
        """Return the evaluation scores for every step."""
        return [e.eval_score for e in self.entries]

    def get_costs(self) -> list[float]:
        """Return the action costs for every step."""
        return [e.action_cost for e in self.entries]

    def get_action_names(self) -> list[str]:
        """Return the action names for every step."""
        return [e.action_name for e in self.entries]

    def steps_where(self, predicate: str) -> list[int]:
        """Return step indices where the named boolean flag is ``True``.

        Parameters
        ----------
        predicate:
            One of ``"goal_revised"``, ``"constraint_violated"``,
            ``"reflection_triggered"``.
        """
        return [
            e.step for e in self.entries if getattr(e, predicate, False)
        ]


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class EventCollector:
    """Repository pattern -- subscribes to the event bus and builds
    :class:`AgentLog` objects in real time.

    Parameters
    ----------
    event_bus:
        The synchronous :class:`EventBus` to subscribe to.

    Usage::

        bus = EventBus()
        collector = EventCollector(bus)
        # ... run the teleological loop ...
        log = collector.get_log("agent-1")
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._logs: dict[str, AgentLog] = {}
        self._current_step: dict[str, int] = {}
        self._pending_entries: dict[str, AgentLogEntry] = {}
        self._subscribe()

    # -- subscription ---------------------------------------------------------

    def _subscribe(self) -> None:
        """Wire up all event handlers."""
        self._event_bus.subscribe(LoopStepCompleted, self._on_step_completed)
        self._event_bus.subscribe(EvaluationCompleted, self._on_evaluation)
        self._event_bus.subscribe(GoalRevised, self._on_goal_revised)
        self._event_bus.subscribe(ActionExecuted, self._on_action)
        self._event_bus.subscribe(ConstraintViolated, self._on_constraint_violated)
        self._event_bus.subscribe(ReflectionTriggered, self._on_reflection)

    # -- internal helpers -----------------------------------------------------

    def _ensure_log(self, agent_id: str) -> AgentLog:
        """Get or create the AgentLog for *agent_id*."""
        if agent_id not in self._logs:
            self._logs[agent_id] = AgentLog(agent_id=agent_id)
            self._current_step[agent_id] = 0
        return self._logs[agent_id]

    def _ensure_pending(self, agent_id: str) -> AgentLogEntry:
        """Get or create the pending (in-progress) entry for *agent_id*."""
        if agent_id not in self._pending_entries:
            step = self._current_step.get(agent_id, 0)
            self._pending_entries[agent_id] = AgentLogEntry(
                step=step, timestamp=time.time()
            )
        return self._pending_entries[agent_id]

    # -- event handlers -------------------------------------------------------

    def _on_step_completed(self, event: DomainEvent) -> None:
        """Flush the pending entry when a loop step completes."""
        assert isinstance(event, LoopStepCompleted)
        agent_id = event.source_id
        log = self._ensure_log(agent_id)

        # Pop the pending entry or create a fresh one if none was buffered
        entry = self._pending_entries.pop(
            agent_id,
            AgentLogEntry(
                step=self._current_step.get(agent_id, 0),
                timestamp=event.timestamp,
            ),
        )

        # Overlay authoritative fields from the LoopStepCompleted event
        entry.step = event.step_number
        entry.goal_id = event.goal_id
        entry.timestamp = event.timestamp

        log.entries.append(entry)
        self._current_step[agent_id] = event.step_number + 1

    def _on_evaluation(self, event: DomainEvent) -> None:
        """Capture evaluation score and confidence."""
        assert isinstance(event, EvaluationCompleted)
        agent_id = event.source_id
        self._ensure_log(agent_id)
        entry = self._ensure_pending(agent_id)

        if event.eval_signal is not None:
            entry.eval_score = event.eval_signal.score
            entry.eval_confidence = event.eval_signal.confidence

    def _on_goal_revised(self, event: DomainEvent) -> None:
        """Record goal revision event."""
        assert isinstance(event, GoalRevised)
        agent_id = event.source_id
        log = self._ensure_log(agent_id)
        log.goal_revisions.append(event)

        entry = self._ensure_pending(agent_id)
        entry.goal_revised = True

    def _on_action(self, event: DomainEvent) -> None:
        """Capture action name and cost."""
        assert isinstance(event, ActionExecuted)
        agent_id = event.source_id
        self._ensure_log(agent_id)
        entry = self._ensure_pending(agent_id)

        if event.action is not None:
            entry.action_name = event.action.name
            entry.action_cost = event.action.cost

    def _on_constraint_violated(self, event: DomainEvent) -> None:
        """Record constraint violation."""
        assert isinstance(event, ConstraintViolated)
        agent_id = event.source_id
        log = self._ensure_log(agent_id)
        log.constraint_violations.append(event)

        entry = self._ensure_pending(agent_id)
        entry.constraint_violated = True

    def _on_reflection(self, event: DomainEvent) -> None:
        """Record that a reflection phase occurred."""
        assert isinstance(event, ReflectionTriggered)
        agent_id = event.source_id
        self._ensure_log(agent_id)
        entry = self._ensure_pending(agent_id)
        entry.reflection_triggered = True

    # -- public query API -----------------------------------------------------

    def get_log(self, agent_id: str) -> AgentLog | None:
        """Return the :class:`AgentLog` for *agent_id*, or ``None``."""
        return self._logs.get(agent_id)

    def get_all_logs(self) -> dict[str, AgentLog]:
        """Return a shallow copy of all collected logs keyed by agent id."""
        return dict(self._logs)

    def agent_ids(self) -> list[str]:
        """Return a list of all agent ids that have at least one log entry."""
        return list(self._logs.keys())

    def clear(self) -> None:
        """Reset all collected data."""
        self._logs.clear()
        self._current_step.clear()
        self._pending_entries.clear()
