"""Stream event formatters for the teleological graph.

Utilities for processing the stream of events produced by ``.stream()``
and bridging them to the measurement layer (AgentLog, MetricsEngine).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any


def format_stream_events(
    stream: Iterator[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Convert LangGraph stream chunks into flat teleological event dicts.

    Each yielded dict has at minimum:
    - ``node``: the graph node name
    - ``timestamp``: when the event was processed
    - Plus all fields from the node's state update

    Parameters
    ----------
    stream:
        The iterator from ``graph.stream(initial_state, stream_mode="updates")``.
        Each chunk is a dict mapping node name to its state update.

    Yields
    ------
    dict[str, Any]
        Formatted event dicts.
    """
    for chunk in stream:
        for node_name, state_update in chunk.items():
            event: dict[str, Any] = {
                "node": node_name,
                "timestamp": time.time(),
            }
            if isinstance(state_update, dict):
                event.update(state_update)
            yield event


def collect_stream_events(
    stream: Iterator[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect all stream events into a list.

    Convenience wrapper around :func:`format_stream_events`.
    """
    return list(format_stream_events(stream))


def stream_to_agent_log_entries(
    stream: Iterator[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert stream events into entries compatible with measurement AgentLog.

    Extracts step-level data suitable for building an ``AgentLog`` for
    the MetricsEngine.

    Parameters
    ----------
    stream:
        The stream from ``.stream()``.

    Returns
    -------
    list[dict]
        One entry per relevant node event, containing:
        - step, eval_score, action_name, goal_id, timestamp
    """
    entries: list[dict[str, Any]] = []

    for chunk in stream:
        for node_name, state_update in chunk.items():
            if not isinstance(state_update, dict):
                continue

            if node_name == "reflect":
                events = state_update.get("events", [])
                for ev in events:
                    if isinstance(ev, dict) and ev.get("type") == "reflection":
                        entries.append({
                            "step": ev.get("step", 0),
                            "eval_score": ev.get("eval_score", 0.0),
                            "stop_reason": ev.get("stop_reason"),
                            "timestamp": ev.get("timestamp", time.time()),
                        })
            elif node_name == "act":
                events = state_update.get("events", [])
                for ev in events:
                    if isinstance(ev, dict) and ev.get("type") == "action_executed":
                        entries.append({
                            "step": ev.get("step", 0),
                            "action_name": ev.get("action_name"),
                            "timestamp": ev.get("timestamp", time.time()),
                        })

    return entries
