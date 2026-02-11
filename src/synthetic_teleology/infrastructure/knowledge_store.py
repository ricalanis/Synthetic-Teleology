"""KnowledgeStore — metacognitive commons.

Thread-safe shared key-value store with typed entries, tag/source/time
queries, EventBus integration, and serialization support.

Implements the metacognitive commons from Haidemariam (2026) — a shared
memory space where agents can deposit and retrieve learned knowledge.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from synthetic_teleology.domain.values import KnowledgeEntry


class KnowledgeStore:
    """Thread-safe metacognitive knowledge store.

    Parameters
    ----------
    event_bus:
        Optional EventBus for publishing KnowledgeUpdated events.
    """

    def __init__(self, event_bus: Any = None) -> None:
        self._entries: dict[str, KnowledgeEntry] = {}
        self._lock = threading.Lock()
        self._event_bus = event_bus

    def put(
        self,
        key: str,
        value: Any,
        source: str = "",
        tags: tuple[str, ...] = (),
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeEntry:
        """Add or update a knowledge entry.

        Returns the created/updated entry.
        """
        entry = KnowledgeEntry(
            key=key,
            value=value,
            source=source,
            tags=tags,
            timestamp=time.time(),
            confidence=confidence,
            metadata=metadata or {},
        )
        with self._lock:
            self._entries[key] = entry

        if self._event_bus is not None:
            try:
                from synthetic_teleology.domain.events import KnowledgeUpdated
                self._event_bus.publish(
                    KnowledgeUpdated(
                        key=key,
                        source=source,
                        tags=tags,
                    )
                )
            except Exception:
                pass  # best-effort

        return entry

    def get(self, key: str) -> KnowledgeEntry | None:
        """Retrieve an entry by key."""
        with self._lock:
            return self._entries.get(key)

    def query_by_tags(self, *tags: str) -> list[KnowledgeEntry]:
        """Return entries matching ALL specified tags."""
        tag_set = set(tags)
        with self._lock:
            return [
                e for e in self._entries.values()
                if tag_set.issubset(set(e.tags))
            ]

    def query_by_source(self, source: str) -> list[KnowledgeEntry]:
        """Return entries from a specific source."""
        with self._lock:
            return [e for e in self._entries.values() if e.source == source]

    def query_recent(self, max_age_seconds: float) -> list[KnowledgeEntry]:
        """Return entries created within the last *max_age_seconds*."""
        cutoff = time.time() - max_age_seconds
        with self._lock:
            return [e for e in self._entries.values() if e.timestamp >= cutoff]

    def keys(self) -> list[str]:
        """Return all keys."""
        with self._lock:
            return list(self._entries.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    def delete(self, key: str) -> bool:
        """Remove an entry. Returns True if it existed."""
        with self._lock:
            return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._entries.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for persistence."""
        with self._lock:
            result = {}
            for key, entry in self._entries.items():
                result[key] = {
                    "entry_id": entry.entry_id,
                    "key": entry.key,
                    "value": entry.value,
                    "source": entry.source,
                    "tags": list(entry.tags),
                    "timestamp": entry.timestamp,
                    "confidence": entry.confidence,
                    "metadata": dict(entry.metadata),
                }
            return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeStore:
        """Deserialize from a dict."""
        store = cls()
        for _key, d in data.items():
            entry = KnowledgeEntry(
                entry_id=d.get("entry_id", ""),
                key=d.get("key", ""),
                value=d.get("value"),
                source=d.get("source", ""),
                tags=tuple(d.get("tags", ())),
                timestamp=d.get("timestamp", 0.0),
                confidence=d.get("confidence", 1.0),
                metadata=d.get("metadata", {}),
            )
            store._entries[entry.key] = entry
        return store

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> KnowledgeStore:
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))
