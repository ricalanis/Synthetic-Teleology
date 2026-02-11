"""Goal Audit Trail — serializable revision history.

Provides a queryable, serializable log of all goal revisions, enabling
goal versioning persistence per Haidemariam (2026).
"""

from __future__ import annotations

import json
import time
from typing import Any

from synthetic_teleology.domain.values import GoalAuditEntry, GoalProvenance
from synthetic_teleology.domain.enums import GoalOrigin


class GoalAuditTrail:
    """Serializable audit trail for goal revisions.

    Thread-safe (append-only design).  Integrates with KnowledgeStore
    if provided.

    Parameters
    ----------
    knowledge_store:
        Optional KnowledgeStore for cross-referencing entries.
    """

    def __init__(self, knowledge_store: Any = None) -> None:
        self._entries: list[GoalAuditEntry] = []
        self._knowledge_store = knowledge_store

    def record(
        self,
        goal_id: str,
        previous_goal_id: str = "",
        revision_reason: str = "",
        eval_score: float = 0.0,
        eval_confidence: float = 1.0,
        provenance: GoalProvenance | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GoalAuditEntry:
        """Record a goal revision event.

        Returns the created entry.
        """
        entry = GoalAuditEntry(
            timestamp=time.time(),
            goal_id=goal_id,
            previous_goal_id=previous_goal_id,
            revision_reason=revision_reason,
            eval_score=eval_score,
            eval_confidence=eval_confidence,
            provenance=provenance,
            metadata=metadata or {},
        )
        self._entries.append(entry)

        # Cross-reference in knowledge store if available
        if self._knowledge_store is not None:
            try:
                self._knowledge_store.put(
                    key=f"audit:{entry.entry_id}",
                    value={"goal_id": goal_id, "reason": revision_reason},
                    source="audit_trail",
                    tags=("audit", "goal_revision"),
                )
            except Exception:
                pass  # best-effort

        return entry

    def query(
        self,
        goal_id: str | None = None,
        reason: str | None = None,
        limit: int = 100,
    ) -> list[GoalAuditEntry]:
        """Query entries by goal_id and/or reason."""
        results = self._entries
        if goal_id is not None:
            results = [e for e in results if e.goal_id == goal_id]
        if reason is not None:
            results = [e for e in results if e.revision_reason == reason]
        return results[:limit]

    @property
    def entries(self) -> list[GoalAuditEntry]:
        """All entries (read-only view)."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialize to a list of dicts."""
        result = []
        for e in self._entries:
            d: dict[str, Any] = {
                "entry_id": e.entry_id,
                "timestamp": e.timestamp,
                "goal_id": e.goal_id,
                "previous_goal_id": e.previous_goal_id,
                "revision_reason": e.revision_reason,
                "eval_score": e.eval_score,
                "eval_confidence": e.eval_confidence,
                "metadata": dict(e.metadata),
            }
            if e.provenance is not None:
                d["provenance"] = {
                    "origin": e.provenance.origin.value,
                    "source_agent_id": e.provenance.source_agent_id,
                    "source_description": e.provenance.source_description,
                    "timestamp": e.provenance.timestamp,
                }
            result.append(d)
        return result

    @classmethod
    def from_dict(cls, data: list[dict[str, Any]]) -> GoalAuditTrail:
        """Deserialize from a list of dicts."""
        trail = cls()
        for d in data:
            prov = None
            if "provenance" in d:
                pd = d["provenance"]
                prov = GoalProvenance(
                    origin=GoalOrigin(pd.get("origin", "design")),
                    source_agent_id=pd.get("source_agent_id", ""),
                    source_description=pd.get("source_description", ""),
                    timestamp=pd.get("timestamp", 0.0),
                )
            entry = GoalAuditEntry(
                entry_id=d.get("entry_id", ""),
                timestamp=d.get("timestamp", 0.0),
                goal_id=d.get("goal_id", ""),
                previous_goal_id=d.get("previous_goal_id", ""),
                revision_reason=d.get("revision_reason", ""),
                eval_score=d.get("eval_score", 0.0),
                eval_confidence=d.get("eval_confidence", 1.0),
                provenance=prov,
                metadata=d.get("metadata", {}),
            )
            trail._entries.append(entry)
        return trail

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> GoalAuditTrail:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
