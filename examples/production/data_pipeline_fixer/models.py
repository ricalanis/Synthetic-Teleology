"""Domain model for the Data Pipeline Fixer agent."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SchemaVersion:
    """An immutable record of a schema version."""

    version: str
    fields: dict[str, str] = field(default_factory=dict)
    breaking_changes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FixAttempt:
    """An immutable record of a single fix attempt."""

    fix_type: str
    target: str
    success: bool
    timestamp: float
    details: str = ""


@dataclass
class PipelineState:
    """Mutable state of the data pipeline being monitored.

    health_score ranges [0, 1] where 1.0 is fully healthy.
    error_rate ranges [0, 1] where 0.0 is no errors.
    throughput is records/sec.
    """

    health_score: float = 0.92
    error_rate: float = 0.03
    throughput: float = 1000.0
    schema_version: str = "v1.0"
    schema_drift_detected: bool = False
    tables: list[str] = field(default_factory=lambda: [
        "users", "orders", "products", "events", "sessions",
        "payments", "inventory", "analytics",
    ])
    fix_history: list[FixAttempt] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fix_attempts_count: int = 0
