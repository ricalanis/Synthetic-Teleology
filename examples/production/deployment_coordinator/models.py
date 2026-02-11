"""Domain model for the Deployment Coordinator example."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CVEReport:
    """A discovered CVE vulnerability."""

    cve_id: str
    severity: str  # critical | high | medium | low
    component: str
    description: str
    patch_available: bool


@dataclass(frozen=True)
class ServiceHealth:
    """Health status of a single service."""

    service: str
    status: str  # healthy | degraded | down
    latency_ms: float
    error_rate: float


@dataclass
class DeploymentState:
    """Mutable deployment state shared across all agents.

    All agent tools read and mutate this single instance to simulate
    a shared deployment environment.
    """

    stage: str = "planning"  # planning | canary | staging | production
    services: list[ServiceHealth] = field(default_factory=list)
    cve_reports: list[CVEReport] = field(default_factory=list)
    canary_passed: bool = False
    staging_passed: bool = False
    rollback_count: int = 0
    patches_applied: int = 0
    sla_violations: int = 0
    deployment_progress: float = 0.0  # 0.0 - 1.0
    cve_discovered: bool = False

    # Internal counter for tool call sequencing
    _scan_call_count: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if not self.services:
            self.services = [
                ServiceHealth(
                    service="api-gateway",
                    status="healthy",
                    latency_ms=45.0,
                    error_rate=0.001,
                ),
                ServiceHealth(
                    service="user-service",
                    status="healthy",
                    latency_ms=32.0,
                    error_rate=0.0005,
                ),
                ServiceHealth(
                    service="payment-service",
                    status="healthy",
                    latency_ms=58.0,
                    error_rate=0.002,
                ),
            ]
