"""Per-agent tools for the Deployment Coordinator.

All tools share and mutate the same ``DeploymentState`` instance so that
one agent's actions are visible to others through the shared environment.
"""

from .models import CVEReport, DeploymentState, ServiceHealth

# ---------------------------------------------------------------------------
# Release Agent Tools
# ---------------------------------------------------------------------------


def make_run_canary_tool(state: DeploymentState):
    """Create a canary deployment tool bound to the shared state."""

    def run_canary(service: str) -> str:
        """Run canary deployment for a service."""
        if state.stage == "planning":
            state.stage = "canary"
        state.deployment_progress = min(1.0, state.deployment_progress + 0.15)

        # Check for blocking CVEs
        critical_cves = [c for c in state.cve_reports if c.severity == "critical"]
        if critical_cves:
            return (
                f"Canary for {service}: BLOCKED by {len(critical_cves)} critical CVE(s). "
                f"Progress: {state.deployment_progress:.0%}"
            )

        state.canary_passed = True
        return (
            f"Canary for {service}: PASSED. "
            f"Deployment progress: {state.deployment_progress:.0%}"
        )

    return run_canary


def make_promote_stage_tool(state: DeploymentState):
    """Create a stage promotion tool bound to the shared state."""

    def promote_stage(target_stage: str) -> str:
        """Promote deployment to target stage (staging or production)."""
        # Check for blocking CVEs
        critical_cves = [c for c in state.cve_reports if c.severity == "critical"]
        if critical_cves and target_stage == "production":
            return (
                f"Promotion to {target_stage}: BLOCKED by {len(critical_cves)} "
                f"unpatched critical CVE(s). Resolve before promoting."
            )

        if target_stage == "staging":
            state.stage = "staging"
            state.staging_passed = True
            state.deployment_progress = min(1.0, state.deployment_progress + 0.25)
            return (
                f"Promoted to staging. Staging validation: PASSED. "
                f"Progress: {state.deployment_progress:.0%}"
            )

        if target_stage == "production":
            state.stage = "production"
            state.deployment_progress = 1.0
            return (
                f"Promoted to production. Deployment COMPLETE. "
                f"Progress: {state.deployment_progress:.0%}"
            )

        return f"Unknown target stage: {target_stage}"

    return promote_stage


# ---------------------------------------------------------------------------
# Security Agent Tools
# ---------------------------------------------------------------------------


def make_scan_vulnerabilities_tool(state: DeploymentState):
    """Create a vulnerability scanner bound to the shared state.

    On the 3rd+ call (simulating round 2), discovers a critical CVE.
    """

    def scan_vulnerabilities(target: str) -> str:
        """Scan a component for vulnerabilities."""
        state._scan_call_count += 1

        if state._scan_call_count >= 3 and not state.cve_discovered:
            # Discover a critical CVE on scan call 3+
            cve = CVEReport(
                cve_id="CVE-2024-31337",
                severity="critical",
                component="api-gateway",
                description=(
                    "Remote code execution in HTTP/2 header parsing. "
                    "Allows unauthenticated attackers to execute arbitrary code."
                ),
                patch_available=True,
            )
            state.cve_reports.append(cve)
            state.cve_discovered = True

            # Degrade the affected service
            state.services = [
                ServiceHealth(
                    service=s.service,
                    status="degraded" if s.service == "api-gateway" else s.status,
                    latency_ms=s.latency_ms + (20.0 if s.service == "api-gateway" else 0),
                    error_rate=s.error_rate + (0.01 if s.service == "api-gateway" else 0),
                )
                for s in state.services
            ]

            return (
                f"CRITICAL: Scan of {target} found CVE-2024-31337 "
                f"(RCE in api-gateway HTTP/2 parsing). Severity: CRITICAL. "
                f"Patch available. Immediate action required."
            )

        # Clean scan
        open_cves = len(state.cve_reports)
        if open_cves > 0:
            return (
                f"Scan of {target}: {open_cves} known CVE(s) pending. "
                f"No new vulnerabilities found."
            )

        return f"Scan of {target}: CLEAN. No vulnerabilities detected."

    return scan_vulnerabilities


def make_apply_patch_tool(state: DeploymentState):
    """Create a patch application tool bound to the shared state."""

    def apply_patch(cve_id: str) -> str:
        """Apply a security patch for a specific CVE."""
        matching = [c for c in state.cve_reports if c.cve_id == cve_id]
        if not matching:
            return f"CVE {cve_id} not found in reports."

        cve = matching[0]
        if not cve.patch_available:
            return f"No patch available for {cve_id}. Manual mitigation required."

        # Remove the CVE and increment patch count
        state.cve_reports = [c for c in state.cve_reports if c.cve_id != cve_id]
        state.patches_applied += 1

        # Restore service health
        comp = cve.component
        state.services = [
            ServiceHealth(
                service=s.service,
                status="healthy" if s.service == comp else s.status,
                latency_ms=(
                    max(30.0, s.latency_ms - 20.0)
                    if s.service == comp else s.latency_ms
                ),
                error_rate=(
                    max(0.001, s.error_rate - 0.01)
                    if s.service == comp else s.error_rate
                ),
            )
            for s in state.services
        ]

        return (
            f"Patch applied for {cve_id} on {cve.component}. "
            f"Service health restored. Total patches applied: {state.patches_applied}."
        )

    return apply_patch


# ---------------------------------------------------------------------------
# SRE Agent Tools
# ---------------------------------------------------------------------------


def make_check_sla_tool(state: DeploymentState):
    """Create an SLA monitoring tool bound to the shared state."""

    def check_sla(service: str) -> str:
        """Check SLA metrics for a service."""
        matching = [s for s in state.services if s.service == service]
        if not matching:
            return f"Service {service} not found."

        svc = matching[0]
        sla_ok = svc.latency_ms < 200.0 and svc.error_rate < 0.01
        if not sla_ok and svc.status != "healthy":
            state.sla_violations += 1

        return (
            f"SLA check for {service}: "
            f"status={svc.status}, latency={svc.latency_ms:.1f}ms, "
            f"error_rate={svc.error_rate:.4f}. "
            f"SLA {'OK' if sla_ok else 'VIOLATION'}. "
            f"Total violations: {state.sla_violations}."
        )

    return check_sla


def make_scale_capacity_tool(state: DeploymentState):
    """Create a capacity scaling tool bound to the shared state."""

    def scale_capacity(service: str, replicas: int) -> str:
        """Scale a service to the specified replica count."""
        matching = [s for s in state.services if s.service == service]
        if not matching:
            return f"Service {service} not found."

        # Scaling improves latency and error rate
        factor = min(0.5, replicas * 0.05)
        state.services = [
            ServiceHealth(
                service=s.service,
                status="healthy" if s.service == service else s.status,
                latency_ms=(
                    max(20.0, s.latency_ms * (1 - factor))
                    if s.service == service else s.latency_ms
                ),
                error_rate=(
                    max(0.0001, s.error_rate * (1 - factor))
                    if s.service == service else s.error_rate
                ),
            )
            for s in state.services
        ]

        updated = [s for s in state.services if s.service == service][0]
        return (
            f"Scaled {service} to {replicas} replicas. "
            f"New latency: {updated.latency_ms:.1f}ms, "
            f"error_rate: {updated.error_rate:.4f}."
        )

    return scale_capacity
