"""Per-agent perception strategies for the Deployment Coordinator.

Each agent sees the same ``DeploymentState`` but through a different lens.
The perception functions return domain-specific observations that feed into
the LLM evaluator and planner via ``WorkingMemory``.

In multi-agent LLM mode, the model handles evaluation and planning
directly (``AgentConfig`` does not support custom evaluators).  The
strategy layer focuses on *how* each agent perceives the shared state.
"""

from .models import DeploymentState


def release_observation(state: DeploymentState) -> str:
    """Format deployment state from the Release agent's perspective."""
    stages_map = {"planning": 0, "canary": 1, "staging": 2, "production": 3}
    current_stage_num = stages_map.get(state.stage, 0)

    blockers = []
    critical_cves = [c for c in state.cve_reports if c.severity == "critical"]
    if critical_cves:
        blockers.append(f"{len(critical_cves)} critical CVE(s) blocking production")
    if not state.canary_passed and current_stage_num < 1:
        blockers.append("Canary deployment not yet started")

    return (
        f"Deployment status for release v2.5:\n"
        f"  Stage: {state.stage} ({current_stage_num}/3)\n"
        f"  Progress: {state.deployment_progress:.0%}\n"
        f"  Canary passed: {state.canary_passed}\n"
        f"  Staging passed: {state.staging_passed}\n"
        f"  Rollbacks: {state.rollback_count}\n"
        f"  Blockers: {', '.join(blockers) if blockers else 'none'}\n"
        f"  Services: {', '.join(s.service + '=' + s.status for s in state.services)}"
    )


def security_observation(state: DeploymentState) -> str:
    """Format deployment state from the Security agent's perspective."""
    cve_lines = []
    for cve in state.cve_reports:
        cve_lines.append(
            f"  - {cve.cve_id} [{cve.severity.upper()}] in {cve.component}: "
            f"{cve.description[:80]}... (patch={'yes' if cve.patch_available else 'no'})"
        )

    return (
        f"Security posture:\n"
        f"  Open CVEs: {len(state.cve_reports)}\n"
        f"  Patches applied: {state.patches_applied}\n"
        f"  CVE discovered this cycle: {state.cve_discovered}\n"
        f"  Deployment stage: {state.stage}\n"
        f"  CVE details:\n"
        + ("\n".join(cve_lines) if cve_lines else "    (none)")
    )


def sre_observation(state: DeploymentState) -> str:
    """Format deployment state from the SRE agent's perspective."""
    svc_lines = []
    for svc in state.services:
        sla_ok = svc.latency_ms < 200.0 and svc.error_rate < 0.01
        svc_lines.append(
            f"  - {svc.service}: status={svc.status}, "
            f"latency={svc.latency_ms:.1f}ms, "
            f"error_rate={svc.error_rate:.4f} "
            f"[{'OK' if sla_ok else 'VIOLATION'}]"
        )

    n = len(state.services) if state.services else 1
    avg_latency = sum(s.latency_ms for s in state.services) / n
    avg_error = sum(s.error_rate for s in state.services) / n

    return (
        f"Infrastructure health:\n"
        f"  SLA violations: {state.sla_violations}\n"
        f"  Average latency: {avg_latency:.1f}ms\n"
        f"  Average error rate: {avg_error:.4f}\n"
        f"  Deployment stage: {state.stage}\n"
        f"  Service details:\n"
        + "\n".join(svc_lines)
    )
