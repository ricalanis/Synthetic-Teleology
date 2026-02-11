"""Simulated diagnostic and repair tools for the data pipeline.

Each tool follows the LangChain-compatible pattern:
- ``.name`` and ``.description`` attributes
- ``.invoke(params)`` method that mutates PipelineState

Tools simulate realistic pipeline behaviour including schema drift
that triggers mid-run at call 8+.
"""

import time

from .models import FixAttempt, PipelineState


class CheckHealthTool:
    """Check pipeline health for a given component.

    At call 8+ introduces schema drift: drops health_score to 0.55
    and raises error_rate to 0.18.
    """

    name = "check_health"
    description = "Check health metrics for a pipeline component. Params: component (str)."

    def __init__(self, state: PipelineState) -> None:
        self._state = state
        self._call_count = 0

    def invoke(self, params: dict) -> str:
        self._call_count += 1
        component = params.get("component", "pipeline")

        # At call 8+ simulate schema drift
        if self._call_count >= 8 and not self._state.schema_drift_detected:
            self._state.schema_drift_detected = True
            self._state.health_score = 0.55
            self._state.error_rate = 0.18
            self._state.schema_version = "v2.0-drift"
            self._state.errors.append(
                "Schema drift detected: upstream producer migrated to v2.0"
            )
            self._state.warnings.append(
                "Tables 'orders' and 'events' receiving incompatible records"
            )
            return (
                f"ALERT: Schema drift detected on {component}! "
                f"health={self._state.health_score:.2f}, "
                f"error_rate={self._state.error_rate:.2f}, "
                f"schema_version={self._state.schema_version}. "
                f"Upstream producer migrated to v2.0 without coordination."
            )

        return (
            f"Health check for {component}: "
            f"health={self._state.health_score:.2f}, "
            f"error_rate={self._state.error_rate:.2f}, "
            f"throughput={self._state.throughput:.0f} rec/s, "
            f"schema={self._state.schema_version}."
        )


class RunDiagnosticTool:
    """Run diagnostic analysis on a pipeline target.

    If schema_drift_detected, returns schema incompatibility details.
    """

    name = "run_diagnostic"
    description = "Run diagnostic analysis on a pipeline target. Params: target (str)."

    def __init__(self, state: PipelineState) -> None:
        self._state = state

    def invoke(self, params: dict) -> str:
        target = params.get("target", "pipeline")

        if self._state.schema_drift_detected:
            return (
                f"Diagnostic for {target}: SCHEMA INCOMPATIBILITY. "
                f"Expected schema v1.0 fields but receiving v2.0 format. "
                f"Affected tables: orders (new field 'currency_code', removed 'price_usd'), "
                f"events (new field 'event_metadata', type change on 'timestamp'). "
                f"Error rate elevated at {self._state.error_rate:.2f}. "
                f"Recommend schema migration or rollback."
            )

        return (
            f"Diagnostic for {target}: all checks passed. "
            f"Schema {self._state.schema_version} consistent. "
            f"No anomalies detected. "
            f"Throughput: {self._state.throughput:.0f} rec/s."
        )


class ApplyFixTool:
    """Apply a fix to the pipeline.

    Gradually improves health_score (by 0.05-0.08 per successful fix).
    Records a FixAttempt. Increments fix_attempts_count.
    """

    name = "apply_fix"
    description = (
        "Apply a fix to the pipeline. "
        "Params: fix_type (str), target (str)."
    )

    def __init__(self, state: PipelineState) -> None:
        self._state = state

    def invoke(self, params: dict) -> str:
        fix_type = params.get("fix_type", "generic")
        target = params.get("target", "pipeline")

        self._state.fix_attempts_count += 1

        # Determine fix effectiveness
        if fix_type == "schema_migration":
            improvement = 0.08
            error_reduction = 0.03
            success = True
            details = f"Migrated {target} to v2.0 schema"
        elif fix_type == "add_adapter":
            improvement = 0.06
            error_reduction = 0.02
            success = True
            details = f"Added compatibility adapter for {target}"
        elif fix_type == "retry_failed":
            improvement = 0.05
            error_reduction = 0.015
            success = True
            details = f"Retried failed records on {target}"
        elif fix_type == "increase_throughput":
            improvement = 0.03
            error_reduction = 0.005
            self._state.throughput = min(
                self._state.throughput * 1.1, 1200.0
            )
            success = True
            details = f"Scaled throughput for {target}"
        elif fix_type == "drop_table":
            # Dangerous action -- should be blocked by safety checker
            improvement = 0.0
            error_reduction = 0.0
            success = False
            details = f"BLOCKED: drop_table on {target} is not allowed"
        else:
            improvement = 0.04
            error_reduction = 0.01
            success = True
            details = f"Applied generic fix ({fix_type}) to {target}"

        if success:
            self._state.health_score = min(
                1.0, self._state.health_score + improvement
            )
            self._state.error_rate = max(
                0.0, self._state.error_rate - error_reduction
            )

        attempt = FixAttempt(
            fix_type=fix_type,
            target=target,
            success=success,
            timestamp=time.time(),
            details=details,
        )
        self._state.fix_history.append(attempt)

        status = "SUCCESS" if success else "FAILED"
        return (
            f"Fix {status}: {details}. "
            f"health={self._state.health_score:.2f}, "
            f"error_rate={self._state.error_rate:.2f}."
        )


class RollbackTool:
    """Roll back a component to a previous version.

    Moderately restores health.
    """

    name = "rollback"
    description = (
        "Roll back a pipeline component to a previous version. "
        "Params: target (str), version (str)."
    )

    def __init__(self, state: PipelineState) -> None:
        self._state = state

    def invoke(self, params: dict) -> str:
        target = params.get("target", "pipeline")
        version = params.get("version", "v1.0")

        self._state.fix_attempts_count += 1

        # Rollback partially restores health
        self._state.health_score = min(
            1.0, self._state.health_score + 0.06
        )
        self._state.error_rate = max(
            0.0, self._state.error_rate - 0.02
        )

        attempt = FixAttempt(
            fix_type="rollback",
            target=target,
            success=True,
            timestamp=time.time(),
            details=f"Rolled back {target} to {version}",
        )
        self._state.fix_history.append(attempt)

        return (
            f"Rollback complete: {target} -> {version}. "
            f"health={self._state.health_score:.2f}, "
            f"error_rate={self._state.error_rate:.2f}."
        )


def build_tools(state: PipelineState) -> list:
    """Create all pipeline tools bound to the given state."""
    return [
        CheckHealthTool(state),
        RunDiagnosticTool(state),
        ApplyFixTool(state),
        RollbackTool(state),
    ]
