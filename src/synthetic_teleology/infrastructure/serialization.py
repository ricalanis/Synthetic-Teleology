"""Serialization utilities for the Synthetic Teleology framework.

Provides ``to_dict`` / ``from_dict`` round-trip conversion for all core domain
value objects, entities, configs, and metrics reports.  JSON is always
available; YAML support is optional (graceful fallback if ``pyyaml`` is not
installed).

Design goals:
- Zero mandatory third-party deps (stdlib ``json`` only).
- Every ``to_dict`` output is JSON-serializable (no numpy, no sets, etc.).
- ``from_dict`` reconstructors accept permissive input and raise ``ValueError``
  for truly unrecoverable data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from typing import Any

from synthetic_teleology.domain.entities import Constraint, Goal
from synthetic_teleology.domain.enums import (
    ConstraintType,
    Direction,
    GoalStatus,
    RevisionReason,
)
from synthetic_teleology.domain.values import (
    ActionSpec,
    ConstraintSpec,
    EvalSignal,
    GoalRevision,
    ObjectiveVector,
    PolicySpec,
    StateSnapshot,
)
from synthetic_teleology.infrastructure.config import (
    AgentConfig,
    BenchmarkConfig,
    EnvironmentConfig,
    LoopConfig,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Optional YAML support                                                       #
# --------------------------------------------------------------------------- #

try:
    import yaml as _yaml  # type: ignore[import-untyped]

    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _yaml = None  # type: ignore[assignment]
    _HAS_YAML = False


# =========================================================================== #
#  Generic helpers                                                             #
# =========================================================================== #

def _enum_val(v: Any) -> Any:
    """Return the ``.value`` if *v* is an enum member, else *v* unchanged."""
    if hasattr(v, "value"):
        return v.value
    return v


def _coerce_floats(seq: Any) -> list[float]:
    """Convert an iterable of numerics to ``list[float]``."""
    if seq is None:
        return []
    return [float(x) for x in seq]


def _safe_dataclass_dict(obj: Any) -> dict[str, Any]:
    """Like ``dataclasses.asdict`` but handles enums and nested objects."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"{obj!r} is not a dataclass instance")


# =========================================================================== #
#  Value objects                                                               #
# =========================================================================== #

def objective_vector_to_dict(ov: ObjectiveVector) -> dict[str, Any]:
    return {
        "values": list(ov.values),
        "weights": list(ov.weights),
        "directions": [_enum_val(d) for d in ov.directions],
    }


def objective_vector_from_dict(data: dict[str, Any]) -> ObjectiveVector:
    return ObjectiveVector(
        values=tuple(_coerce_floats(data["values"])),
        weights=tuple(_coerce_floats(data["weights"])),
        directions=tuple(Direction(d) for d in data["directions"]),
    )


def eval_signal_to_dict(es: EvalSignal) -> dict[str, Any]:
    return {
        "objective_scores": list(es.objective_scores),
        "constraint_violations": list(es.constraint_violations),
        "aggregated_score": es.aggregated_score,
        "metadata": dict(es.metadata) if es.metadata else {},
    }


def eval_signal_from_dict(data: dict[str, Any]) -> EvalSignal:
    return EvalSignal(
        objective_scores=tuple(_coerce_floats(data["objective_scores"])),
        constraint_violations=tuple(_coerce_floats(data["constraint_violations"])),
        aggregated_score=float(data["aggregated_score"]),
        metadata=dict(data.get("metadata", {})),
    )


def constraint_spec_to_dict(cs: ConstraintSpec) -> dict[str, Any]:
    return {
        "name": cs.name,
        "constraint_type": _enum_val(cs.constraint_type),
        "threshold": cs.threshold,
        "weight": cs.weight,
    }


def constraint_spec_from_dict(data: dict[str, Any]) -> ConstraintSpec:
    return ConstraintSpec(
        name=str(data["name"]),
        constraint_type=ConstraintType(data["constraint_type"]),
        threshold=float(data["threshold"]),
        weight=float(data.get("weight", 1.0)),
    )


def action_spec_to_dict(a: ActionSpec) -> dict[str, Any]:
    return {
        "name": a.name,
        "parameters": dict(a.parameters) if a.parameters else {},
        "cost": a.cost,
    }


def action_spec_from_dict(data: dict[str, Any]) -> ActionSpec:
    return ActionSpec(
        name=str(data["name"]),
        parameters=dict(data.get("parameters", {})),
        cost=float(data.get("cost", 0.0)),
    )


def policy_spec_to_dict(ps: PolicySpec) -> dict[str, Any]:
    return {
        "name": ps.name,
        "actions": [action_spec_to_dict(a) for a in ps.actions],
        "metadata": dict(ps.metadata) if ps.metadata else {},
    }


def policy_spec_from_dict(data: dict[str, Any]) -> PolicySpec:
    return PolicySpec(
        name=str(data["name"]),
        actions=tuple(action_spec_from_dict(a) for a in data.get("actions", [])),
        metadata=dict(data.get("metadata", {})),
    )


def state_snapshot_to_dict(ss: StateSnapshot) -> dict[str, Any]:
    return {
        "step": ss.step,
        "state_vector": list(ss.state_vector),
        "timestamp": ss.timestamp,
        "source": _enum_val(ss.source),
        "metadata": dict(ss.metadata) if ss.metadata else {},
    }


def state_snapshot_from_dict(data: dict[str, Any]) -> StateSnapshot:
    from synthetic_teleology.domain.enums import StateSource

    return StateSnapshot(
        step=int(data["step"]),
        state_vector=tuple(_coerce_floats(data["state_vector"])),
        timestamp=float(data["timestamp"]),
        source=StateSource(data["source"]),
        metadata=dict(data.get("metadata", {})),
    )


def goal_revision_to_dict(gr: GoalRevision) -> dict[str, Any]:
    return {
        "old_objective": objective_vector_to_dict(gr.old_objective),
        "new_objective": objective_vector_to_dict(gr.new_objective),
        "reason": _enum_val(gr.reason),
        "step": gr.step,
        "timestamp": gr.timestamp,
        "metadata": dict(gr.metadata) if gr.metadata else {},
    }


def goal_revision_from_dict(data: dict[str, Any]) -> GoalRevision:
    return GoalRevision(
        old_objective=objective_vector_from_dict(data["old_objective"]),
        new_objective=objective_vector_from_dict(data["new_objective"]),
        reason=RevisionReason(data["reason"]),
        step=int(data["step"]),
        timestamp=float(data["timestamp"]),
        metadata=dict(data.get("metadata", {})),
    )


# =========================================================================== #
#  Entities                                                                    #
# =========================================================================== #

def goal_to_dict(g: Goal) -> dict[str, Any]:
    return {
        "goal_id": g.goal_id,
        "name": g.name,
        "objective": objective_vector_to_dict(g.objective),
        "constraints": [constraint_spec_to_dict(c) for c in g.constraints],
        "status": _enum_val(g.status),
        "priority": g.priority,
        "parent_id": g.parent_id,
        "revisions": [goal_revision_to_dict(r) for r in g.revisions],
        "metadata": dict(g.metadata) if g.metadata else {},
    }


def goal_from_dict(data: dict[str, Any]) -> Goal:
    return Goal(
        goal_id=str(data["goal_id"]),
        name=str(data["name"]),
        objective=objective_vector_from_dict(data["objective"]),
        constraints=tuple(
            constraint_spec_from_dict(c) for c in data.get("constraints", [])
        ),
        status=GoalStatus(data.get("status", "active")),
        priority=float(data.get("priority", 1.0)),
        parent_id=data.get("parent_id"),
        revisions=tuple(
            goal_revision_from_dict(r) for r in data.get("revisions", [])
        ),
        metadata=dict(data.get("metadata", {})),
    )


def constraint_to_dict(c: Constraint) -> dict[str, Any]:
    return {
        "constraint_id": c.constraint_id,
        "name": c.name,
        "constraint_type": _enum_val(c.constraint_type),
        "threshold": c.threshold,
        "weight": c.weight,
        "active": c.active,
        "metadata": dict(c.metadata) if c.metadata else {},
    }


def constraint_from_dict(data: dict[str, Any]) -> Constraint:
    return Constraint(
        constraint_id=str(data["constraint_id"]),
        name=str(data["name"]),
        constraint_type=ConstraintType(data["constraint_type"]),
        threshold=float(data["threshold"]),
        weight=float(data.get("weight", 1.0)),
        active=bool(data.get("active", True)),
        metadata=dict(data.get("metadata", {})),
    )


# =========================================================================== #
#  Configs                                                                     #
# =========================================================================== #

def config_to_dict(cfg: LoopConfig | AgentConfig | BenchmarkConfig | EnvironmentConfig) -> dict[str, Any]:
    """Serialize any of the four config dataclasses to a plain dict."""
    return cfg.to_dict()


def config_from_dict(
    data: dict[str, Any],
    config_type: type[LoopConfig] | type[AgentConfig] | type[BenchmarkConfig] | type[EnvironmentConfig],
) -> LoopConfig | AgentConfig | BenchmarkConfig | EnvironmentConfig:
    """Deserialize a dict back into the specified config type."""
    return config_type.from_dict(data)


# =========================================================================== #
#  MetricsReport (forward-compatible stub)                                     #
# =========================================================================== #

def metrics_report_to_dict(report: Any) -> dict[str, Any]:
    """Serialize a metrics report.

    Accepts any object with a ``to_dict()`` method **or** any dataclass.
    Falls back to ``vars()`` for plain objects.
    """
    if hasattr(report, "to_dict") and callable(report.to_dict):
        return report.to_dict()
    if is_dataclass(report) and not isinstance(report, type):
        return asdict(report)
    if hasattr(report, "__dict__"):
        return dict(vars(report))
    raise TypeError(f"Cannot serialize metrics report of type {type(report).__name__}")


def metrics_report_from_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Deserialize a metrics report.

    Since ``MetricsReport`` is defined in the measurement layer (which may
    not yet exist), this returns a plain dict.  The measurement layer can
    provide its own typed ``from_dict`` that delegates here for raw parsing.
    """
    return dict(data)


# =========================================================================== #
#  Unified serializer                                                          #
# =========================================================================== #

# Maps type -> (to_dict_fn, from_dict_fn)
_SERIALIZERS: dict[type, tuple[Any, Any]] = {
    ObjectiveVector: (objective_vector_to_dict, objective_vector_from_dict),
    EvalSignal: (eval_signal_to_dict, eval_signal_from_dict),
    ConstraintSpec: (constraint_spec_to_dict, constraint_spec_from_dict),
    ActionSpec: (action_spec_to_dict, action_spec_from_dict),
    PolicySpec: (policy_spec_to_dict, policy_spec_from_dict),
    StateSnapshot: (state_snapshot_to_dict, state_snapshot_from_dict),
    GoalRevision: (goal_revision_to_dict, goal_revision_from_dict),
    Goal: (goal_to_dict, goal_from_dict),
    Constraint: (constraint_to_dict, constraint_from_dict),
    LoopConfig: (config_to_dict, None),
    AgentConfig: (config_to_dict, None),
    BenchmarkConfig: (config_to_dict, None),
    EnvironmentConfig: (config_to_dict, None),
}


def serialize(obj: Any) -> dict[str, Any]:
    """Serialize a known domain/infrastructure object to a dict.

    Raises ``TypeError`` for unsupported types.
    """
    ser = _SERIALIZERS.get(type(obj))
    if ser is not None:
        to_fn, _ = ser
        return to_fn(obj)
    # Fallback for metrics reports or unknown dataclasses
    return metrics_report_to_dict(obj)


def deserialize(data: dict[str, Any], target_type: type) -> Any:
    """Deserialize a dict into *target_type*.

    For configs, use their ``from_dict`` classmethod directly.
    """
    ser = _SERIALIZERS.get(target_type)
    if ser is not None:
        _, from_fn = ser
        if from_fn is not None:
            return from_fn(data)
        # Configs
        if hasattr(target_type, "from_dict"):
            return target_type.from_dict(data)
    raise TypeError(f"No deserializer registered for {target_type.__name__}")


# =========================================================================== #
#  JSON helpers                                                                #
# =========================================================================== #

def to_json(obj: Any, *, indent: int | None = 2) -> str:
    """Serialize a domain/infra object to a JSON string."""
    d = serialize(obj)
    return json.dumps(d, indent=indent, default=str)


def from_json(json_str: str, target_type: type) -> Any:
    """Deserialize a JSON string into *target_type*."""
    data = json.loads(json_str)
    return deserialize(data, target_type)


# =========================================================================== #
#  YAML helpers (optional)                                                     #
# =========================================================================== #

def to_yaml(obj: Any) -> str:
    """Serialize a domain/infra object to a YAML string.

    Raises ``RuntimeError`` if PyYAML is not installed.
    """
    if not _HAS_YAML:
        raise RuntimeError(
            "PyYAML is not installed. Install it with: pip install pyyaml"
        )
    d = serialize(obj)
    return _yaml.dump(d, default_flow_style=False, sort_keys=False)


def from_yaml(yaml_str: str, target_type: type) -> Any:
    """Deserialize a YAML string into *target_type*.

    Raises ``RuntimeError`` if PyYAML is not installed.
    """
    if not _HAS_YAML:
        raise RuntimeError(
            "PyYAML is not installed. Install it with: pip install pyyaml"
        )
    data = _yaml.safe_load(yaml_str)
    return deserialize(data, target_type)


def yaml_available() -> bool:
    """Return ``True`` if PyYAML is importable."""
    return _HAS_YAML
