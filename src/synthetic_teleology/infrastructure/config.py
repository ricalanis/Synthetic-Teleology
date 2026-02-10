"""Configuration dataclasses for the Synthetic Teleology framework.

Each config is a plain ``dataclass`` with a ``validate()`` method that raises
``ValueError`` on invalid combinations.  No third-party dependencies (no
Pydantic, attrs, etc.) -- just stdlib ``dataclasses``.

Configs are intentionally **frozen** (``frozen=True``) so they can be hashed
and used as dict keys in caches without risking silent mutation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any


# ===================================================================== #
#  Loop Configuration                                                    #
# ===================================================================== #

@dataclass(frozen=True)
class LoopConfig:
    """Parameters governing the teleological perception-action loop.

    Attributes
    ----------
    max_steps:
        Hard upper limit on loop iterations.
    stop_threshold:
        If the aggregated objective score exceeds this value the loop halts
        early (higher = "good enough").
    reflection_interval:
        Run the reflection / meta-evaluation phase every *n* steps.
    eval_window:
        Number of recent steps considered for gradient / trend analysis.
    patience:
        Consecutive steps without improvement before the loop considers
        aborting or triggering a goal revision.
    revision_cooldown:
        Minimum steps between successive goal revisions to prevent
        oscillation.
    error_budget:
        Maximum cumulative handler errors before the loop aborts.
    enable_async:
        If ``True``, the loop uses the async event bus and async hooks.
    """

    max_steps: int = 200
    stop_threshold: float = 0.95
    reflection_interval: int = 10
    eval_window: int = 20
    patience: int = 30
    revision_cooldown: int = 5
    error_budget: int = 10
    enable_async: bool = False

    def validate(self) -> None:
        """Raise ``ValueError`` if any field is out of valid range."""
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if not (0.0 <= self.stop_threshold <= 1.0):
            raise ValueError(
                f"stop_threshold must be in [0, 1], got {self.stop_threshold}"
            )
        if self.reflection_interval < 1:
            raise ValueError(
                f"reflection_interval must be >= 1, got {self.reflection_interval}"
            )
        if self.eval_window < 1:
            raise ValueError(f"eval_window must be >= 1, got {self.eval_window}")
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if self.revision_cooldown < 0:
            raise ValueError(
                f"revision_cooldown must be >= 0, got {self.revision_cooldown}"
            )
        if self.error_budget < 0:
            raise ValueError(
                f"error_budget must be >= 0, got {self.error_budget}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoopConfig:
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        cfg = cls(**filtered)
        cfg.validate()
        return cfg


# ===================================================================== #
#  Agent Configuration                                                   #
# ===================================================================== #

_VALID_AGENT_TYPES = frozenset({
    "numeric",
    "llm",
    "hybrid",
    "multi_agent",
    "random",
    "custom",
})


@dataclass(frozen=True)
class AgentConfig:
    """Describes which concrete components an agent should use.

    Attributes
    ----------
    agent_type:
        One of the recognized agent archetypes.
    evaluator:
        Name of the evaluator component (looked up in the registry).
    updater:
        Name of the goal-updater component.
    planner:
        Name of the planner / action-selection component.
    llm_provider:
        LLM backend identifier (e.g. ``"anthropic"``, ``"openai"``).
        Only relevant for ``llm`` or ``hybrid`` agent types.
    llm_model:
        Model name / identifier.
    temperature:
        Sampling temperature for LLM calls.
    max_tokens:
        Maximum tokens per LLM response.
    extra:
        Catch-all for provider-specific or experimental parameters.
    """

    agent_type: str = "numeric"
    evaluator: str = "numeric"
    updater: str = "gradient"
    planner: str = "greedy"
    llm_provider: str = ""
    llm_model: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # frozen=True prevents normal assignment; use object.__setattr__
        # to coerce ``extra`` from a non-dict (e.g. None) into a dict.
        if self.extra is None:
            object.__setattr__(self, "extra", {})

    def validate(self) -> None:
        if self.agent_type not in _VALID_AGENT_TYPES:
            raise ValueError(
                f"agent_type must be one of {sorted(_VALID_AGENT_TYPES)}, "
                f"got '{self.agent_type}'"
            )
        if not self.evaluator:
            raise ValueError("evaluator name must not be empty")
        if not self.updater:
            raise ValueError("updater name must not be empty")
        if not self.planner:
            raise ValueError("planner name must not be empty")
        if self.agent_type in ("llm", "hybrid"):
            if not self.llm_provider:
                raise ValueError(
                    f"llm_provider is required for agent_type='{self.agent_type}'"
                )
            if not self.llm_model:
                raise ValueError(
                    f"llm_model is required for agent_type='{self.agent_type}'"
                )
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(
                f"temperature must be in [0, 2], got {self.temperature}"
            )
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        cfg = cls(**filtered)
        cfg.validate()
        return cfg


# ===================================================================== #
#  Benchmark Configuration                                               #
# ===================================================================== #

_VALID_BENCHMARK_TYPES = frozenset({
    "convergence",
    "adaptability",
    "multi_objective",
    "constraint_satisfaction",
    "llm_alignment",
    "custom",
})


@dataclass(frozen=True)
class BenchmarkConfig:
    """Parameters for a benchmark / experiment run.

    Attributes
    ----------
    benchmark_type:
        The class of benchmark to execute.
    num_runs:
        Number of independent runs for statistical robustness.
    seed:
        Base random seed (incremented per run).
    record_trajectory:
        If ``True``, record the full trajectory of each run for later
        analysis.
    output_dir:
        Directory to write results. Empty string means no disk output.
    tags:
        Free-form metadata tags for experiment tracking.
    extra:
        Provider-specific or experimental parameters.
    """

    benchmark_type: str = "convergence"
    num_runs: int = 10
    seed: int = 42
    record_trajectory: bool = True
    output_dir: str = ""
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tags is None:
            object.__setattr__(self, "tags", [])
        if self.extra is None:
            object.__setattr__(self, "extra", {})

    def validate(self) -> None:
        if self.benchmark_type not in _VALID_BENCHMARK_TYPES:
            raise ValueError(
                f"benchmark_type must be one of {sorted(_VALID_BENCHMARK_TYPES)}, "
                f"got '{self.benchmark_type}'"
            )
        if self.num_runs < 1:
            raise ValueError(f"num_runs must be >= 1, got {self.num_runs}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkConfig:
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        cfg = cls(**filtered)
        cfg.validate()
        return cfg


# ===================================================================== #
#  Environment Configuration                                             #
# ===================================================================== #

_VALID_ENV_TYPES = frozenset({
    "numeric_landscape",
    "grid_world",
    "resource_allocation",
    "multi_agent_negotiation",
    "llm_text_world",
    "custom",
})


@dataclass(frozen=True)
class EnvironmentConfig:
    """Parameters describing an evaluation environment.

    Attributes
    ----------
    env_type:
        The kind of environment to instantiate.
    dimensions:
        Dimensionality of the state / action space.
    initial_state:
        Starting state vector as a list of floats (length = *dimensions*).
        If empty, the environment will use its own default.
    noise_level:
        Standard deviation of observation noise (0 = deterministic).
    step_limit:
        Maximum environment steps before forced termination.
    reward_shaping:
        If ``True``, the environment provides intermediate reward signals
        beyond the raw objective evaluation.
    extra:
        Environment-specific parameters (e.g. grid size, negotiation
        protocol).
    """

    env_type: str = "numeric_landscape"
    dimensions: int = 2
    initial_state: list[float] = field(default_factory=list)
    noise_level: float = 0.0
    step_limit: int = 500
    reward_shaping: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.initial_state is None:
            object.__setattr__(self, "initial_state", [])
        if self.extra is None:
            object.__setattr__(self, "extra", {})

    def validate(self) -> None:
        if self.env_type not in _VALID_ENV_TYPES:
            raise ValueError(
                f"env_type must be one of {sorted(_VALID_ENV_TYPES)}, "
                f"got '{self.env_type}'"
            )
        if self.dimensions < 1:
            raise ValueError(f"dimensions must be >= 1, got {self.dimensions}")
        if self.initial_state and len(self.initial_state) != self.dimensions:
            raise ValueError(
                f"initial_state length ({len(self.initial_state)}) must match "
                f"dimensions ({self.dimensions})"
            )
        if self.noise_level < 0.0:
            raise ValueError(
                f"noise_level must be >= 0, got {self.noise_level}"
            )
        if self.step_limit < 1:
            raise ValueError(f"step_limit must be >= 1, got {self.step_limit}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentConfig:
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        cfg = cls(**filtered)
        cfg.validate()
        return cfg


# ===================================================================== #
#  Unified config loader                                                 #
# ===================================================================== #

_CONFIG_MAP: dict[str, type] = {
    "loop": LoopConfig,
    "agent": AgentConfig,
    "benchmark": BenchmarkConfig,
    "environment": EnvironmentConfig,
}


def load_config_from_json(json_str: str) -> dict[str, Any]:
    """Parse a JSON string into a dict of typed config objects.

    The JSON is expected to be an object whose top-level keys correspond to
    config section names (``loop``, ``agent``, ``benchmark``,
    ``environment``).  Unknown sections are preserved as raw dicts.

    Returns a dict mapping section name -> config instance (or raw dict).
    """
    raw = json.loads(json_str)
    if not isinstance(raw, dict):
        raise ValueError("Top-level JSON must be an object")
    result: dict[str, Any] = {}
    for section, data in raw.items():
        cls = _CONFIG_MAP.get(section)
        if cls is not None and isinstance(data, dict):
            result[section] = cls.from_dict(data)
        else:
            result[section] = data
    return result
