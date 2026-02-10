# Changelog

All notable changes to the Synthetic Teleology Framework.

## [0.1.0] — 2026-02-10

Initial release implementing the full Synthetic Teleology framework per Haidemariam (2026).

### Phase 1: Domain Layer

- **`domain/enums.py`** — 8 domain enumerations: Direction, ConstraintType, StateSource, AgentState, GoalStatus, RevisionReason, NegotiationStrategy, ErrorAction.
- **`domain/values.py`** — 7 frozen value objects: ObjectiveVector (with `distance_to()`, `with_values()`), EvalSignal (score/confidence validation), ConstraintSpec, ActionSpec, PolicySpec, StateSnapshot, GoalRevision.
- **`domain/entities.py`** — Goal entity (versioned, with `revise()`), Constraint entity (activation lifecycle).
- **`domain/aggregates.py`** — GoalTree (Composite pattern for hierarchical goals), ConstraintSet (priority-sorted), AgentIdentity (owns goal history).
- **`domain/events.py`** — 15+ frozen domain event dataclasses: GoalCreated, GoalRevised, GoalAbandoned, GoalAchieved, EvaluationCompleted, PlanGenerated, ActionExecuted, ConstraintViolated, ConstraintRestored, ReflectionTriggered, LoopStepCompleted, AgentRegistered, StateChanged, NegotiationStarted, ConsensusReached, PerturbationInjected.
- **`domain/exceptions.py`** — SyntheticTeleologyError base class + GoalCoherenceError, ConstraintViolationError, EvaluationError, PlanningError, NegotiationDeadlock.

### Phase 2: Infrastructure Layer

- **`infrastructure/event_bus.py`** — EventBus (synchronous, thread-safe pub/sub), AsyncEventBus, EventStore (append + query by type/time).
- **`infrastructure/registry.py`** — ComponentRegistry with decorator-based registration (`@registry.register("name")`).
- **`infrastructure/config.py`** — LoopConfig, AgentConfig, BenchmarkConfig, EnvironmentConfig dataclasses with validation.
- **`infrastructure/serialization.py`** — JSON/YAML serialization for all domain types.

### Phase 3: Service Layer

- **`services/evaluation.py`** — BaseEvaluator ABC, NumericEvaluator (vector distance scoring), CompositeEvaluator (weighted aggregation, Composite pattern), ReflectiveEvaluator (Decorator pattern with EMA drift detection and confidence adjustment), LLMCriticEvaluator (LLM-as-judge).
- **`services/goal_revision.py`** — BaseGoalUpdater ABC, ThresholdUpdater, GradientUpdater, GoalUpdaterChain (Chain of Responsibility), HierarchicalUpdater (meta-goal regularization), UncertaintyAwareUpdater (active inference style), ConstrainedUpdater (constraint-respecting), LLMGoalEditor.
- **`services/planning.py`** — BasePlanner ABC, GreedyPlanner, StochasticPlanner, HierarchicalPlanner (sub-goal decomposition), LLMPlanner.
- **`services/constraint_engine.py`** — BaseConstraintChecker ABC, SafetyChecker, BudgetChecker, EthicalChecker, ConstraintPipeline (Chain of Responsibility with fail_fast), PolicyFilter.
- **`services/loop.py`** — StopReason enum, RunResult, BaseAgenticLoop (10-step template: perceive → evaluate → revise → plan → filter → act → transition), SyncAgenticLoop, AsyncAgenticLoop.
- **`services/coordination.py`** — BaseNegotiator ABC, ConsensusNegotiator (iterative averaging), VotingNegotiator (social choice), AuctionNegotiator (bid-based), CoordinationMediator (Mediator pattern).

### Phase 4: Agents

- **`agents/base.py`** — BaseAgent ABC with finite state machine (IDLE → PERCEIVING → EVALUATING → REVISING → PLANNING → ACTING → REFLECTING), transition validation, abstract strategy methods.
- **`agents/teleological.py`** — TeleologicalAgent(BaseAgent) wiring evaluator + updater + planner + constraints into the full Eq 2-6 loop.
- **`agents/bdi.py`** — BDIAgent(BaseAgent) with Beliefs→S_t, Desires→G_t, Intentions→π_t classical mapping.
- **`agents/llm.py`** — LLMAgent(TeleologicalAgent) with LLMAgentConfig, `from_provider()` class method. Uses LLMCriticEvaluator, LLMGoalEditor, LLMPlanner.
- **`agents/factory.py`** — AgentFactory (Factory Method) + AgentBuilder (fluent Builder pattern for complex configurations).

### Phase 5: Environments

- **`environments/base.py`** — BaseEnvironment ABC: `step()`, `observe()`, `reset()`, `inject_perturbation()`.
- **`environments/numeric.py`** — NumericEnvironment: N-dimensional continuous space with configurable dynamics.
- **`environments/resource.py`** — ResourceEnvironment: competing resource allocation with scarcity.
- **`environments/research.py`** — ResearchEnvironment: simulated knowledge synthesis (Section 4.1.2).
- **`environments/shared.py`** — SharedEnvironment: multi-agent shared state with per-agent observations.

### Phase 6: Measurement

- **`measurement/collector.py`** — AgentLogEntry, AgentLog (derived properties: scores, costs, revision counts), EventCollector (subscribes to domain events, builds per-agent logs).
- **`measurement/metrics/base.py`** — MetricResult (frozen), BaseMetric ABC with Template Method (`compute()`, `validate()`, `describe()`).
- **`measurement/metrics/`** — 7 metric implementations:
  - GoalPersistence (GP) — fraction of steps maintaining same goal
  - TeleologicalCoherence (TC) — correlation between eval score improvement and goal stability
  - ReflectiveEfficiency (RE) — cost-weighted score improvement per step
  - Adaptivity (AD) — post-perturbation recovery rate
  - NormativeFidelity (NF) — fraction of steps without constraint violations
  - InnovationYield (IY) — score improvement from novel (unique) actions
  - LyapunovStability (LS) — score variance convergence over time
- **`measurement/engine.py`** — MetricsEngine: Composite of BaseMetric instances, ships with all 7 defaults.
- **`measurement/report.py`** — MetricsReport: frozen result container with `to_dict()`, `summary()`.

### Phase 7-8: Benchmarks

- **`measurement/benchmarks/base.py`** — BaseBenchmark ABC with Template Method: setup → run_scenario → collect_metrics → teardown.
- **`measurement/benchmarks/suite.py`** — BenchmarkSuite: Composite pattern, OrderedDict of named benchmarks, `run_all()` with summary table.
- **`measurement/benchmarks/distribution_shift.py`** — Two-phase scenario testing agent adaptivity with mid-run perturbation injection.
- **`measurement/benchmarks/conflicting_obj.py`** — Coupled dynamics with conflict_strength parameter testing multi-objective coherence.
- **`measurement/benchmarks/negotiation.py`** — Multi-agent negotiation with CoordinationMediator (centroid consensus, momentum, convergence threshold).
- **`measurement/benchmarks/knowledge_synthesis.py`** — ResearchEnvironment with heuristic ResearchPlanner.

### Phase 9-10: LLM Providers

- **`infrastructure/llm/__init__.py`** — BaseLLMProvider ABC, LLMConfig, LLMMessage, LLMResponse, error hierarchy. Lazy `__getattr__` for concrete providers.
- **`infrastructure/llm/anthropic.py`** — AnthropicProvider with exponential backoff on rate limits.
- **`infrastructure/llm/openai_provider.py`** — OpenAIProvider with backoff, system prompt handling.
- **`infrastructure/llm/openapi.py`** — GenericOpenAPIProvider using httpx for any OpenAI-compatible endpoint.
- **`infrastructure/llm/huggingface.py`** — HuggingFaceLocalProvider with lazy model loading, GPU cleanup.
- **`infrastructure/llm/router.py`** — ProviderRouter: Strategy + Fallback Chain with health tracking and cooldown.
- **`infrastructure/llm/factory.py`** — LLMProviderFactory: registry-based auto-discovery, availability checking.

### Phase 11: Presentation + CLI

- **`presentation/console.py`** — ConsoleDashboard with Rich-based rendering (optional) and plain-text fallback. `print_report()`, `print_comparison()`, `print_trajectory()` (Unicode sparklines), `print_benchmark_results()`.
- **`presentation/plots.py`** — Matplotlib plotting: `plot_score_trajectory()`, `plot_goal_revisions()`, `plot_metric_comparison()` (radar chart), `plot_phase_portrait()`.
- **`presentation/export.py`** — Export to JSON, CSV, HTML (self-contained with inline CSS).
- **`cli.py`** — argparse CLI with 4 subcommands: `run`, `benchmark`, `report`, `info`.
- **`__main__.py`** — Enables `python -m synthetic_teleology`.

### Phase 12: Examples

- **`examples/01_basic_loop.py`** — End-to-end SyncAgenticLoop with NumericEnvironment.
- **`examples/02_hierarchical_goals.py`** — GoalTree with subgoals and coherence validation.
- **`examples/03_reflective_critic.py`** — ReflectiveEvaluator confidence adjustment demo.
- **`examples/04_multi_agent_negotiation.py`** — SharedEnvironment with consensus negotiation.
- **`examples/05_benchmark_suite.py`** — MetricsEngine with synthetic AgentLog data.
- **`examples/06_llm_research_agent.py`** — LLMAgent configuration with mock provider.
- **`examples/07_distribution_shift.py`** — DistributionShiftBenchmark end-to-end.
- **`examples/08_constraint_integration.py`** — Constraint pipeline with SafetyChecker and BudgetChecker.

### Phase 13: Tests

- **380 tests** across 15 test modules:
  - `tests/domain/` — test_values, test_entities, test_aggregates, test_events
  - `tests/services/` — test_evaluation, test_goal_revision, test_planning, test_constraints, test_loop
  - `tests/agents/` — test_teleological, test_factory
  - `tests/environments/` — test_numeric, test_shared
  - `tests/infrastructure/` — test_event_bus, test_registry, test_llm_providers
  - `tests/measurement/` — test_collector, test_metrics, test_benchmarks

### Project Configuration

- **`pyproject.toml`** — hatchling build, Python >=3.11, numpy dependency. Optional dependency groups: llm-anthropic, llm-openai, llm-huggingface, llm-generic, viz, all, dev.
- **`README.md`** — Project documentation with installation, usage, examples.

### Summary

- **64 source files** across 8 packages (domain, services, agents, environments, measurement, infrastructure, presentation, CLI)
- **27 test files** with 380 passing tests
- **8 example scripts** demonstrating all major features
- **12+ GoF design patterns**: Strategy, Observer, Composite, Builder, Factory, Chain of Responsibility, Mediator, Decorator, Template Method, State Machine, Repository, Value Object
