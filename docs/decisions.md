# Architecture Decisions

## 2026-02-10: Overall Architecture

### Decision: Domain-Driven Design with Bounded Contexts
- **Context:** Framework implementing Haidemariam (2026) Synthetic Teleology theory with complex interactions between goals, agents, evaluation, planning, and coordination.
- **Choice:** Five bounded contexts: Teleology Domain, Agency Domain, Coordination Domain, Measurement Domain, Infrastructure.
- **Rationale:** Clean separation of concerns. Each context has its own entities, value objects, and events. Integration happens through domain events on the EventBus, not direct coupling.

### Decision: Frozen Dataclasses for Value Objects
- **Context:** Goals, states, signals, and specifications need to be passed between components without risk of mutation.
- **Choice:** All value objects in `domain/values.py` are frozen `@dataclass(frozen=True)` with `__post_init__` validation.
- **Rationale:** Immutability prevents accidental state corruption. Goal revisions produce new objects rather than mutating existing ones, maintaining a clear history. Frozen dataclasses are hashable by default.

### Decision: Event-Driven Architecture with Synchronous EventBus
- **Context:** The agentic loop needs to communicate with metrics, visualization, and logging without tight coupling.
- **Choice:** `EventBus` with synchronous dispatch (thread-safe via `threading.Lock`). Separate `AsyncEventBus` for async contexts.
- **Rationale:** Synchronous dispatch keeps the default case simple and deterministic. Events are immutable (frozen dataclasses), so handlers cannot corrupt state. The `EventStore` enables replay for debugging.

### Decision: `src/` Layout with Hatchling Build
- **Context:** Need a clean Python package structure that supports editable installs and modern tooling.
- **Choice:** `src/synthetic_teleology/` layout with hatchling as build backend.
- **Rationale:** `src/` layout prevents accidental imports from the project root. Hatchling is lightweight and well-supported by uv.

## 2026-02-10: Service Layer Patterns

### Decision: Template Method for Agentic Loop
- **Context:** The loop `perceive → evaluate → revise → plan → filter → act → transition` is the invariant algorithm. Individual steps vary.
- **Choice:** `BaseAgenticLoop.run()` is the fixed template. Subclasses override `before_step()`, `after_step()`, `on_error()`. Strategy methods live on the agent.
- **Rationale:** The 10-step sequence is fundamental to the theory. Making it a template method ensures all agents follow the same loop structure while allowing customization of individual steps. `SyncAgenticLoop` and `AsyncAgenticLoop` vary only in execution model.

### Decision: Strategy Pattern for Evaluators, Updaters, Planners
- **Context:** Each of these services has multiple implementations (numeric vs LLM, threshold vs gradient vs hierarchical, greedy vs stochastic).
- **Choice:** ABC base classes with interchangeable implementations. Agents receive strategies via constructor injection.
- **Rationale:** Open/Closed principle — new evaluation/update/planning strategies can be added without modifying existing code. The `AgentBuilder` makes wiring strategies ergonomic.

### Decision: Chain of Responsibility for Goal Revision and Constraints
- **Context:** Goal revision and constraint checking involve multiple independent checks that should run in sequence with optional short-circuit.
- **Choice:** `GoalUpdaterChain` and `ConstraintPipeline` implement CoR. Each handler decides whether to process and/or pass to the next.
- **Rationale:** Composable pipelines. `ConstraintPipeline` supports `fail_fast=True` for safety-critical constraints. Order of handlers matters and is explicit.

### Decision: Decorator Pattern for ReflectiveEvaluator
- **Context:** Need to add self-model/drift-detection behavior on top of any existing evaluator without modifying it.
- **Choice:** `ReflectiveEvaluator` wraps any `BaseEvaluator`, tracks score history, computes EMA drift, and adjusts confidence.
- **Rationale:** Any evaluator (numeric, LLM, composite) can gain reflective capabilities by wrapping. No inheritance needed — true Decorator pattern.

## 2026-02-10: Agent Architecture

### Decision: Finite State Machine for Agent Lifecycle
- **Context:** Agents progress through well-defined phases (perceive, evaluate, revise, plan, act, reflect).
- **Choice:** `AgentState` enum with `_VALID_TRANSITIONS` dict. `_transition_to()` validates transitions.
- **Rationale:** Prevents invalid state sequences (e.g., acting before evaluating). The FSM is fundamental to the teleological loop structure. Invalid transitions raise `ValueError` immediately.

### Decision: BDI Agent as Alternative Reference Implementation
- **Context:** Need to validate the framework against classical BDI (Belief-Desire-Intention) agent architecture.
- **Choice:** `BDIAgent(BaseAgent)` mapping Beliefs→S_t, Desires→G_t, Intentions→π_t.
- **Rationale:** BDI is the most widely understood agent architecture. Having it alongside TeleologicalAgent demonstrates the framework's flexibility and enables direct comparison in benchmarks.

### Decision: Fluent Builder for Agent Construction
- **Context:** Fully configured agents require evaluator + updater + planner + constraints + event bus + optional LLM provider.
- **Choice:** `AgentBuilder` with fluent API: `.with_goal()`, `.with_evaluator()`, `.with_planner()`, etc.
- **Rationale:** Complex object construction without constructor explosion. Builder validates completeness on `build()`. Factory methods provide shortcuts for common configurations.

## 2026-02-10: Measurement Layer

### Decision: 7 Metrics from Haidemariam (2026) Theory
- **Context:** The paper defines specific metrics for evaluating teleological agents.
- **Choice:** GoalPersistence, TeleologicalCoherence, ReflectiveEfficiency, Adaptivity, NormativeFidelity, InnovationYield, LyapunovStability.
- **Rationale:** Direct mapping to the theoretical framework. Each metric is a `BaseMetric` subclass with `compute()`, `validate()`, `describe()`. The `MetricsEngine` composes all of them.

### Decision: Template Method Pattern for Benchmarks
- **Context:** Need a reusable framework for running different benchmark scenarios.
- **Choice:** `BaseBenchmark` ABC: `setup() → run_scenario(seed) → collect_metrics(log) → teardown()`. The `run()` method orchestrates N independent runs.
- **Rationale:** Each benchmark defines its own scenario while sharing orchestration, error handling, and aggregation logic.

### Decision: Composite Pattern for BenchmarkSuite
- **Context:** Need to run multiple benchmarks together and aggregate results.
- **Choice:** `BenchmarkSuite` wraps an `OrderedDict[str, BaseBenchmark]`.
- **Rationale:** Composability — suites can contain any mix of benchmarks, results keyed by name.

## 2026-02-10: LLM Provider Layer

### Decision: Lazy Imports via `__getattr__` in `llm/__init__.py`
- **Context:** Concrete providers depend on optional packages (anthropic, openai, httpx, transformers/torch).
- **Choice:** Module-level `__getattr__` that lazy-imports concrete provider classes on first attribute access.
- **Rationale:** Users who only need one provider don't need all others installed. Import errors are deferred to actual usage.

### Decision: Strategy + Fallback Chain in ProviderRouter
- **Context:** Production systems need resilience when an LLM provider goes down.
- **Choice:** Priority-ordered routing with health tracking (consecutive failure threshold) and time-based cooldown re-enabling.
- **Rationale:** Automatic failover without manual intervention. Cooldown prevents hammering a down provider while eventually retrying.

### Decision: Registry-Based Factory with Auto-Discovery
- **Context:** Need a single entry point for creating any LLM provider by name.
- **Choice:** `LLMProviderFactory` with pre-registered lazy constructors and `register()` for custom providers.
- **Rationale:** Decouples provider creation from usage. Declarative configuration support enables config-file-driven setup.

### Decision: Exponential Backoff for API Providers
- **Context:** API-based providers can hit rate limits.
- **Choice:** Built-in retry with exponential backoff in each provider's `generate()` method.
- **Rationale:** Avoids requiring external retry middleware. Each provider handles its own rate limit format.

## 2026-02-10: Presentation Layer

### Decision: Optional Rich/Matplotlib Dependencies
- **Context:** Console dashboard and plotting are nice-to-have but shouldn't be required dependencies.
- **Choice:** Rich and matplotlib are optional (`[viz]` extra). All presentation modules gracefully degrade to plain-text output when dependencies are missing.
- **Rationale:** Core framework stays lightweight. `presentation/console.py` renders plain-text tables if Rich is unavailable. `presentation/plots.py` raises ImportError only when plotting functions are called.
