# Architecture Decisions

## 2026-02-11: Agent Feedback Loop (v1.2.0)

### Decision: Observation Enrichment Over Service Signature Changes
- **Context:** LLM services (evaluator, planner, reviser) had no memory of previous steps — they saw only the current snapshot.
- **Choice:** Enrich the `observation` text in `perceive_node` with action results, eval trends, and goal revision counts. All LLM services read `{observation}` in their prompts, so enriching observation = enriching all prompts automatically.
- **Rationale:** Zero changes to `BaseGoalUpdater` or any of its 7 concrete subclasses. No new parameters on any service. The enrichment is invisible to numeric-mode agents (no feedback on step 1). Fully backward compatible.

### Decision: WorkingMemory as Graph-Layer Utility
- **Context:** LLM examples 11+12 had no `perceive_fn`, so the agent reasoned in a vacuum with a static "Awaiting observation" message.
- **Choice:** `WorkingMemory` provides `perceive` and `record` callbacks that accumulate agent actions as a perception source. It lives in `graph/` (not `services/`) because it's a graph-layer concern.
- **Rationale:** Closes the perception-action loop for LLM agents that don't have external APIs. The `record()` callback accepts 1 or 2 args to match `act_node`'s `inspect.signature` detection for constraint-conditioned transitions.

### Decision: action_feedback as Append-Only State Channel
- **Context:** `act_node` captured tool results in events only; `perceive_node` never saw them.
- **Choice:** New `action_feedback: Annotated[list, operator.add]` channel in `TeleologicalState`. Each entry has action name, tool_name, result, step, timestamp.
- **Rationale:** Append-only channels (like `events`, `eval_history`) are the established pattern for accumulating data across iterations. Keeping feedback separate from events makes it easy for `perceive_node` to read recent results without filtering.

---

## 2026-02-11: Full Paper Alignment (v1.1.0)

### Decision: Three-Tier TC Metric Computation
- **Context:** TC was using normalized mean of eval scores, not the correlation formula from the paper.
- **Choice:** Three tiers: (1) Pearson correlation between goal-change magnitudes and eval scores (when `goal_values` available), (2) responsive-revision proxy (when revisions present), (3) legacy mean-score fallback.
- **Rationale:** Primary tier directly implements the paper's formula. Proxy and legacy tiers ensure backward compatibility when data is sparse.

### Decision: Decorator Pattern for SelfModelingEvaluator
- **Context:** Paper pattern (b) requires a self-model that predicts evaluation scores and detects surprise.
- **Choice:** `SelfModelingEvaluator` wraps any `BaseEvaluator` (Decorator pattern), adding linear regression prediction, surprise EMA, and confidence adjustment.
- **Rationale:** Any evaluator gains self-modeling by wrapping — no inheritance changes needed. R-squared gate prevents unreliable models from triggering revisions.

### Decision: Active Inference via Expected Free Energy
- **Context:** Paper Section 5.4.3 calls for active inference in goal revision.
- **Choice:** `ActiveInferenceUpdater` decomposes free energy into pragmatic (goal-state distance) and epistemic (confidence-based) components, revising when both exceed thresholds.
- **Rationale:** Dual threshold prevents spurious revisions. Blend of pragmatic and epistemic components balances exploitation and exploration.

### Decision: LLM Negotiation as 3-Phase Protocol
- **Context:** Multi-agent negotiation was limited to numeric averaging.
- **Choice:** `LLMNegotiator` implements Propose → Critique → Synthesize protocol operating on `agent_results` dicts.
- **Rationale:** Operating on result dicts (not BaseAgent objects) bridges graph and service layers cleanly. The 3-phase structure mirrors dialogue-based negotiation theory.

### Decision: Parallel Multi-Agent via LangGraph Send API
- **Context:** Sequential agent execution limits throughput in multi-agent scenarios.
- **Choice:** `parallel=True` uses LangGraph `Send` API for fan-out/fan-in with custom `_merge_agent_results` reducer.
- **Rationale:** LangGraph's native Send API handles parallelism correctly, including state merging. Sequential mode remains default for backward compatibility.

### Decision: BDI Bridge via Node Factories
- **Context:** BDI agent exists but is disconnected from LangGraph.
- **Choice:** Factory functions (`make_bdi_*_node`) wrap BDI agent methods as LangGraph nodes. `build_bdi_teleological_graph()` compiles a full graph with BDI-augmented nodes.
- **Rationale:** Factory functions keep the bridge lightweight and composable. BDI methods augment (not replace) standard nodes — falling back to standard behavior when BDI doesn't trigger.

### Decision: Rule-Based Grounding Fallback
- **Context:** Intentional grounding needs LLM for assessment, but should work without one.
- **Choice:** `IntentionalGroundingManager` supports both LLM-based and rule-based grounding. Rule-based fallback triggers when high-priority directives exceed threshold.
- **Rationale:** Enables grounding in numeric-mode agents without an LLM. LLM mode provides richer assessment when available.

---

## 2026-02-10: LLM-First Probabilistic Rewrite (v1.0.0)

### Decision: Transform Every Node from Numeric Computation to LLM-Driven Reasoning
- **Context:** v0.2.x correctly implements the Haidemariam (2026) loop structure but reasoning at each step is mechanical math (Euclidean distance, greedy action selection, boolean predicates). The goal is agents that reason, not compute.
- **Choice:** Rewrite all graph node implementations to delegate to LLM-backed services by default, while preserving numeric mode as backward-compatible fallback. The builder auto-detects mode based on `with_model()` vs `with_objective()`.
- **Rationale:** LLMs bring genuine probabilistic reasoning: multi-hypothesis planning with confidence scores, soft constraint checking with severity, chain-of-thought evaluation, and reflective goal revision. The graph topology stays identical — only node implementations change.

### Decision: LangChain `BaseChatModel` as the Universal LLM Interface
- **Context:** v0.2.x had a custom `LLMProvider` abstraction with Anthropic/OpenAI/HuggingFace/Generic providers.
- **Choice:** Use `langchain_core.language_models.BaseChatModel` directly. Deprecate custom providers with `DeprecationWarning`. Provide `LLMProviderToChatModel` bridge for migration.
- **Rationale:** LangChain's model ecosystem (ChatAnthropic, ChatOpenAI, ChatOllama, ChatGoogleGenerativeAI, etc.) is battle-tested and maintained. `with_structured_output()` gives reliable Pydantic schema parsing. No need to maintain our own provider layer.

### Decision: Pydantic Schemas for Structured LLM Output
- **Context:** LLM nodes need reliable structured responses (scores, reasoning, actions, constraint assessments).
- **Choice:** Define Pydantic `BaseModel` schemas (`EvaluationOutput`, `PlanningOutput`, `RevisionOutput`, `ConstraintCheckOutput`) and use `model.with_structured_output(Schema)` for type-safe LLM responses.
- **Rationale:** Structured output via tool-calling/JSON mode is far more reliable than regex parsing. Pydantic validation catches malformed responses. Schemas serve as both prompt documentation and runtime contracts.

### Decision: Multi-Hypothesis Planning with Softmax Selection
- **Context:** Numeric planner picks a single greedy action. LLM reasoning should explore multiple approaches.
- **Choice:** `LLMPlanner` generates N candidate hypotheses, each with confidence + reasoning + risk assessment. Softmax over confidences produces a probability distribution for `PolicySpec`.
- **Rationale:** Multiple hypotheses with probabilistic selection enables exploration vs exploitation tradeoff. Users control diversity via `num_hypotheses` and `temperature`. Metadata preserves all hypotheses for observability.

### Decision: Natural Language Goals with Optional Numeric Backing
- **Context:** v0.2.x goals are `ObjectiveVector` (float tuples with direction enums). v1.0 goals should be text-first.
- **Choice:** `Goal.description` becomes the primary representation. `success_criteria: list[str]` defines what the LLM evaluates against. `objective: ObjectiveVector | None` remains optional for hybrid use cases.
- **Rationale:** Text goals align with how humans think about objectives. Success criteria give the LLM evaluator clear checkpoints. Numeric backing is still available for environments that produce scalar observations.

### Decision: Soft Constraint Reasoning (Not Boolean Predicates)
- **Context:** v0.2.x constraints are `Callable[..., bool]` predicates — either satisfied or not.
- **Choice:** `LLMConstraintChecker` evaluates each constraint with `severity: float [0,1]`, `reasoning: str`, and `suggested_mitigation: str`. Overall safety is a reasoned judgment, not a conjunction of booleans.
- **Rationale:** Real-world constraints have nuance. "Stay within budget" might be 98% satisfied — an LLM can reason about whether the 2% overrun is acceptable. Severity scores enable graduated responses instead of hard stops.

### Decision: ReasoningQuality as 8th Metric
- **Context:** 7 metrics from the paper measure numeric agent behavior (persistence, coherence, efficiency, etc.). None measure the quality of LLM reasoning itself.
- **Choice:** Add `ReasoningQuality` metric: composite of presence ratio (% of steps with reasoning), diversity ratio (unique reasoning / total), and average length ratio (normalized reasoning depth).
- **Rationale:** LLM agents that produce repetitive, absent, or shallow reasoning should score lower. This metric differentiates between agents that genuinely reason and those that produce boilerplate. Numeric-mode agents score 0.0 (no reasoning traces).

### Decision: Dual-Mode Architecture (LLM + Numeric)
- **Context:** ~380 existing tests and 10 examples depend on numeric mode. Breaking backward compatibility would be disruptive.
- **Choice:** `GraphBuilder` detects mode automatically: `with_model()` → LLM mode (LLMEvaluator, LLMPlanner, LLMReviser, LLMConstraintChecker), `with_objective()` → numeric mode (NumericEvaluator, GreedyPlanner, ThresholdUpdater). Both use the same graph topology.
- **Rationale:** Zero migration cost for existing users. New users get LLM mode by default. The builder's `build()` method wires the correct services based on which configuration methods were called.

---

## 2026-02-10: Expanded Conceptual Examples (v0.2.2)

### Decision: Align Examples 1:1 with Paper Concepts
- **Context:** The 3 existing conceptual examples covered basic loop, multi-agent, and constraints — but left ~28 major concepts from Haidemariam (2026) undemonstrated (reflective evaluation, gradient goal revision, hierarchical goals, stochastic planning, all 7 metrics, resource/research environments, ethical constraints).
- **Choice:** Add 7 new examples (04–10) that comprehensively demonstrate every major theoretical concept. No new framework code — examples use existing classes.
- **Rationale:** Examples are the primary learning path for users. Covering all paper concepts ensures the framework's full capabilities are discoverable. Each example focuses on one service area (evaluation, revision, planning, goals, environments, metrics, constraints) and shows how to swap strategies via GraphBuilder.

---

## 2026-02-10: Example Restructuring (v0.2.1)

### Decision: Separate Conceptual vs Production Examples
- **Context:** The 16-file flat `examples/` directory mixed legacy (v0.1.0) and LangGraph examples. Most users only need 2-3 to understand the framework, while advanced users want real-world production patterns.
- **Choice:** Split into `examples/conceptual/` (3 core demos) and `examples/production/` (self-contained mini-packages with custom strategies, domain models, and CLI entry points).
- **Rationale:** Conceptual examples stay simple and focused on framework APIs. Production examples demonstrate the full Strategy pattern — custom evaluators, planners, and constraint checkers for real domains — without cluttering the learning path. Each production agent is a self-contained package runnable as `python -m examples.production.<agent>.main`.

### Decision: Production Examples as Mini-Packages (Not Single Scripts)
- **Context:** A single-file production example would be too large and hard to navigate.
- **Choice:** Each production agent has its own package: `models.py` (domain), `market_data.py`/`crm.py` (data layer), `strategies.py` (custom evaluator/planner/constraints), `agent.py` (graph wiring), `main.py` (CLI).
- **Rationale:** Mirrors how a user would structure a real agent project. Demonstrates clean separation of concerns. The data layer provides both a real API client and a simulated fallback, so examples work out of the box without API keys.

---

## 2026-02-10: LangGraph Migration (v0.2.0)

### Decision: Replace Custom Agentic Loop with LangGraph StateGraph
- **Context:** The v0.1.0 custom `SyncAgenticLoop` / `AsyncAgenticLoop` worked but was isolated from the LangGraph ecosystem (checkpointing, streaming, human-in-the-loop, LangGraph Studio).
- **Choice:** Create a new `graph/` package that implements the teleological loop as a LangGraph `StateGraph`. The 10-step loop becomes 8 nodes + 2 conditional edges. Strategy services (evaluators, planners, etc.) are injected via the state dict.
- **Rationale:** LangGraph provides battle-tested orchestration (checkpointing, streaming, interrupts, subgraphs) without reimplementing these features. The teleological theory layer (domain, services) remains unchanged — only the orchestration layer is replaced.

### Decision: TypedDict State (Not Pydantic)
- **Context:** LangGraph supports both TypedDict and Pydantic for state schemas.
- **Choice:** `TeleologicalState(TypedDict, total=False)` with `Annotated[list, operator.add]` for append-only channels.
- **Rationale:** TypedDict is lightweight, requires no extra dependencies, and works well with LangGraph's channel system. `total=False` allows partial state updates from nodes. The `operator.add` reducer enables event accumulation without full state replacement.

### Decision: No `from __future__ import annotations` in State/Schema Files
- **Context:** LangGraph uses `get_type_hints(..., include_extras=True)` at runtime to resolve TypedDict annotations.
- **Choice:** Use explicit imports instead of forward references in `state.py` and `multi_agent.py`.
- **Rationale:** `from __future__ import annotations` converts all annotations to strings, which breaks LangGraph's runtime type resolution. This is a known LangGraph/Python 3.11 compatibility constraint.

### Decision: GraphBuilder Returns `(app, initial_state)` Tuple
- **Context:** Unlike the old `AgentBuilder.build()` which returned a configured agent, the graph needs both the compiled graph and an initial state dict.
- **Choice:** `GraphBuilder.build()` returns a `(CompiledStateGraph, dict)` tuple.
- **Rationale:** The initial state contains injected strategies (evaluator, planner, etc.) as values in the dict. This is the cleanest way to pass non-serializable callables into the graph without global state.

### Decision: Deprecate Rather Than Remove Legacy Loop
- **Context:** 380 existing tests use `SyncAgenticLoop`. Breaking them would reduce confidence in the migration.
- **Choice:** Add `DeprecationWarning` to `SyncAgenticLoop.__init__` and `AsyncAgenticLoop.__init__`. Keep all existing tests passing.
- **Rationale:** Gradual migration path. Users can switch at their own pace. The parity test (`test_graph.py::TestGraphParityWithSyncLoop`) ensures the graph produces comparable results.

### Decision: Multi-Agent via Subgraph Composition
- **Context:** Multi-agent coordination needs each agent to run its own teleological loop, then negotiate.
- **Choice:** `build_multi_agent_graph()` creates per-agent nodes that internally invoke `build_teleological_graph()` as subgraphs. A negotiation node runs between rounds.
- **Rationale:** LangGraph subgraph composition is the natural pattern. Each agent's subgraph is fully independent, enabling different configurations per agent.

---

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
