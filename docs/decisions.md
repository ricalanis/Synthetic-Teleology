# Architecture Decisions

## Graph Architecture

### StateGraph as Teleological Loop
- **Context:** The teleological loop `perceive → evaluate → revise → plan → filter → act → transition` is the invariant algorithm from Haidemariam (2026).
- **Choice:** Implement the loop as a LangGraph `StateGraph` with 8 nodes + 2 conditional edges. Strategy services (evaluators, planners, etc.) are injected via the state dict.
- **Rationale:** LangGraph provides battle-tested orchestration (checkpointing, streaming, interrupts, subgraphs) without reimplementing these features. The teleological theory layer (domain, services) remains unchanged — only the orchestration layer is replaced.

### TypedDict State (Not Pydantic)
- **Context:** LangGraph supports both TypedDict and Pydantic for state schemas.
- **Choice:** `TeleologicalState(TypedDict, total=False)` with `Annotated[list, operator.add]` for append-only channels.
- **Rationale:** TypedDict is lightweight, requires no extra dependencies, and works well with LangGraph's channel system. `total=False` allows partial state updates from nodes. The `operator.add` reducer enables event accumulation without full state replacement.

### No `from __future__ import annotations` in State/Schema Files
- **Context:** LangGraph uses `get_type_hints(..., include_extras=True)` at runtime to resolve TypedDict annotations.
- **Choice:** Use explicit imports instead of forward references in `state.py` and `multi_agent.py`.
- **Rationale:** `from __future__ import annotations` converts all annotations to strings, which breaks LangGraph's runtime type resolution. This is a known LangGraph/Python 3.11 compatibility constraint.

### GraphBuilder Returns `(app, initial_state)` Tuple
- **Context:** Unlike the old `AgentBuilder.build()` which returned a configured agent, the graph needs both the compiled graph and an initial state dict.
- **Choice:** `GraphBuilder.build()` returns a `(CompiledStateGraph, dict)` tuple.
- **Rationale:** The initial state contains injected strategies (evaluator, planner, etc.) as values in the dict. This is the cleanest way to pass non-serializable callables into the graph without global state.

### Closure Injection Over State-Stored Strategies
- **Context:** Strategies (evaluator, planner, etc.) stored as Python objects in state caused LangGraph checkpointing to fail with `TypeError: Type is not msgpack serializable`.
- **Choice:** `build_teleological_graph()` accepts optional strategy kwargs. When provided, factory functions (`make_evaluate_node()`, etc.) create closure-wrapped node functions. Strategies are no longer stored in state.
- **Rationale:** LangGraph checkpointers (MemorySaver, SqliteSaver) serialize state with msgpack. Non-serializable objects (class instances) must be kept out of state. Closures are the cleanest mechanism — strategies are captured at graph build time, invisible to the serialization layer. Backward compatible: when kwargs are absent, nodes read strategies from state as before.

### Dual-Mode Architecture (LLM + Numeric)
- **Context:** Existing tests and examples depend on numeric mode. Breaking backward compatibility would be disruptive.
- **Choice:** `GraphBuilder` detects mode automatically: `with_model()` → LLM mode (LLMEvaluator, LLMPlanner, LLMReviser, LLMConstraintChecker), `with_objective()` → numeric mode (NumericEvaluator, GreedyPlanner, ThresholdUpdater). Both use the same graph topology.
- **Rationale:** Zero migration cost for existing users. New users get LLM mode by default. The builder's `build()` method wires the correct services based on which configuration methods were called.

---

## LLM Reasoning

### LLM-Driven Node Implementations
- **Context:** The graph structure correctly implements the Haidemariam (2026) loop, but reasoning at each step was mechanical math (Euclidean distance, greedy action selection, boolean predicates). The goal is agents that reason, not compute.
- **Choice:** Rewrite all graph node implementations to delegate to LLM-backed services by default, while preserving numeric mode as fallback. The builder auto-detects mode based on `with_model()` vs `with_objective()`.
- **Rationale:** LLMs bring genuine probabilistic reasoning: multi-hypothesis planning with confidence scores, soft constraint checking with severity, chain-of-thought evaluation, and reflective goal revision. The graph topology stays identical — only node implementations change.

### LangChain `BaseChatModel` as the Universal LLM Interface
- **Context:** A custom `LLMProvider` abstraction with Anthropic/OpenAI/HuggingFace/Generic providers existed previously.
- **Choice:** Use `langchain_core.language_models.BaseChatModel` directly. Deprecate custom providers with `DeprecationWarning`. Provide `LLMProviderToChatModel` bridge for migration.
- **Rationale:** LangChain's model ecosystem (ChatAnthropic, ChatOpenAI, ChatOllama, ChatGoogleGenerativeAI, etc.) is battle-tested and maintained. `with_structured_output()` gives reliable Pydantic schema parsing. No need to maintain a custom provider layer.

### Pydantic Schemas for Structured LLM Output
- **Context:** LLM nodes need reliable structured responses (scores, reasoning, actions, constraint assessments).
- **Choice:** Define Pydantic `BaseModel` schemas (`EvaluationOutput`, `PlanningOutput`, `RevisionOutput`, `ConstraintCheckOutput`) and use `model.with_structured_output(Schema)` for type-safe LLM responses.
- **Rationale:** Structured output via tool-calling/JSON mode is far more reliable than regex parsing. Pydantic validation catches malformed responses. Schemas serve as both prompt documentation and runtime contracts.

### Multi-Hypothesis Planning with Softmax Selection
- **Context:** Numeric planner picks a single greedy action. LLM reasoning should explore multiple approaches.
- **Choice:** `LLMPlanner` generates N candidate hypotheses, each with confidence + reasoning + risk assessment. Softmax over confidences produces a probability distribution for `PolicySpec`.
- **Rationale:** Multiple hypotheses with probabilistic selection enables exploration vs exploitation tradeoff. Users control diversity via `num_hypotheses` and `temperature`. Metadata preserves all hypotheses for observability.

### Natural Language Goals with Optional Numeric Backing
- **Context:** Previously, goals were `ObjectiveVector` (float tuples with direction enums). Goals should be text-first.
- **Choice:** `Goal.description` becomes the primary representation. `success_criteria: list[str]` defines what the LLM evaluates against. `objective: ObjectiveVector | None` remains optional for hybrid use cases.
- **Rationale:** Text goals align with how humans think about objectives. Success criteria give the LLM evaluator clear checkpoints. Numeric backing is still available for environments that produce scalar observations.

### Fail-Closed Constraint Checking + Error Metadata + Timeouts
- **Context:** All LLM services silently swallowed errors. `LLMConstraintChecker` failed open on error (assumed safe), which is dangerous — an LLM outage would bypass all constraint checks.
- **Choice:** (1) `LLMConstraintChecker` now fails closed on error: `return False, "error..."`. (2) `LLMEvaluator` error fallback sets `confidence=0.0` with `metadata={"llm_error": True}`. (3) `LLMPlanner` returns a `noop_fallback` action instead of empty policy. (4) All 4 services accept `timeout: float | None` for `concurrent.futures`-based timeout.
- **Rationale:** Safety-critical components must fail safely. A constraint checker that assumes "safe" on error undermines the entire constraint envelope. Error metadata enables downstream nodes and metrics to detect degraded operation. Timeouts prevent indefinite hangs on LLM API issues.

### Soft Constraint Reasoning (Not Boolean Predicates)
- **Context:** Previously, constraints were `Callable[..., bool]` predicates — either satisfied or not.
- **Choice:** `LLMConstraintChecker` evaluates each constraint with `severity: float [0,1]`, `reasoning: str`, and `suggested_mitigation: str`. Overall safety is a reasoned judgment, not a conjunction of booleans.
- **Rationale:** Real-world constraints have nuance. "Stay within budget" might be 98% satisfied — an LLM can reason about whether the 2% overrun is acceptable. Severity scores enable graduated responses instead of hard stops.

### ReasoningQuality as 8th Metric
- **Context:** 7 metrics from the paper measure numeric agent behavior (persistence, coherence, efficiency, etc.). None measure the quality of LLM reasoning itself.
- **Choice:** Add `ReasoningQuality` metric: composite of presence ratio (% of steps with reasoning), diversity ratio (unique reasoning / total), and average length ratio (normalized reasoning depth).
- **Rationale:** LLM agents that produce repetitive, absent, or shallow reasoning should score lower. This metric differentiates between agents that genuinely reason and those that produce boilerplate. Numeric-mode agents score 0.0 (no reasoning traces).

---

## Goal Management & Feedback

### ConstraintResult as Opt-In Severity Model
- **Context:** `BaseConstraintChecker.check()` returns `(bool, str)` — no severity, no structured detail. The LLM constraint checker internally produces severity scores but they're lost in the return type.
- **Choice:** Added `ConstraintResult` frozen dataclass (`passed`, `message`, `severity`, `checker_name`, `suggested_mitigation`, `metadata`). `check_detailed()` on BaseConstraintChecker returns `ConstraintResult` (default wraps `check()`). `ConstraintPipeline.check_all_detailed()` returns `list[ConstraintResult]`.
- **Rationale:** Opt-in — existing `check()` and `check_all()` are unchanged. `check_detailed()` provides a richer interface for consumers that need severity-based decision-making. Default implementation wraps `check()` so no existing checker needs modification.

### Goal Immutability (Frozen Dataclass)
- **Context:** `Goal.revise()` mutated `self.status = GoalStatus.REVISED` in-place, breaking functional node contracts. Lifecycle methods (`achieve`, `abandon`, `suspend`, `reactivate`) also mutated in-place. Haidemariam (2026) treats G_t → G_{t+1} as producing a NEW goal entity.
- **Choice:** Made `Goal` a frozen dataclass (`@dataclass(frozen=True)`). All lifecycle methods now return new `Goal` instances via `dataclasses.replace()`. `revise()` no longer sets `self.status = REVISED` — the old goal is simply replaced.
- **Rationale:** Paper alignment: goals are immutable values. Frozen dataclasses prevent accidental state corruption in concurrent or checkpointed contexts. All callers already used the return value from `revise()` — removing the side-effect is safe.

### Revision Threshold Semantics
- **Context:** The `should_revise` edge used `abs(score) >= 0.5 or score <= -0.3`, meaning GOOD scores (e.g., 0.8) also triggered revision — counterintuitive and paper-incorrect.
- **Choice:** Changed to `score <= -0.3` only. Good scores indicate the agent is on track and should not trigger revision.
- **Rationale:** Paper says revision happens when evaluation indicates poor performance. A score of 0.8 means "close to goal achieved" — revision would be counterproductive.

### Thread Safety via threading.Lock on Mutable Shared State
- **Context:** `BudgetChecker`, `EvolvingConstraintManager`, `IntentionalGroundingManager`, and `GoalTree` have mutable state accessed during graph execution. Multi-agent and parallel graphs could corrupt this state.
- **Choice:** Added `threading.Lock()` to all four classes, wrapping mutating methods. Recursive/nested calls use internal `_locked` methods to avoid deadlock.
- **Rationale:** Minimal overhead for single-threaded usage. Correctness guarantee for multi-agent/parallel execution. Goal immutability (Phase 2) handles Goal itself — locks handle the containers.

### Class-Level ThreadPoolExecutor for LLM Services
- **Context:** All 4 LLM services created a new `ThreadPoolExecutor` per `invoke` call when timeout was set, causing resource leaks.
- **Choice:** Moved executor creation to `__init__` (one per service instance). Added `shutdown()` method to all services and `LLMNegotiator`.
- **Rationale:** One executor per service instance matches the lifecycle. `shutdown()` enables clean cleanup in long-running processes.

### Observation Enrichment Over Service Signature Changes
- **Context:** LLM services (evaluator, planner, reviser) had no memory of previous steps — they saw only the current snapshot.
- **Choice:** Enrich the `observation` text in `perceive_node` with action results, eval trends, and goal revision counts. All LLM services read `{observation}` in their prompts, so enriching observation = enriching all prompts automatically.
- **Rationale:** Zero changes to `BaseGoalUpdater` or any of its 7 concrete subclasses. No new parameters on any service. The enrichment is invisible to numeric-mode agents (no feedback on step 1). Fully backward compatible.

### WorkingMemory as Graph-Layer Utility
- **Context:** LLM examples had no `perceive_fn`, so the agent reasoned in a vacuum with a static "Awaiting observation" message.
- **Choice:** `WorkingMemory` provides `perceive` and `record` callbacks that accumulate agent actions as a perception source. It lives in `graph/` (not `services/`) because it's a graph-layer concern.
- **Rationale:** Closes the perception-action loop for LLM agents that don't have external APIs. The `record()` callback accepts 1 or 2 args to match `act_node`'s `inspect.signature` detection for constraint-conditioned transitions.

### action_feedback as Append-Only State Channel
- **Context:** `act_node` captured tool results in events only; `perceive_node` never saw them.
- **Choice:** New `action_feedback: Annotated[list, operator.add]` channel in `TeleologicalState`. Each entry has action name, tool_name, result, step, timestamp.
- **Rationale:** Append-only channels (like `events`, `eval_history`) are the established pattern for accumulating data across iterations. Keeping feedback separate from events makes it easy for `perceive_node` to read recent results without filtering.

### Three-Tier TC Metric Computation
- **Context:** TC was using normalized mean of eval scores, not the correlation formula from the paper.
- **Choice:** Three tiers: (1) Pearson correlation between goal-change magnitudes and eval scores (when `goal_values` available), (2) responsive-revision proxy (when revisions present), (3) mean-score fallback.
- **Rationale:** Primary tier directly implements the paper's formula. Proxy and fallback tiers ensure graceful degradation when data is sparse.

### Decorator Pattern for SelfModelingEvaluator
- **Context:** Paper pattern (b) requires a self-model that predicts evaluation scores and detects surprise.
- **Choice:** `SelfModelingEvaluator` wraps any `BaseEvaluator` (Decorator pattern), adding linear regression prediction, surprise EMA, and confidence adjustment.
- **Rationale:** Any evaluator gains self-modeling by wrapping — no inheritance changes needed. R-squared gate prevents unreliable models from triggering revisions.

### Active Inference via Expected Free Energy
- **Context:** Paper Section 5.4.3 calls for active inference in goal revision.
- **Choice:** `ActiveInferenceUpdater` decomposes free energy into pragmatic (goal-state distance) and epistemic (confidence-based) components, revising when both exceed thresholds.
- **Rationale:** Dual threshold prevents spurious revisions. Blend of pragmatic and epistemic components balances exploitation and exploration.

---

## Multi-Agent Coordination

### EvolvingConstraintManager Wired via Builder
- **Context:** `EvolvingConstraintManager` existed as a standalone service but was never connected to the graph — a dead abstraction.
- **Choice:** `GraphBuilder.with_evolving_constraints(manager)` sets the manager. `build_teleological_graph(enable_evolving_constraints=True)` wires an `evolve_constraints_node` between `check_constraints` and `plan`. The node records violations, calls `manager.step()`, and emits evolution reasoning to the trace.
- **Rationale:** Wiring through the builder + graph keeps the integration consistent with existing patterns (grounding_manager, knowledge_store). The evolve node is optional and only added when enabled — zero overhead for existing graphs.

### LLM Negotiation as 3-Phase Protocol
- **Context:** Multi-agent negotiation was limited to numeric averaging.
- **Choice:** `LLMNegotiator` implements Propose → Critique → Synthesize protocol operating on `agent_results` dicts.
- **Rationale:** Operating on result dicts (not BaseAgent objects) bridges graph and service layers cleanly. The 3-phase structure mirrors dialogue-based negotiation theory.

### Parallel Multi-Agent via LangGraph Send API
- **Context:** Sequential agent execution limits throughput in multi-agent scenarios.
- **Choice:** `parallel=True` uses LangGraph `Send` API for fan-out/fan-in with custom `_merge_agent_results` reducer.
- **Rationale:** LangGraph's native Send API handles parallelism correctly, including state merging. Sequential mode remains default for deterministic behavior.

### Multi-Agent via Subgraph Composition
- **Context:** Multi-agent coordination needs each agent to run its own teleological loop, then negotiate.
- **Choice:** `build_multi_agent_graph()` creates per-agent nodes that internally invoke `build_teleological_graph()` as subgraphs. A negotiation node runs between rounds.
- **Rationale:** LangGraph subgraph composition is the natural pattern. Each agent's subgraph is fully independent, enabling different configurations per agent.

### Rule-Based Grounding Fallback
- **Context:** Intentional grounding needs LLM for assessment, but should work without one.
- **Choice:** `IntentionalGroundingManager` supports both LLM-based and rule-based grounding. Rule-based fallback triggers when high-priority directives exceed threshold.
- **Rationale:** Enables grounding in numeric-mode agents without an LLM. LLM mode provides richer assessment when available.

### Intentional State Mapping Bridge via Node Factories
- **Context:** Intentional State agent exists but was disconnected from LangGraph.
- **Choice:** Factory functions (`make_intentional_*_node`) wrap IntentionalStateAgent methods as LangGraph nodes. `build_intentional_teleological_graph()` compiles a full graph with intentional-state-augmented nodes.
- **Rationale:** Factory functions keep the bridge lightweight and composable. Intentional state methods augment (not replace) standard nodes — falling back to standard behavior when the agent doesn't trigger.

### BDI → Intentional State Mapping Rename
- **Context:** The "BDI" label carries philosophical baggage from the Bratman/Rao-Georgeff tradition that doesn't fully map to Haidemariam's framework. Senior review flagged this as confusing positioning.
- **Choice:** Renamed all BDI-labeled code to "Intentional State Mapping" (ISM). `BDIAgent` → `IntentionalStateAgent`, `bdi_bridge.py` → `intentional_bridge.py`, all factory functions and graph builders renamed. Old files kept as thin backward-compatibility shims re-exporting new names.
- **Rationale:** ISM accurately describes what the bridge does — mapping intentional states (beliefs, desires, intentions) to LangGraph nodes — without claiming adherence to a specific philosophical tradition. Backward-compat shims prevent breaking existing code.

---

## Domain & Service Patterns

### Domain-Driven Design with Bounded Contexts
- **Context:** Framework implementing Haidemariam (2026) Synthetic Teleology theory with complex interactions between goals, agents, evaluation, planning, and coordination.
- **Choice:** Five bounded contexts: Teleology Domain, Agency Domain, Coordination Domain, Measurement Domain, Infrastructure.
- **Rationale:** Clean separation of concerns. Each context has its own entities, value objects, and events. Integration happens through domain events on the EventBus, not direct coupling.

### Frozen Dataclasses for Value Objects
- **Context:** Goals, states, signals, and specifications need to be passed between components without risk of mutation.
- **Choice:** All value objects in `domain/values.py` are frozen `@dataclass(frozen=True)` with `__post_init__` validation.
- **Rationale:** Immutability prevents accidental state corruption. Goal revisions produce new objects rather than mutating existing ones, maintaining a clear history. Frozen dataclasses are hashable by default.

### Event-Driven Architecture with Synchronous EventBus
- **Context:** The agentic loop needs to communicate with metrics, visualization, and logging without tight coupling.
- **Choice:** `EventBus` with synchronous dispatch (thread-safe via `threading.Lock`). Separate `AsyncEventBus` for async contexts.
- **Rationale:** Synchronous dispatch keeps the default case simple and deterministic. Events are immutable (frozen dataclasses), so handlers cannot corrupt state. The `EventStore` enables replay for debugging.

### Strategy Pattern for Evaluators, Updaters, Planners
- **Context:** Each of these services has multiple implementations (numeric vs LLM, threshold vs gradient vs hierarchical, greedy vs stochastic).
- **Choice:** ABC base classes with interchangeable implementations. Agents receive strategies via constructor injection.
- **Rationale:** Open/Closed principle — new evaluation/update/planning strategies can be added without modifying existing code. The `AgentBuilder` makes wiring strategies ergonomic.

### Chain of Responsibility for Goal Revision and Constraints
- **Context:** Goal revision and constraint checking involve multiple independent checks that should run in sequence with optional short-circuit.
- **Choice:** `GoalUpdaterChain` and `ConstraintPipeline` implement CoR. Each handler decides whether to process and/or pass to the next.
- **Rationale:** Composable pipelines. `ConstraintPipeline` supports `fail_fast=True` for safety-critical constraints. Order of handlers matters and is explicit.

### Decorator Pattern for ReflectiveEvaluator
- **Context:** Need to add self-model/drift-detection behavior on top of any existing evaluator without modifying it.
- **Choice:** `ReflectiveEvaluator` wraps any `BaseEvaluator`, tracks score history, computes EMA drift, and adjusts confidence.
- **Rationale:** Any evaluator (numeric, LLM, composite) can gain reflective capabilities by wrapping. No inheritance needed — true Decorator pattern.

### Template Method for Agentic Loop
- **Context:** The loop `perceive → evaluate → revise → plan → filter → act → transition` is the invariant algorithm. Individual steps vary.
- **Choice:** `BaseAgenticLoop.run()` is the fixed template. Subclasses override `before_step()`, `after_step()`, `on_error()`. Strategy methods live on the agent.
- **Rationale:** The 10-step sequence is fundamental to the theory. Making it a template method ensures all agents follow the same loop structure while allowing customization of individual steps.

### Finite State Machine for Agent Lifecycle
- **Context:** Agents progress through well-defined phases (perceive, evaluate, revise, plan, act, reflect).
- **Choice:** `AgentState` enum with `_VALID_TRANSITIONS` dict. `_transition_to()` validates transitions.
- **Rationale:** Prevents invalid state sequences (e.g., acting before evaluating). The FSM is fundamental to the teleological loop structure. Invalid transitions raise `ValueError` immediately.

### Intentional State Agent as Alternative Reference Implementation
- **Context:** Need to validate the framework against classical belief-desire-intention agent architecture.
- **Choice:** `IntentionalStateAgent(BaseAgent)` mapping Beliefs→S_t, Desires→G_t, Intentions→π_t.
- **Rationale:** Intentional state mapping is the most widely understood agent architecture. Having it alongside TeleologicalAgent demonstrates the framework's flexibility and enables direct comparison in benchmarks.

### Fluent Builder for Agent Construction
- **Context:** Fully configured agents require evaluator + updater + planner + constraints + event bus + optional LLM provider.
- **Choice:** `AgentBuilder` with fluent API: `.with_goal()`, `.with_evaluator()`, `.with_planner()`, etc.
- **Rationale:** Complex object construction without constructor explosion. Builder validates completeness on `build()`. Factory methods provide shortcuts for common configurations.

---

## Project Structure & Testing

### `src/` Layout with Hatchling Build
- **Context:** Need a clean Python package structure that supports editable installs and modern tooling.
- **Choice:** `src/synthetic_teleology/` layout with hatchling as build backend.
- **Rationale:** `src/` layout prevents accidental imports from the project root. Hatchling is lightweight and well-supported by uv.

### Separate Conceptual vs Production Examples
- **Context:** A flat `examples/` directory mixed different kinds of examples. Most users only need 2-3 to understand the framework, while advanced users want real-world production patterns.
- **Choice:** Split into `examples/conceptual/` (14 core demos) and `examples/production/` (self-contained mini-packages with custom strategies, domain models, and CLI entry points).
- **Rationale:** Conceptual examples stay simple and focused on framework APIs. Production examples demonstrate the full Strategy pattern — custom evaluators, planners, and constraint checkers for real domains — without cluttering the learning path.

### Hybrid Mock Pattern for Production Examples
- **Context:** Production examples need to run without API keys (CI, demos, learning) while also supporting real LLM mode. Fully mocking all 5 LLM services (evaluator, planner, reviser, constraints, negotiation) creates fragile mock response ordering that breaks when node execution order varies.
- **Choice:** Custom evaluator (reads simulated environment state → deterministic scores) + LLM planner/reviser via `MockStructuredChatModel`. Simulated tools mutate shared domain state. Custom `BaseConstraintChecker` subclasses instead of `LLMConstraintChecker` (except multi-agent example). Real LLM replaces mock when API key present; custom evaluator and tools remain.
- **Rationale:** Custom evaluator eliminates EvaluationOutput from the mock sequence, halving mock complexity. Deterministic scores ensure predictable revision triggers (score <= -0.3). Custom constraint checkers avoid interleaving ConstraintCheckOutput in the mock. The mock only serves planner + reviser, giving a simple linear response list. Real LLM mode is a single toggle — swap the model, everything else stays.

### Production Examples as Mini-Packages
- **Context:** A single-file production example would be too large and hard to navigate.
- **Choice:** Each production agent has its own package: `models.py` (domain), `market_data.py`/`crm.py` (data layer), `strategies.py` (custom evaluator/planner/constraints), `agent.py` (graph wiring), `main.py` (CLI).
- **Rationale:** Mirrors how a user would structure a real agent project. Demonstrates clean separation of concerns. The data layer provides both a real API client and a simulated fallback, so examples work out of the box without API keys.

### Conceptual Examples Aligned 1:1 with Paper Concepts
- **Context:** Early examples covered basic loop, multi-agent, and constraints — but left many major concepts from Haidemariam (2026) undemonstrated.
- **Choice:** 14 examples that comprehensively demonstrate every major theoretical concept. No new framework code — examples use existing classes.
- **Rationale:** Examples are the primary learning path for users. Covering all paper concepts ensures the framework's full capabilities are discoverable.

### Self-Contained LLM Examples
- **Context:** `MockStructuredChatModel` lived in `tests/helpers/mock_llm.py`. Examples couldn't import from tests, so LLM examples required real API keys to run.
- **Choice:** Create `synthetic_teleology.testing` public package with `MockStructuredChatModel`. Re-export from `tests/helpers/mock_llm.py` for backward compatibility.
- **Rationale:** Examples should be self-contained and runnable without API keys. A public testing module also helps downstream users write tests for their own agents.

### Examples Default to Mock, Optionally Use Real LLM
- **Context:** LLM examples previously exited with `sys.exit(1)` if no API key was set.
- **Choice:** Examples auto-detect API keys: if present, use real LLM; otherwise, use `MockStructuredChatModel` with pre-configured realistic responses that exercise the full teleological loop.
- **Rationale:** Self-contained examples are the primary learning path. Users can see the full feedback loop working immediately, then switch to a real LLM when ready. Mock responses are interleaved in the exact call order so the agent's behavior is deterministic and educational.

---

## Measurement

### 7 Metrics from Haidemariam (2026) Theory
- **Context:** The paper defines specific metrics for evaluating teleological agents.
- **Choice:** GoalPersistence, TeleologicalCoherence, ReflectiveEfficiency, Adaptivity, NormativeFidelity, InnovationYield, LyapunovStability.
- **Rationale:** Direct mapping to the theoretical framework. Each metric is a `BaseMetric` subclass with `compute()`, `validate()`, `describe()`. The `MetricsEngine` composes all of them.

### Template Method Pattern for Benchmarks
- **Context:** Need a reusable framework for running different benchmark scenarios.
- **Choice:** `BaseBenchmark` ABC: `setup() → run_scenario(seed) → collect_metrics(log) → teardown()`. The `run()` method orchestrates N independent runs.
- **Rationale:** Each benchmark defines its own scenario while sharing orchestration, error handling, and aggregation logic.

### Composite Pattern for BenchmarkSuite
- **Context:** Need to run multiple benchmarks together and aggregate results.
- **Choice:** `BenchmarkSuite` wraps an `OrderedDict[str, BaseBenchmark]`.
- **Rationale:** Composability — suites can contain any mix of benchmarks, results keyed by name.

---

## Infrastructure

### KnowledgeStore Connected to Observation Enrichment
- **Context:** `KnowledgeStore` existed but was never queried by the graph nodes — another dead abstraction.
- **Choice:** (1) `_build_enriched_observation()` queries `knowledge_store.query_recent(300)` and appends the last 5 entries to observation text. (2) `reflect_node` writes reflection data (`eval_score`, `stop_reason`, `goal_id`) to the knowledge store via `put()`.
- **Rationale:** Both integrations are best-effort (try/except) — a KnowledgeStore failure never breaks the loop. Reading in observation enrichment gives LLM services access to accumulated knowledge. Writing in reflection creates a growing corpus of agent self-assessment that future perception steps can leverage.

### Deferred: Agent API Consolidation
- **Context:** The codebase has both a class hierarchy (`BaseAgent` → `TeleologicalAgent`) and a graph-based API (`GraphBuilder`). The review flagged this as confusing.
- **Choice:** Defer to a future version. Document the graph API as primary, class hierarchy as adapter.
- **Rationale:** Consolidation affects 40+ files and would destabilize the codebase. The graph API is already the recommended path. The class hierarchy serves as a compatibility layer and reference implementation.

### Deferred: Streaming-First Design
- **Context:** The review noted that streaming is bolted on rather than being a first-class concern.
- **Choice:** Defer to a future version. Add streaming helpers to the builder incrementally.
- **Rationale:** A streaming-first redesign would affect all examples and the graph builder architecture. Better approached incrementally: add `.stream()` convenience methods, streaming callbacks in the builder, and SSE adapters.

### Lazy Imports via `__getattr__` in `llm/__init__.py`
- **Context:** Concrete providers depend on optional packages (anthropic, openai, httpx, transformers/torch).
- **Choice:** Module-level `__getattr__` that lazy-imports concrete provider classes on first attribute access.
- **Rationale:** Users who only need one provider don't need all others installed. Import errors are deferred to actual usage.

### Registry-Based Factory with Auto-Discovery
- **Context:** Need a single entry point for creating any LLM provider by name.
- **Choice:** `LLMProviderFactory` with pre-registered lazy constructors and `register()` for custom providers.
- **Rationale:** Decouples provider creation from usage. Declarative configuration support enables config-file-driven setup.

### Strategy + Fallback Chain in ProviderRouter
- **Context:** Production systems need resilience when an LLM provider goes down.
- **Choice:** Priority-ordered routing with health tracking (consecutive failure threshold) and time-based cooldown re-enabling.
- **Rationale:** Automatic failover without manual intervention. Cooldown prevents hammering a down provider while eventually retrying.

### Exponential Backoff for API Providers
- **Context:** API-based providers can hit rate limits.
- **Choice:** Built-in retry with exponential backoff in each provider's `generate()` method.
- **Rationale:** Avoids requiring external retry middleware. Each provider handles its own rate limit format.

### Optional Rich/Matplotlib Dependencies
- **Context:** Console dashboard and plotting are nice-to-have but shouldn't be required dependencies.
- **Choice:** Rich and matplotlib are optional (`[viz]` extra). All presentation modules gracefully degrade to plain-text output when dependencies are missing.
- **Rationale:** Core framework stays lightweight. `presentation/console.py` renders plain-text tables if Rich is unavailable. `presentation/plots.py` raises ImportError only when plotting functions are called.

---

## Environment Layer

### PyTorch-Style state_dict / load_state_dict
- **Context:** Environments had no serialization mechanism — saving and restoring state required ad-hoc code.
- **Choice:** Added `state_dict()` and `load_state_dict()` to `BaseEnvironment`, mirroring PyTorch's `nn.Module` serialization API. Subclasses implement `_state_dict_impl()` / `_load_state_dict_impl()` hooks.
- **Rationale:** PyTorch conventions are well-understood by ML practitioners. Template Method pattern keeps BaseEnvironment clean while allowing each environment to serialize its own state. All 4 environments implement the hooks.

### EnvironmentWrapper (Composable Decorator Pattern)
- **Context:** Users need to modify environment behavior (add noise, track history, enforce quotas) without subclassing.
- **Choice:** `EnvironmentWrapper(BaseEnvironment)` base class (like `gym.Wrapper`) that delegates all methods to a wrapped environment. Three concrete wrappers: `NoisyObservationWrapper`, `HistoryTrackingWrapper`, `ResourceQuotaWrapper`.
- **Rationale:** Decorator pattern enables composable modifications. Multiple wrappers can be stacked: `ResourceQuotaWrapper(NoisyObservationWrapper(env))`. The `unwrapped` property always returns the innermost concrete environment.

### Bounded Accumulation Channels
- **Context:** Append-only state channels (`events`, `eval_history`, `action_history`, `reasoning_trace`, `action_feedback`, `goal_history`) grow without bound, causing memory bloat in long-running agents.
- **Choice:** `make_bounded_state(max_history)` factory creates a `BoundedTeleologicalState` TypedDict where all append channels use a `_make_bounded_add(max_size)` reducer that keeps only the last N items. `GraphBuilder.with_max_history(n)` integrates this. `build_teleological_graph()` accepts optional `state_schema`.
- **Rationale:** Opt-in — default `TeleologicalState` is still unbounded. Bounded variant drops oldest entries (FIFO), which is the natural choice for history channels. Factory function approach avoids class proliferation while keeping LangGraph's runtime type resolution working correctly.
