# Changelog

All notable changes to the Synthetic Teleology Framework.

## [1.1.0] — 2026-02-11

### Major: Full Haidemariam (2026) Theoretical Alignment

Closes all 15 identified gaps between the library and the paper, bringing alignment from A- (91%) to A+/98%.

#### Phase 1: Fixed Metrics + Empowerment
- **Teleological Coherence (TC)** — rewritten with 3-tier computation: Pearson correlation (primary), responsive-revision proxy, legacy mean-score fallback
- **Innovation Yield (IY)** — attribution formula: `0.6 * novelty_ratio + 0.4 * quality_improvement` when revisions present, unique/total fallback otherwise
- **Empowerment** — new information-theoretic metric: `I(A; S'|S)` mutual information between actions and state transitions (opt-in via `engine.add_metric(Empowerment())`)
- `AgentLogEntry.goal_values` field added for correlation-based TC

#### Phase 2: Self-Modeling + Active Inference
- **`SelfModelingEvaluator`** — Decorator wrapping any evaluator; adds linear regression self-model, surprise EMA tracking, confidence adjustment, `recommends_goal_edit` property
- **`ActiveInferenceUpdater`** — Expected free energy decomposition: pragmatic + epistemic components; triggers revision when free energy and prediction error exceed thresholds
- New enum values: `RevisionReason.ACTIVE_INFERENCE`, `RevisionReason.DIALOGUE`

#### Phase 3: Goal Provenance + Audit Trail + Knowledge Store
- **`GoalProvenance`** value object (origin, source_agent_id, source_description, timestamp)
- **`GoalOrigin`** enum: DESIGN, USER, NORMATIVE, ENDOGENOUS, NEGOTIATED, PROPAGATED
- **`GoalAuditTrail`** — serializable audit trail with `record()`, `query()`, `to_json()`/`from_json()`
- **`KnowledgeStore`** — thread-safe shared key-value store with tag/source queries, EventBus integration
- **`KnowledgeUpdated`** domain event
- Wired into graph: `TeleologicalState` fields, `GraphBuilder.with_knowledge_store()`, `with_audit_trail()`

#### Phase 4: Constraint-Conditioned Transitions + Human-in-the-Loop
- `act_node` detects 2-arg `transition_fn(action, constraints_context)` via `inspect.signature`
- `build_teleological_graph()` accepts `interrupt_before`/`interrupt_after` for human-in-the-loop
- `GraphBuilder.with_human_approval(before=[], after=[])`
- `ground_goal_node` for intentional grounding

#### Phase 5: LLM Multi-Agent Negotiation + Parallel Execution
- **`LLMNegotiator`** — 3-phase protocol: Propose → Critique → Synthesize; operates on `agent_results` dict
- **`build_multi_agent_graph()`** — new params: `negotiation_model`, `max_dialogue_rounds`, `parallel`
- Parallel execution via LangGraph `Send` API with custom `_merge_agent_results` reducer
- `shared_direction` prepended to agent goals in LLM mode

#### Phase 6: Evolving Constraints + Distributed Grounding
- **`EvolvingConstraintManager`** — LLM-based co-evolutionary constraint management; analyzes violation patterns, proposes additions/removals/modifications
- **`IntentionalGroundingManager`** — accumulates external directives (user, normative, negotiated); LLM or rule-based grounding assessment
- **`GoalSource`** enum, **`ExternalDirective`** model

#### Phase 7: BDI-LangGraph Bridge + Benchmark Upgrades
- **`bdi_bridge.py`** — `make_bdi_perceive_node`, `make_bdi_revise_node`, `make_bdi_plan_node`, `build_bdi_teleological_graph()`
- BDI belief updating, desire reconsideration, intention reuse as LangGraph nodes
- **Distribution Shift** — `shift_mode` (sudden/gradual), `transition_steps`, `shift_type` params
- **Conflicting Objectives** — `tradeoff_step`, `tradeoff_multiplier`, added InnovationYield metric
- **Negotiation** — `strategy` param (consensus/voting/auction), added GoalPersistence + Adaptivity metrics
- **Knowledge Synthesis** — `critic_interval` param, added Empowerment metric

#### New Files (8)
- `src/.../measurement/metrics/empowerment.py`
- `src/.../services/llm_negotiation.py`
- `src/.../services/evolving_constraints.py`
- `src/.../services/goal_grounding.py`
- `src/.../services/audit_trail.py`
- `src/.../infrastructure/knowledge_store.py`
- `src/.../graph/bdi_bridge.py`
- 5 new test files

#### Stats
- **617 tests** (498 → 617, +119 new), all passing
- All 15 paper gaps closed
- Lint clean on all new files

---

## [1.0.0] — 2026-02-10

### Major: LLM-First Probabilistic Architecture

Complete rewrite transforming the framework from deterministic numeric computation to LLM-driven probabilistic reasoning, while preserving full backward compatibility with numeric mode.

#### New: LLM Reasoning Services
- **`services/llm_evaluation.py`** — `LLMEvaluator`: structured output evaluation with chain-of-thought reasoning, per-criterion scores, and confidence
- **`services/llm_planning.py`** — `LLMPlanner`: multi-hypothesis planning (N candidates), softmax probability distribution, tool-aware action proposals
- **`services/llm_revision.py`** — `LLMReviser`: LLM-driven goal revision with text-based description/criteria changes and optional numeric adjustments
- **`services/llm_constraints.py`** — `LLMConstraintChecker`: soft constraint reasoning with severity scores, mitigation suggestions, and detailed assessments

#### New: Dual-Mode GraphBuilder
- `GraphBuilder.with_model(model)` — triggers LLM mode with auto-defaulting to LLM services
- `GraphBuilder.with_goal(description, criteria=[...])` — natural language goals with success criteria
- `GraphBuilder.with_tools(*tools)` — LangChain tools for agent actions
- `GraphBuilder.with_constraints(*constraints)` — natural language constraints
- `GraphBuilder.with_num_hypotheses(n)` — multi-hypothesis planning configuration
- `GraphBuilder.with_temperature(t)` — LLM sampling temperature
- Numeric mode preserved: `.with_objective(values)` still works as before

#### New: Prebuilt Constructors
- `create_llm_agent(model, goal, ...)` — one-liner for LLM agents
- `create_numeric_agent(target_values, perceive_fn, ...)` — renamed legacy constructor

#### Modified: Domain Layer
- `Goal` — added `success_criteria: list[str]`, `priority: float`, text-based `revise()` with `new_description`/`new_criteria` kwargs
- `EvalSignal` — added `reasoning: str`, `criteria_scores: Mapping[str, float]`
- `ActionSpec` — added `description: str`, `tool_name: str | None`, `reasoning: str`
- `StateSnapshot` — added `observation: str`, `context: Mapping[str, Any]`, `values` default `()`
- New value object: `Hypothesis(actions, confidence, reasoning, expected_outcome, risk_assessment)`
- New enum value: `RevisionReason.EVALUATION_FEEDBACK`

#### Modified: Graph Layer
- `TeleologicalState` — new fields: `observation`, `hypotheses`, `selected_plan`, `model`, `tools`, `num_hypotheses`, `reasoning_trace` (append-reducer), `constraint_assessments`
- All 8 nodes — dual-mode support (LLM reasoning traces + numeric fallback)
- `MultiAgentState` — new fields: `model`, `tools`, `shared_direction`, `reasoning_trace`
- `AgentConfig` — accepts `Union[Goal, str]` goal, per-agent `model`, `tools`, `criteria`, `constraints`

#### Modified: Measurement Layer
- `AgentLogEntry` — added `reasoning: str`, `hypotheses_count: int`
- New metric: `ReasoningQuality` — measures presence, diversity, and depth of LLM reasoning traces

#### Modified: Infrastructure
- `LLMProvider` — deprecated with `DeprecationWarning` in favor of `BaseChatModel`
- New: `LLMProviderToChatModel` bridge (`infrastructure/llm/langchain_bridge.py`)

#### New: Tests (49 new)
- `tests/services/test_llm_evaluation.py` — 6 tests
- `tests/services/test_llm_planning.py` — 12 tests (incl. softmax)
- `tests/services/test_llm_revision.py` — 5 tests
- `tests/services/test_llm_constraints.py` — 5 tests
- `tests/graph/test_builder_llm.py` — 14 tests
- `tests/graph/test_llm_graph.py` — 7 tests (full integration + metric)
- `tests/helpers/mock_llm.py` — `MockStructuredChatModel` for testing

#### New: Examples
- `11_llm_quickstart.py` — LLM agent with natural language goals
- `12_llm_tools.py` — Agent with LangChain tools
- `13_llm_multi_agent.py` — Multi-agent LLM coordination
- `14_llm_metrics.py` — ReasoningQuality metric demo (no API key needed)

#### Stats
- **498 tests** (449 existing + 49 new), all passing
- **14 conceptual examples** (10 numeric + 4 LLM)
- Full backward compatibility — all v0.2.x code works unchanged

---

## [0.2.2] — 2026-02-10

### Expanded: Conceptual examples covering full Haidemariam (2026) theory

Added 7 new conceptual examples (04–10) that demonstrate every major theoretical concept from the Synthetic Teleology paper. No new framework code — only new example scripts.

#### New examples
- `04_evaluation_strategies.py` — NumericEvaluator, CompositeEvaluator, ReflectiveEvaluator with drift detection
- `05_goal_revision.py` — ThresholdUpdater, GradientUpdater, UncertaintyAwareUpdater, GoalUpdaterChain
- `06_planning_strategies.py` — GreedyPlanner, StochasticPlanner (temperature effects), HierarchicalPlanner decomposition
- `07_hierarchical_goals.py` — GoalTree construction, coherence validation, revision propagation, HierarchicalUpdater
- `08_environments.py` — ResourceEnvironment (scarcity/regeneration), ResearchEnvironment (knowledge synthesis), perturbation injection
- `09_metrics_measurement.py` — All 7 metrics (GP, TC, RE, AD, NF, IY, LS), MetricsEngine, AgentLog, MetricsReport
- `10_ethical_constraints.py` — EthicalChecker with custom predicates, ConstraintPipeline (fail_fast), PolicyFilter

#### Paper concepts now demonstrated
- Delta(G_t, S_t) evaluation function, composite evaluation, reflective self-model
- G_t → G_{t+1} revision dynamics: threshold, gradient, uncertainty, chain of responsibility
- pi_t policy generation: deterministic, stochastic (softmax), hierarchical decomposition
- Hierarchical goal structure: GoalTree, coherence, propagation, parent regularization
- Environment types: resource scarcity, knowledge synthesis, perturbation injection
- All 7 teleological metrics with measurement engine and reporting
- Normative constraint envelope E_t: ethical predicates, policy filtering, fail-fast pipeline

---

## [0.2.1] — 2026-02-10

### Restructured: Examples directory

Replaced the flat 16-file examples directory (8 legacy + 8 LangGraph) with a clean `conceptual/` vs `production/` split.

#### `examples/conceptual/` — 3 core framework demos
- `01_basic_loop.py` — Basic StateGraph `.invoke()` with GraphBuilder
- `02_multi_agent.py` — Two agents + ConsensusNegotiator subgraphs
- `03_constraints.py` — SafetyChecker + BudgetChecker with streaming

#### `examples/production/polymarket_trader/` — Prediction market trading agent
- Custom `PortfolioEvaluator` scoring conviction alignment
- `TradingPlanner` with Kelly criterion position sizing
- `RiskChecker` (max position, exposure, drawdown) + `CapitalChecker`
- Real Polymarket CLOB API client + simulated random-walk fallback
- CLI: `python -m examples.production.polymarket_trader.main`

#### `examples/production/sales_sdr/` — Sales development agent
- Custom `PipelineEvaluator` scoring funnel health
- `OutreachPlanner` with lead prioritization and channel selection
- `ContactFrequencyChecker` + `DailyLimitChecker`
- Simulated CRM with realistic response probability model + HubSpot API fallback
- CLI: `python -m examples.production.sales_sdr.main`

### Deleted
- 13 example files (8 legacy v0.1.0 + 5 non-core LangGraph examples)

---

## [0.2.0] — 2026-02-10

Major release: **LangGraph migration**. The custom agentic loop is replaced by a LangGraph StateGraph while preserving all existing domain/service abstractions.

### New: LangGraph Layer (`graph/`)

- **`graph/state.py`** — `TeleologicalState` TypedDict with append-reducer channels for events, goal_history, eval_history, action_history.
- **`graph/nodes.py`** — 8 pure node functions: perceive, evaluate, revise, check_constraints, plan, filter_policy, act, reflect. Each delegates to existing service classes.
- **`graph/edges.py`** — Conditional routing: `should_continue()` (loop/end), `should_revise()` (revise/skip).
- **`graph/graph.py`** — `build_teleological_graph()`: wires nodes and edges into a compiled StateGraph.
- **`graph/builder.py`** — `GraphBuilder` fluent API mirroring `AgentBuilder` but producing `(compiled_graph, initial_state)`.
- **`graph/prebuilt.py`** — One-liner constructors: `create_teleological_agent()`, `create_llm_teleological_agent()`, `create_react_teleological_agent()`.
- **`graph/multi_agent.py`** — `build_multi_agent_graph()`: per-agent subgraphs with negotiation rounds. `AgentConfig` dataclass, `MultiAgentState` TypedDict.
- **`graph/streaming.py`** — Stream event formatters: `format_stream_events()`, `collect_stream_events()`, `stream_to_agent_log_entries()`.

### Modified

- **`pyproject.toml`** — Version 0.2.0. Added `langgraph>=0.2`, `langchain-core>=0.3` as core deps. Added `llm-anthropic-lc`, `llm-openai-lc` optional groups. Added `langgraph-cli` to dev deps.
- **`__init__.py`** — Version 0.2.0. Exports `build_teleological_graph`, `GraphBuilder`, `TeleologicalState` at top level.
- **`services/loop.py`** — `SyncAgenticLoop` and `AsyncAgenticLoop` now emit `DeprecationWarning` on init.
- **`agents/factory.py`** — `AgentBuilder.build_graph()` method added. `AgentFactory.create_teleological_graph()` static method added.
- **`infrastructure/config.py`** — `GraphConfig` dataclass added with `max_steps`, `goal_achieved_threshold`, `enable_checkpointing`, `stream_mode`.
- **`langgraph.json`** — LangGraph Platform configuration.

### Examples (rewritten for LangGraph)

- `01_langgraph_basic_loop.py` — Basic StateGraph `.invoke()` with `GraphBuilder`
- `02_llm_goal_directed_agent.py` — LLM agent with streaming
- `03_multi_agent_negotiation.py` — Two agents + ConsensusNegotiator subgraphs
- `04_human_in_the_loop.py` — Custom review node for human approval
- `05_hierarchical_goals.py` — GoalTree + per-leaf subgraphs
- `06_benchmark_measurement.py` — Stream events → measurement bridge
- `07_react_research_agent.py` — ReAct agent with tool metadata
- `08_constraint_aware_planning.py` — SafetyChecker + BudgetChecker streaming

### Tests

- **69 new tests** in `tests/graph/`: test_state, test_nodes, test_edges, test_graph, test_builder, test_prebuilt, test_multi_agent, test_streaming.
- **449 total tests** (380 original + 69 new), all passing.
- Parity test: graph produces comparable results to legacy `SyncAgenticLoop`.

---

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
