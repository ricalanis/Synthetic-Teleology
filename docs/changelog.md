# Changelog

All notable changes to the Synthetic Teleology Framework.

## Unreleased

### Normalize Production Examples

Standardized all 5 production examples for consistency (14 files modified):

- **`_get_model()` pattern**: All 5 examples now return `None` (not mock), caller uses `_get_model() or _build_mock_model()`. Fixed investment_thesis, learning_curriculum (returned mock directly), data_pipeline_fixer (temperature 0.3 → 0.5).
- **`__init__.py` exports**: All 5 now export their `build_*` function.
- **KnowledgeStore + AuditTrail**: Added to data_pipeline_fixer (had neither) and learning_curriculum (had KS but no AT). Both now seed KS and wire AT.
- **Mode detection**: Added `LIVE LLM` vs `SIMULATED` banner to data_pipeline_fixer and learning_curriculum main.py (competitive_research, investment_thesis, deployment_coordinator already had it).
- **Verbose output**: competitive_research main.py now shows step-by-step eval/action/feedback when `--verbose` is passed (flag existed but was unused).
- **Private attribute access**: investment_thesis main.py replaced `audit_trail._entries` and `knowledge_store._entries` with public API (`audit_trail.entries`, `knowledge_store.keys()`/`.get()`).
- **README diagram**: Added ASCII architecture diagram to investment_thesis README.md.

### 5 Production Examples (30 new files)

Five production-grade examples demonstrating the framework's LLM-first features in realistic, long-running scenarios. All use the hybrid mock pattern: custom evaluator for deterministic scoring + MockStructuredChatModel for planner/reviser. Self-contained (no API key required); set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` for real LLM mode.

1. **`competitive_research/`** — Competitive Research Analyst (18 steps, goal revision at pivot discovery)
   - Features: tool routing, knowledge store, audit trail, goal revision
   - Plot twist: competitor strategic pivot to AI discovered at step 8

2. **`investment_thesis/`** — Investment Thesis Builder (30 steps, evidence-driven goal revision)
   - Features: tool routing, knowledge store, audit trail, constraint checking
   - Plot twist: lawsuit discovery at step 10 triggers risk reassessment

3. **`data_pipeline_fixer/`** — Autonomous Pipeline Monitor (20 steps, goal achieved)
   - Features: evolving constraints, goal revision, multi-hypothesis repair
   - Plot twist: schema drift at step 8 degrades health → adapts to schema v2.0

4. **`learning_curriculum/`** — Adaptive Learning Curriculum (35 steps, longest example)
   - Features: feedback loop, knowledge store, prerequisite + time budget constraints
   - Plot twist: quiz failure at step 15 reveals knowledge gap → remedial modules added

5. **`deployment_coordinator/`** — Multi-Agent Deployment (3 agents, 3 rounds)
   - Features: multi-agent (Send API), LLM negotiation, per-agent goal revision
   - Plot twist: CVE discovery in round 2 → negotiated production pause → patching

## [1.5.0] — 2026-02-11

### Senior Review Fix — 12 Issues Resolved

Addresses 12 issues from a senior architecture review, focusing on correctness, immutability, thread safety, and test coverage. All changes are paper-aligned per Haidemariam (2026).

#### Phase 1: Quick Wins (5 issues)
- **Removed `from __future__ import annotations`** from 4 graph files (`nodes.py`, `graph.py`, `builder.py`, `edges.py`) — fixes LangGraph TypedDict resolution
- **Revision threshold fix**: changed from `abs(score) >= 0.5 or score <= -0.3` to `score <= -0.3` only — good scores no longer trigger revision
- **GoalTree memory leak fix**: `propagate_revision()` now removes old child from `_all_goals` dict
- **Softmax division-by-zero fix**: `_softmax()` and `LLMPlanner.__init__()` validate `temperature > 0`
- **Empty violations fix**: `LLMConstraintChecker` now falls back to overall reasoning when individual violations list is empty but `overall_safe=False`

#### Phase 2: Goal Immutability (2 issues)
- **Goal is now `@dataclass(frozen=True)`** — paper alignment: G_t → G_{t+1} produces NEW entity
- Lifecycle methods (`achieve`, `abandon`, `suspend`, `reactivate`) return new `Goal` via `dataclasses.replace()`
- `revise()` no longer mutates `self.status = REVISED` — old goal is simply replaced
- Updated all callers: `reflect_node`, `loop.py`, `goal_grounding.py`, `GoalTree.add_subgoal`
- Updated all tests asserting on mutation behavior

#### Phase 3: Thread Safety (1 issue)
- **`threading.Lock()`** added to 4 mutable shared-state classes:
  - `BudgetChecker` — `record_cost()`, `reset()`
  - `EvolvingConstraintManager` — `record_violations()`, `step()`, `evolve()`
  - `IntentionalGroundingManager` — `add_directive()`, `ground()`
  - `GoalTree` — `add_subgoal()`, `remove_subgoal()`, `propagate_revision()`
- Recursive calls use internal `_locked` methods to avoid deadlock

#### Phase 4: LLM Service Improvements (2 issues)
- **Class-level `ThreadPoolExecutor`** in all 4 LLM services (was per-call)
- Added `shutdown()` method to `LLMEvaluator`, `LLMPlanner`, `LLMReviser`, `LLMConstraintChecker`, `LLMNegotiator`
- **Timeout support for `LLMNegotiator`** — `_invoke_with_timeout()` method, replaces 3 direct `model.invoke()` calls

#### Phase 5: Test Coverage (1 issue)
- **`PromptCapturingMock`** — records all prompts sent to LLM for test assertion
- **`test_prompt_contents.py`** — 4 tests verifying prompt content (evaluator, planner, reviser, enriched observation)
- **`test_revision_e2e.py`** — end-to-end test: graph executes revision path and populates `goal_history`

#### Phase 6: Examples (2 issues)
- **Example 13 rewritten** — uses `build_multi_agent_graph()` with `AgentConfig` list, `negotiation_model` for LLM negotiation, `WorkingMemory` per agent
- **Sales SDR converted to LLM mode** — `.with_objective()` → `.with_model()` + `.with_goal()`, custom evaluator/planner override LLM defaults, `MockStructuredChatModel` for simulated mode

#### Phase 7: Docs + Version Bump
- `docs/decisions.md` — 4 new decisions (Goal immutability, revision threshold, thread safety, class-level executor)
- `docs/changelog.md` — this entry
- `docs/known_issues.md` — 12 issues logged
- Version bump: `1.4.0` → `1.5.0`

#### New Files (3)
- `src/.../testing/mock_llm.py` — `PromptCapturingMock` class added
- `tests/graph/test_prompt_contents.py` — 4 tests
- `tests/graph/test_revision_e2e.py` — 1 test

#### New Test Files (5)
- `tests/test_v150_phase1.py` — 8 tests (GoalTree leak, softmax, temperature, empty violations)
- `tests/test_v150_phase2.py` — 9 tests (frozen Goal, lifecycle, reflect_node)
- `tests/test_v150_phase3.py` — 4 tests (concurrent safety)
- `tests/graph/test_prompt_contents.py` — 4 tests
- `tests/graph/test_revision_e2e.py` — 1 test

#### Stats
- **741 tests** (713 + 28 new), all passing
- Lint clean on all changed files
- Version: `1.5.0`

---

## [1.4.0] — 2026-02-11

### Production-Grade Refinements

Focuses on positioning, environment layer maturity, and memory safety. Design philosophy: "Like PyTorch — academic in spirit, production grade."

#### Phase 1: BDI → Intentional State Mapping
- **Rename**: `BDIAgent` → `IntentionalStateAgent`, `bdi_bridge.py` → `intentional_bridge.py`
- All factory functions renamed: `make_bdi_*_node` → `make_intentional_*_node`, `build_bdi_teleological_graph` → `build_intentional_teleological_graph`
- Event types renamed: `bdi_goal_revised` → `intentional_goal_revised`, etc.
- `GraphBuilder.with_intentional_agent()` replaces `.with_bdi_agent()` (deprecated alias kept)
- `AgentConfig.intentional_agent` replaces `.bdi_agent`
- **Backward-compat shims**: old files re-export new names — zero breaking changes

#### Phase 2: Environment Layer — PyTorch-Style Features
- **`state_dict()` / `load_state_dict()`** on `BaseEnvironment` (PyTorch-style serialization)
  - `_state_dict_impl()` / `_load_state_dict_impl()` hooks implemented on all 4 environments
  - Round-trip serialization: save → modify → restore verified for Numeric, Resource, Research, Shared
- **`EnvironmentWrapper`** base class (Decorator pattern, like `gym.Wrapper`)
  - `NoisyObservationWrapper(env, noise_std)` — Gaussian noise on `observe()`
  - `HistoryTrackingWrapper(env, max_history)` — bounded `(action, snapshot)` recording
  - `ResourceQuotaWrapper(env, quotas, strict)` — per-resource usage caps
  - Composable stacking + `unwrapped` property

#### Phase 3: Bounded Accumulation Channels
- `_make_bounded_add(max_size)` — reducer factory capping append-only channels
- `make_bounded_state(max_history)` — creates `BoundedTeleologicalState` TypedDict
- `GraphBuilder.with_max_history(n)` — builder integration
- `build_teleological_graph(state_schema=...)` — accepts custom state schema
- Default behavior unchanged (unbounded `TeleologicalState`)

#### Phase 4: README + CLAUDE.md
- **README.md** — full rewrite to descriptive style (no version numbers in features, grouped by capability domain, "Design Philosophy" section)
- **CLAUDE.md** — project root rules file (documentation style, code conventions, architecture)

#### Phase 5: Docs + Verification
- `docs/decisions.md` — renamed BDI sections → ISM, added 4 new decisions
- `docs/changelog.md` — this entry
- Version bump: `1.3.0` → `1.4.0`

#### New Files (6)
- `src/.../agents/intentional.py`
- `src/.../graph/intentional_bridge.py`
- `src/.../environments/wrappers.py`
- `tests/graph/test_intentional_bridge.py`
- `tests/environments/test_state_dict.py`
- `tests/environments/test_wrappers.py`

#### New Test Files (4)
- `tests/graph/test_intentional_bridge.py` — 18 tests
- `tests/environments/test_state_dict.py` — 8 tests
- `tests/environments/test_wrappers.py` — 17 tests
- `tests/graph/test_bounded_state.py` — 10 tests

#### Stats
- **~709 tests** (656 + 53 new), all passing
- Lint clean on all changed files
- Version: `1.4.0`

---

## [1.3.0] — 2026-02-11

### Senior Review Gap Closure — 8 Architectural Improvements

Addresses 8 of 10 gaps identified in the senior architecture review. Two gaps (agent API consolidation and streaming-first design) are documented as future work.

#### Phase 1: LLM Error Handling + Timeout
- **LLMEvaluator**: error fallback now sets `confidence=0.0` (was 0.1) with `metadata={"llm_error": True, "error_type": ..., "error": ...}`
- **LLMPlanner**: error fallback returns a `noop_fallback` ActionSpec instead of empty PolicySpec (prevents agent freeze)
- **LLMConstraintChecker**: **CRITICAL** — changed from fail-open to fail-closed on error (`return False, "error..."` instead of `return True, ""`)
- **LLMReviser**: confirmed safe (already returns None on error)
- All 4 LLM services: added `timeout: float | None` parameter with `concurrent.futures`-based timeout

#### Phase 2: Strategy Closure Refactor (Checkpointing Support)
- `build_teleological_graph()` accepts optional strategy kwargs (`evaluator`, `goal_updater`, `planner`, `constraint_pipeline`, `policy_filter`)
- When provided, node functions capture strategies via closures instead of reading from state
- `GraphBuilder` now passes strategies as kwargs — strategies no longer stored in `initial_state`
- **Enables LangGraph checkpointing** (non-serializable objects removed from state)
- Backward compatible: graphs built without kwargs still read strategies from state
- Factory functions: `make_evaluate_node()`, `make_revise_node()`, `make_plan_node()`, `make_check_constraints_node()`, `make_filter_policy_node()`
- Updated: `multi_agent.py`, `bdi_bridge.py` to use strategy kwargs

#### Phase 3: Domain Refinements
- **`ConstraintResult`** value object: `passed`, `message`, `severity`, `checker_name`, `suggested_mitigation`, `metadata`
- **`BaseConstraintChecker.check_detailed()`**: returns `ConstraintResult` (default wraps `check()`)
- **`ConstraintPipeline.check_all_detailed()`**: returns `list[ConstraintResult]` with per-checker detail
- **`ActionSpec`**: added `effect: tuple[float, ...] | None` and `preconditions: Mapping[str, Any]` fields (backward compatible defaults)
- **`Goal.revise()`**: documented intentional mutation behavior in docstring
- **`BudgetChecker.reset()`**: added prominent docstring about reuse between runs

#### Phase 4: Wire Dead Abstractions
- **EvolvingConstraintManager**: wired into graph via `GraphBuilder.with_evolving_constraints(manager)`
  - New `evolve_constraints_node`: records violations, calls `manager.step()`, emits reasoning trace
  - `build_teleological_graph(enable_evolving_constraints=True)` wires node after check_constraints
- **KnowledgeStore**: connected to observation enrichment and reflection
  - `_build_enriched_observation()` queries `knowledge_store.query_recent()` and appends entries
  - `reflect_node` writes reflection data (`eval_score`, `stop_reason`, `goal_id`) to knowledge store

#### Files Changed
- `services/llm_evaluation.py` — error metadata + timeout
- `services/llm_planning.py` — noop fallback + timeout
- `services/llm_revision.py` — timeout
- `services/llm_constraints.py` — fail-closed + timeout
- `graph/nodes.py` — factory functions + knowledge + evolving constraints node
- `graph/graph.py` — strategy kwargs + evolving constraints wiring
- `graph/builder.py` — pass strategies as kwargs + evolving constraints
- `graph/state.py` — deprecation comments + evolving_constraint_manager field
- `graph/multi_agent.py` — use strategy kwargs
- `graph/bdi_bridge.py` — accept strategy kwargs
- `domain/values.py` — ConstraintResult + ActionSpec fields
- `domain/entities.py` — Goal.revise() docstring
- `services/constraint_engine.py` — check_detailed() + check_all_detailed()

#### New Test Files (4)
- `tests/services/test_llm_error_handling.py` — 5 tests
- `tests/graph/test_closure_strategies.py` — 3 tests
- `tests/domain/test_constraint_result.py` — 7 tests
- `tests/graph/test_integrations.py` — 5 tests

#### Stats
- **656 tests** (636 + 20 new), all passing
- Lint clean on all changed files
- Version: `1.3.0`

#### Deferred
- Agent API consolidation (class vs graph) — fundamental redesign affecting 40+ files
- Streaming-first design — architecture change affecting all examples

---

## [1.2.0] — 2026-02-11

### Agent Feedback Loop — Closes 3 Structural Gaps

Agents built with the toolkit now behave like paper-described agents: tool/action results flow back into perception, LLM services see history through enriched observations, and LLM examples have a real environment to reason about.

#### Gap 1: Action results flow back into perception
- New `action_feedback: Annotated[list, operator.add]` state channel in `TeleologicalState`
- `act_node` emits structured feedback entries (action name, tool_name, result, step, timestamp)
- `perceive_node` injects `action_feedback[-3:]` into `snapshot.context["recent_action_results"]`

#### Gap 2: LLM services see history without signature changes
- New `_build_enriched_observation()` helper appends to the observation text:
  - Recent action results (last 3): action name, tool, truncated result
  - Eval score trend (last 5): `0.30 -> 0.50 -> 0.70`
  - Goal revision count: "N revision(s) so far"
- No-op on step 1 (no history exists yet)
- All 3 LLM services (LLMEvaluator, LLMPlanner, LLMReviser) automatically see history through their `{observation}` prompt variable — zero changes to `BaseGoalUpdater` or any of its 7 concrete subclasses

#### Gap 3: LLM examples have an environment
- New `WorkingMemory` utility (`graph/working_memory.py`):
  - `perceive()` callback returns accumulated memory as `StateSnapshot.observation`
  - `record()` callback accepts 1 or 2 args (plain + constraint-conditioned signatures)
  - FIFO eviction at configurable `max_entries`
- Examples 11 (quickstart) and 12 (tools) updated with `WorkingMemory` and realistic initial context

#### Files Changed
- `graph/state.py` — +1 field: `action_feedback`
- `graph/nodes.py` — modified `perceive_node` (enrichment) + `act_node` (emit feedback) + new `_build_enriched_observation` helper
- `graph/builder.py` — `action_feedback: []` in both build methods
- `graph/working_memory.py` — **NEW**: WorkingMemory utility
- `graph/__init__.py` — added WorkingMemory export
- `examples/conceptual/11_llm_quickstart.py` — WorkingMemory + `.with_environment()`
- `examples/conceptual/12_llm_tools.py` — WorkingMemory + `.with_environment()`

#### Self-Contained LLM Examples
- Examples 11-13 rewritten as self-contained (no API key required)
- `synthetic_teleology.testing` module: `MockStructuredChatModel` promoted from `tests/helpers/` to public API
- Examples default to mock LLM, optionally use real LLM when API key is present
- All examples demonstrate the feedback loop: action_feedback, enriched observation, WorkingMemory

#### Files Changed (examples rewrite)
- `src/synthetic_teleology/testing/__init__.py` — **NEW**: public testing utilities
- `src/synthetic_teleology/testing/mock_llm.py` — **NEW**: MockStructuredChatModel (promoted)
- `tests/helpers/mock_llm.py` — re-exports from new location
- `examples/conceptual/11_llm_quickstart.py` — self-contained + feedback loop demo
- `examples/conceptual/12_llm_tools.py` — self-contained + tool feedback demo
- `examples/conceptual/13_llm_multi_agent.py` — self-contained + per-agent feedback

#### Stats
- **636 tests** (617 + 19 new), all passing
- 19 new tests in `test_working_memory.py` (8) + `test_action_feedback.py` (11)
- Lint clean on all changed files

---

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
