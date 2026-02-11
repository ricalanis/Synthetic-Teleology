# Known Issues

## 2026-02-11: 12 Senior Review Issues Fixed in v1.5.0

The following 12 issues were identified in a senior architecture review and resolved:

1. **`from __future__ import annotations` in graph files** — broke LangGraph TypedDict resolution. Fixed: removed from `nodes.py`, `graph.py`, `builder.py`, `edges.py`. Forward references in `builder.py` quoted as `-> "GraphBuilder":`.
2. **Goal mutation breaking functional node contracts** — `Goal` was mutable, lifecycle methods mutated in-place. Fixed: `@dataclass(frozen=True)`, all lifecycle methods return new instances via `dataclasses.replace()`.
3. **`revise()` side-effect** — set `self.status = REVISED` on the original goal. Fixed: removed the side-effect. Old goal stays `ACTIVE`, new goal is independent.
4. **Thread safety on shared mutable state** — `BudgetChecker`, `EvolvingConstraintManager`, `IntentionalGroundingManager`, `GoalTree` had unprotected mutable state. Fixed: `threading.Lock()` on all four.
5. **GoalTree memory leak** — `propagate_revision()` added new children but never removed old ones from `_all_goals`. Fixed: `self._all_goals.pop(child.goal_id, None)`.
6. **Revision threshold triggered on good scores** — `abs(score) >= 0.5` meant scores like 0.8 triggered revision. Fixed: only `score <= -0.3` triggers revision.
7. **Per-call ThreadPoolExecutor** — LLM services created a new executor per `invoke` with timeout. Fixed: class-level executor in `__init__`, plus `shutdown()` method.
8. **Softmax division by zero** — `_softmax(values, temperature=0)` caused division by zero. Fixed: validation in `_softmax()` and `LLMPlanner.__init__()`.
9. **Empty constraint violations** — `LLMConstraintChecker` could return `overall_safe=False` with empty violations list. Fixed: fallback to overall reasoning text.
10. **No prompt content tests** — LLM services weren't verified to include goal/criteria in prompts. Fixed: `PromptCapturingMock` + `test_prompt_contents.py`.
11. **Example 13 not using multi-agent API** — ran agents independently, no negotiation. Fixed: rewritten with `build_multi_agent_graph()`, `AgentConfig`, `negotiation_model`.
12. **Sales SDR in numeric mode** — used `.with_objective()` instead of LLM mode. Fixed: converted to `.with_model()` + `.with_goal()` with custom evaluator/planner overrides.

## 2026-02-11: Removing `from __future__ import annotations` from builder.py breaks forward references

- **What happened:** After removing the import, `-> GraphBuilder:` return type annotations caused `NameError` at class definition time.
- **Root cause:** Without `from __future__ import annotations`, forward references to the class being defined need to be quoted strings.
- **Fix:** Changed all `-> GraphBuilder:` to `-> "GraphBuilder":` with `replace_all=true`.
- **How to avoid:** When removing `from __future__ import annotations`, check for self-referencing return types and quote them.

---

## 2026-02-10: Editable install .pth file not processed by uv venv

- **What happened:** After `uv pip install -e .`, the generated `_synthetic_teleology.pth` file exists in site-packages with correct content (`/path/to/src`), but Python does not add the path to `sys.path`. Tests fail with `ModuleNotFoundError`.
- **Root cause:** The uv-created virtualenv uses symlinked Python from `~/.local/share/uv/python/cpython-3.11.13-macos-aarch64-none/`. The `.pth` file processing appears to silently fail in this configuration.
- **Workaround:** Use `PYTHONPATH=src` when running tests or examples: `PYTHONPATH=src .venv/bin/python -m pytest tests/`.
- **How to avoid:** Always prefix test/run commands with `PYTHONPATH=src` until the uv editable install issue is resolved upstream.

## 2026-02-10: Test used wrong keyword argument for ConstraintViolated event

- **What happened:** `tests/measurement/test_collector.py::test_constraint_violation_captured` failed with `TypeError: ConstraintViolated.__init__() got an unexpected keyword argument 'violation_message'`.
- **Root cause:** Test code used `violation_message=` but the `ConstraintViolated` event's field is named `violation_details` (defined in `domain/events.py`).
- **Fix:** Changed `violation_message="out of bounds"` to `violation_details="out of bounds"` in the test.
- **How to avoid:** When writing tests for domain events, always verify field names against the event dataclass definition. Frozen dataclasses don't provide helpful error messages for wrong kwargs.

## 2026-02-10: ReflectiveEvaluator drift test used equidistant states

- **What happened:** `tests/services/test_evaluation.py::test_drift_reduces_confidence` asserted `confidences[-1] < 1.0` but confidence stayed at 1.0.
- **Root cause:** The test oscillated state between 0.0 and 10.0 with goal at `values=(5.0,)`. Both states are equidistant from the goal (distance=5), producing identical scores (~0.0). With zero score variance, drift detection never triggers.
- **Fix:** Changed goal from `values=(5.0,)` to `values=(0.0,)` so oscillation between 0 and 10 produces genuinely different scores (~1.0 and ~-1.0), triggering the drift detector.
- **How to avoid:** When testing score-based statistical detectors, verify that the test inputs actually produce diverse scores. Calculate expected scores manually before writing assertions.

## 2026-02-10: knowledge_synthesis.py forward reference

- **What happened:** Initial implementation of `ResearchPlanner.plan()` used a placeholder type `state_snapshot_type = type(None)` with a module-level re-import hack for the `StateSnapshot` type annotation.
- **Root cause:** Copy-paste from a pattern that handled circular imports, but `StateSnapshot` was already available from the `values` module with no circular dependency.
- **Fix:** Added `StateSnapshot` directly to the imports and removed the hacky placeholder.
- **How to avoid:** Always check if the needed type is already importable before introducing workarounds. Run import tests immediately after writing new files.

## 2026-02-10: numpy not found in default Python

- **What happened:** Running tests with the system Python (3.12 in openhands venv) failed because numpy was not installed there.
- **Root cause:** The project has its own `.venv` (Python 3.11.13) at `.venv/bin/python` with all dependencies installed.
- **Fix:** Switched to using the project venv for all test runs.
- **How to avoid:** Always use the project's virtual environment for testing. Check for `.venv/bin/python` in the project root first.

## 2026-02-10: `from __future__ import annotations` breaks LangGraph TypedDict resolution

- **What happened:** `NameError: name 'Goal' is not defined` when LangGraph compiled a `StateGraph(TeleologicalState)`. Also affected `MultiAgentState` in `multi_agent.py`.
- **Root cause:** LangGraph uses `get_type_hints(..., include_extras=True)` at runtime to resolve TypedDict field annotations (especially `Annotated[list, operator.add]` reducers). When `from __future__ import annotations` is active, all annotations become strings. LangGraph's `get_type_hints()` then fails to resolve forward references like `"Goal"` or `"Annotated[list, operator.add]"`.
- **Fix:** Removed `from __future__ import annotations` from `graph/state.py` and `graph/multi_agent.py`. Used explicit `Union[X, None]` instead of `X | None` for optional types.
- **How to avoid:** Never use `from __future__ import annotations` in files that define TypedDict schemas consumed by LangGraph. This is a known LangGraph/Python 3.11 compatibility constraint. Document this in any new state/schema files.

## 2026-02-10: LangGraph checkpointer cannot serialize custom strategy objects

- **What happened:** `TypeError: Type is not msgpack serializable: NumericEvaluator` when using `MemorySaver` checkpointer with a teleological graph.
- **Root cause:** The `TeleologicalState` stores injected strategies (evaluator, planner, goal_updater, etc.) as dict values. LangGraph's `MemorySaver` (and other checkpointers) use msgpack serialization, which cannot serialize arbitrary Python objects. The strategies are callables/class instances, not JSON-serializable data.
- **Workaround:** For examples requiring checkpointing, avoid storing custom objects in state. The `04_human_in_the_loop.py` example uses a simulated approval callback pattern instead of LangGraph `interrupt()`/`Command(resume=...)`.
- **How to avoid:** If full checkpointing is needed, strategies would need to be stored outside the state (e.g., via closures in node functions, or a global registry). This is a fundamental tension between LangGraph's serialization model and the dependency-injection approach. A future version could use a serializable strategy identifier + registry lookup pattern.

## 2026-02-11: Goal.revise() returns tuple, not just Goal

- **What happened:** `goal_grounding.py` initially called `return goal.revise(new_description=..., reason=...)` expecting a `Goal` object back.
- **Root cause:** `Goal.revise()` returns `tuple[Goal, GoalRevision]` (the new goal plus a revision record). Code that assigns the result directly as a Goal gets a tuple instead.
- **Fix:** Unpack the tuple: `new_goal, _revision = goal.revise(...)`. Set provenance on `new_goal` separately since `revise()` doesn't accept a `provenance` kwarg.
- **How to avoid:** Always check the return type of `Goal.revise()` — it returns a 2-tuple. Any code calling `revise()` must unpack.

## 2026-02-11: StateSnapshot requires timestamp argument

- **What happened:** `StateSnapshot(values=(1.0,))` raised `TypeError: missing 1 required positional argument: 'timestamp'` in BDI bridge tests.
- **Root cause:** `StateSnapshot` is a frozen dataclass with `timestamp: float` as a required field. Easy to forget when constructing test fixtures.
- **Fix:** Added `timestamp=0.0` to all StateSnapshot constructor calls in tests.
- **How to avoid:** When creating StateSnapshot in tests, always include `timestamp=0.0` (or appropriate value).

## 2026-02-10: LangGraph stream returns dict chunks, not tuples

- **What happened:** `ValueError: not enough values to unpack (expected 2, got 1)` in `streaming.py`.
- **Root cause:** LangGraph `.stream(mode="updates")` returns `dict[str, dict]` chunks (mapping node name → state update), not `tuple[str, dict]` as initially assumed.
- **Fix:** Changed iteration from `for node_name, state_update in stream:` to `for chunk in stream: for node_name, state_update in chunk.items():`.
- **How to avoid:** Always test streaming code against a real compiled graph. The LangGraph stream API returns chunks where each chunk is a dict, not a sequence of tuples.
