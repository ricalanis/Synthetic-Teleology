# Known Issues

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
