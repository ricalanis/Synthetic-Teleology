# Normalize 5 Production Examples — Session Summary

**Date:** 2026-02-11
**Status:** DONE
**Session:** normalize-prod-examples

## What Was Done

Normalized all 5 new production examples (competitive_research, investment_thesis, data_pipeline_fixer, learning_curriculum, deployment_coordinator) for consistency. 14 files modified + README updated.

### Changes by Category

#### A) `_get_model()` pattern — all 5 now return `None` on no API key

| Example | Before | After |
|---------|--------|-------|
| competitive_research | Already correct (returns None) | No change |
| investment_thesis | Returned `_build_mock_model()` directly | Returns `None`, caller does `_get_model() or _build_mock_model()` |
| data_pipeline_fixer | Returned None but temperature=0.3 | Fixed temperature to 0.5, cleaned docstring |
| learning_curriculum | Returned `MockStructuredChatModel(...)` inline | Returns `None`, added `_build_mock_model()` helper |
| deployment_coordinator | Already correct (returns None) | No change |

#### B) `__init__.py` exports — all 5 now export their build function

```
competitive_research:    from .agent import build_research_agent
investment_thesis:       from .agent import build_thesis_agent
data_pipeline_fixer:     from .agent import build_pipeline_agent
learning_curriculum:     from .agent import build_curriculum_agent
deployment_coordinator:  from .agent import build_deployment_coordinator
```

#### C) KnowledgeStore + AuditTrail wiring

| Example | KS Before | KS After | AT Before | AT After |
|---------|-----------|----------|-----------|----------|
| competitive_research | Had both | No change | Had both | No change |
| investment_thesis | Had both | No change | Had both | No change |
| data_pipeline_fixer | None | Added + seeded (pipeline:config, pipeline:initial_health) | None | Added, wired via `.with_audit_trail()` |
| learning_curriculum | Had KS | No change | None | Added, wired via `.with_audit_trail()` |
| deployment_coordinator | N/A (multi-agent) | Skip | N/A | Skip |

#### D) `main.py` mode detection + output

| Example | Mode detection | Verbose output | Private attr fix |
|---------|---------------|----------------|-----------------|
| competitive_research | Already had it | Added step-by-step eval/action/feedback (was ignored before) | N/A |
| investment_thesis | Fixed string to include "(MockStructuredChatModel)" | Already had it | Replaced `._entries` with `.entries`, `.keys()`, `.get()` |
| data_pipeline_fixer | Added `has_key`/`mode` block + print | Added KS/AT output sections + counts in summary | N/A |
| learning_curriculum | Added `has_key`/`mode` block + print | Added AT output section + count in summary | N/A |
| deployment_coordinator | Already had it | Already had it | N/A |

#### E) Return tuple changes

| Example | Before | After |
|---------|--------|-------|
| data_pipeline_fixer | `(app, initial_state, pipeline_state)` | `(app, initial_state, pipeline_state, knowledge_store, audit_trail)` |
| learning_curriculum | `(app, initial_state, curriculum_state, knowledge_store)` | `(app, initial_state, curriculum_state, knowledge_store, audit_trail)` |

#### F) README

- `investment_thesis/README.md`: Added ASCII architecture diagram matching competitive_research pattern
- Root `README.md`: Updated Production table from 2 rows (Polymarket, Sales SDR) to 7 rows (all production examples), updated footnote

### Files Modified (14 + README)

1. `examples/production/competitive_research/__init__.py`
2. `examples/production/competitive_research/main.py`
3. `examples/production/investment_thesis/__init__.py`
4. `examples/production/investment_thesis/agent.py`
5. `examples/production/investment_thesis/main.py`
6. `examples/production/investment_thesis/README.md`
7. `examples/production/data_pipeline_fixer/__init__.py`
8. `examples/production/data_pipeline_fixer/agent.py`
9. `examples/production/data_pipeline_fixer/main.py`
10. `examples/production/learning_curriculum/__init__.py`
11. `examples/production/learning_curriculum/agent.py`
12. `examples/production/learning_curriculum/main.py`
13. `examples/production/deployment_coordinator/__init__.py`
14. `README.md`
15. `docs/changelog.md`

## Verification

All passed:

```
PYTHONPATH=src .venv/bin/python -m examples.production.competitive_research.main --verbose   # 18 steps, goal_achieved
PYTHONPATH=src .venv/bin/python -m examples.production.investment_thesis.main --verbose       # 30 steps, max_steps
PYTHONPATH=src .venv/bin/python -m examples.production.data_pipeline_fixer.main --verbose     # 20 steps, goal_achieved
PYTHONPATH=src .venv/bin/python -m examples.production.learning_curriculum.main --verbose      # 35 steps, max_steps
PYTHONPATH=src .venv/bin/python -m examples.production.deployment_coordinator.main --verbose   # 3 rounds, done
.venv/bin/ruff check examples/production/   # All checks passed!
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q   # 741 passed
```

## Still Not Committed

All changes are unstaged. Ready for commit when the user approves.

## Potential Follow-Up Work

- Normalize README.md section headers across all 5 examples ("What It Does" vs "What it demonstrates")
- Add `--verbose` to deployment_coordinator (currently `--verbose` shows reasoning traces, different from the step-by-step pattern of the other 4)
- Consider adding KS/AT to deployment_coordinator if multi-agent infra supports it in the future
- Verify that real LLM mode works end-to-end (needs API key)
