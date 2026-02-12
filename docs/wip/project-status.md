# Project Status — Synthetic Teleology

**Date:** 2026-02-11
**Version:** 1.5.0
**Session:** v1.5.0-senior-review + normalize-prod-examples

---

## Current State

### Version & Tests
- **Version**: 1.5.0 (committed at `0a0eed2`)
- **Tests**: 741 passing (`PYTHONPATH=src .venv/bin/python -m pytest tests/ -v`)
- **Lint**: clean on new files (`.venv/bin/ruff check src/ tests/ examples/`)

### Git Status
- **Branch**: `main`
- **Recent commits**:
  - `ddc0c19` Standard examples
  - `1f4cc30` new examples
  - `0a0eed2` v 1.5.0

- **3 uncommitted files** (normalization work — ready to commit):
  - `README.md` — adds 5 new production examples to Production table + updated footnote
  - `docs/decisions.md` — adds "Normalized Production Example Conventions" decision
  - `docs/known_issues.md` — adds 2 new issues (private `._entries` access, inconsistent `_get_model()`)

### What To Do Next
1. **Commit the 3 uncommitted files** — normalization docs are complete and verified
2. **Clean `docs/wip/`** — `normalize-production-examples.md` is marked DONE, can be deleted after commit

---

## What Was Completed

### v1.5.0: Senior Review Fix (7 phases, all done)
1. **Quick Wins** — removed `from __future__ import annotations` from 4 graph files, fixed revision threshold to `score <= -0.3` only, fixed GoalTree memory leak, softmax div-by-zero validation, empty violations fallback
2. **Goal Immutability** — `Goal` is now `@dataclass(frozen=True)`, lifecycle methods return new instances via `dataclasses.replace()`, `revise()` no longer mutates `self.status`
3. **Thread Safety** — `threading.Lock()` on BudgetChecker, EvolvingConstraintManager, IntentionalGroundingManager, GoalTree
4. **LLM Service Improvements** — class-level `ThreadPoolExecutor`, timeout on `LLMNegotiator`, JSON parse error handling
5. **Test Coverage** — `PromptCapturingMock`, prompt content verification tests, e2e revision path test
6. **Examples** — Example 13 rewritten with `build_multi_agent_graph()` + negotiation, Sales SDR converted to LLM mode
7. **Docs + Version Bump** — changelog, decisions, known_issues updated, version 1.4.0 → 1.5.0

### Production Examples (5 new, committed)
- `competitive_research/` — Competitive intelligence (18 steps, pivot discovery)
- `investment_thesis/` — Investment analysis (30 steps, lawsuit discovery)
- `data_pipeline_fixer/` — Pipeline monitoring (20 steps, schema drift)
- `learning_curriculum/` — Adaptive curriculum (35 steps, quiz failure)
- `deployment_coordinator/` — Multi-agent deployment (3 agents, CVE discovery)

### Normalization (done, 3 files uncommitted)
- Standardized `_get_model()` pattern across all 5 examples
- Added `__init__.py` exports, KnowledgeStore/AuditTrail wiring
- Fixed private attribute access, mode detection banners, verbose output

---

## Project Stats
- **Tests**: 741 (498 v1.0 + 119 v1.1.0 + 19 v1.2.0 + 20 v1.3.0 + 57 v1.4.0 + 28 v1.5.0)
- **Conceptual examples**: 14 (10 numeric + 4 LLM)
- **Production examples**: 7 (polymarket, sales_sdr, competitive_research, investment_thesis, data_pipeline_fixer, learning_curriculum, deployment_coordinator)
- **Source files**: 80+ across 9 packages

---

## Key Patterns to Remember

1. **Always use `PYTHONPATH=src`** for tests/examples
2. **Never `from __future__ import annotations`** in LangGraph TypedDict files
3. **Goal is frozen** — lifecycle methods return new instances
4. **MockStructuredChatModel** from `tests/helpers/mock_llm.py` (not FakeListChatModel)
5. **Hybrid Mock Pattern** for production examples: custom evaluator + mock LLM for planner/reviser
6. **`_get_model()` returns `None`** when no API key; caller does `_get_model() or _build_mock_model()`
7. **Revision threshold**: only `score <= -0.3` triggers revision

---

## Potential Future Work
- Normalize README section headers across all 5 new production examples
- Add `--verbose` step-by-step pattern to deployment_coordinator
- Add KS/AT to deployment_coordinator (multi-agent infra limitation)
- Streaming-first design (deferred architectural decision)
- Agent API consolidation (deferred — graph API is primary)
- Verify real LLM mode works end-to-end on all 7 production examples
