# Data Pipeline Fixer

Autonomous data pipeline monitoring and repair agent built with Synthetic Teleology.

## What It Does

The agent monitors a simulated data pipeline with 8 tables, running health checks and applying fixes when problems are detected. Mid-run, the pipeline experiences **schema drift** (upstream producer migrates to v2.0 without coordination), causing health to drop and error rate to spike. The agent detects this, revises its goal to include schema migration, and applies a series of diagnostic and repair actions.

This example demonstrates:

- **Goal revision**: when schema drift triggers a score below -0.3, the LLM reviser fires and adapts the goal to include schema v2.0 migration criteria.
- **Evolving constraints**: an `EvolvingConstraintManager` periodically proposes new constraints (e.g., "schema changes must be validated before deployment") based on observed violation patterns.
- **Hybrid mock pattern**: custom `PipelineEvaluator` reads simulated state for deterministic scores while `MockStructuredChatModel` provides LLM planner and reviser responses.
- **Simulated tools**: `check_health`, `run_diagnostic`, `apply_fix`, and `rollback` tools mutate `PipelineState` to model realistic pipeline behaviour.
- **Constraint checkers**: budget limits (max 15 fix attempts) and safety checks (block `drop_table`).

## Architecture

```
PipelineState (mutable)
    |
    v
Tools (check_health, run_diagnostic, apply_fix, rollback)
    |                                           |
    v                                           v
PipelineEvaluator ---- score ----> GraphBuilder loop
    |                                           |
    |   (score <= -0.3)                         |
    +---------> LLM Reviser ---> goal revision  |
                                                |
EvolvingConstraintManager ---> constraint co-evolution
```

## Running

```bash
PYTHONPATH=src python -m examples.production.data_pipeline_fixer.main
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 25 | Max monitoring/repair rounds |
| `--verbose` | off | Show evaluation and goal history |

### With a real LLM

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` to use a live model instead of mocks:

```bash
ANTHROPIC_API_KEY=sk-... PYTHONPATH=src python -m examples.production.data_pipeline_fixer.main
```

## Files

| File | Purpose |
|------|---------|
| `models.py` | `PipelineState`, `FixAttempt`, `SchemaVersion` domain dataclasses |
| `tools.py` | Simulated LangChain-compatible tools that mutate `PipelineState` |
| `strategies.py` | `PipelineEvaluator`, `BudgetConstraintChecker`, `SafetyConstraintChecker` |
| `agent.py` | Graph wiring, mock response sequences, `build_pipeline_agent()` |
| `main.py` | CLI entry point with argparse |

## Timeline

1. **Steps 1-7**: routine health monitoring across pipeline components
2. **Step 8**: schema drift detected, health drops to 0.55, error rate spikes to 0.18
3. **Step 8**: eval score drops below -0.3, triggering goal revision to include schema v2.0 migration
4. **Steps 9-10**: diagnostics on affected tables (orders, events)
5. **Steps 11-18**: rollback, adapter, migration, and retry fixes
6. **Steps 19-25**: continued repair and final health checks
7. **Every 3 steps**: evolving constraint manager proposes constraint updates
