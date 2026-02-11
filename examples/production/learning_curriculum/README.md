# Adaptive Learning Curriculum

Goal-directed curriculum optimization agent that adapts teaching to simulated learner performance.

## What it demonstrates

- **Custom evaluator + LLM planner/reviser hybrid**: The `CurriculumEvaluator` reads simulated learner state for deterministic scores, while the LLM handles planning (tool selection) and revision (goal adaptation).
- **Quiz-triggered goal revision**: When the learner fails a quiz on a hard topic (state_management), the eval score drops to -0.5, triggering the revision pipeline. The reviser adds remedial criteria.
- **Tool-augmented planning**: Four simulated tools (`generate_lesson`, `create_quiz`, `assess_performance`, `search_resources`) are invoked via `tool_name` in `PlanningOutput`.
- **Knowledge store**: Tracks learner profile and reflection data across the run.
- **Constraint checking**: Prerequisites and time budget constraints gate curriculum progression.
- **Long-running feedback loop**: 35 steps with complex state evolution — the longest production example.

## Architecture

```
CurriculumState (mutable)
  |
  +-- Tools mutate state (lessons, quizzes, resources)
  |
  +-- CurriculumEvaluator reads state -> EvalSignal
  |
  +-- LLMPlanner (mock) -> PlanningOutput with tool_name
  |
  +-- LLMReviser (mock) -> RevisionOutput on quiz failure
  |
  +-- PrerequisiteChecker + TimeBudgetChecker
```

## Curriculum

10 React topics ordered by difficulty:

| Topic            | Difficulty | Prerequisites             |
|-----------------|-----------|---------------------------|
| jsx_basics       | 0.2       | (none)                    |
| components       | 0.3       | jsx_basics                |
| props            | 0.3       | components                |
| state_management | 0.6       | props                     |
| hooks            | 0.7       | state_management          |
| effects          | 0.6       | hooks                     |
| context          | 0.7       | state_management          |
| routing          | 0.5       | components                |
| forms            | 0.5       | state_management          |
| testing          | 0.8       | components, hooks         |

## Learner simulation

The simulated learner starts strong on basics (jsx_basics=0.8, components=0.7) but weak on advanced topics (state_management=0.2, hooks=0.3). Tool invocations gradually boost mastery:

- `generate_lesson`: +0.1 mastery
- `search_resources`: +0.05 mastery
- `create_quiz`: no direct boost (tests current mastery)

## Running

```bash
# Default (35 steps)
PYTHONPATH=src python -m examples.production.learning_curriculum.main

# Verbose output
PYTHONPATH=src python -m examples.production.learning_curriculum.main --verbose

# Shorter run
PYTHONPATH=src python -m examples.production.learning_curriculum.main --steps 20
```

No API key required — uses `MockStructuredChatModel` in simulated mode. Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` for real LLM planning and revision.

## Key design decisions

1. **Deterministic evaluator**: Scoring from CurriculumState ensures reproducible revision triggers regardless of LLM model.
2. **Mock response ordering**: PlanningOutput responses follow the topic progression order. A single RevisionOutput is injected at the exact point where the state_management quiz fails.
3. **Mastery accumulation**: Multiple tool invocations per topic allow mastery to cross the 0.5 threshold for quiz passing after remedial content.
4. **Time budget constraint**: 180-minute budget prevents infinite remediation loops.
