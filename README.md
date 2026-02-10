# Synthetic Teleology Framework

Production-grade Python framework implementing Haidemariam (2026) *"From the logic of coordination to goal-directed reasoning"* — the theory of **Synthetic Teleology** in Agentic AI.

## Overview

Synthetic Teleology provides a complete architecture for building goal-directed agents that can:

- **Evaluate** their progress toward objectives using pluggable evaluation strategies
- **Revise** goals dynamically based on feedback, constraints, and environmental changes
- **Plan** and execute actions through configurable planning strategies
- **Reflect** on their own performance to adjust confidence and detect drift
- **Coordinate** with other agents through negotiation protocols
- **Measure** their behavior against 7 teleological metrics from the theory

The framework implements the recursive agentic loop: **Perceive → Evaluate → Revise → Plan → Act → Reflect**.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TELEOLOGY DOMAIN                         │
│  Goal, GoalTree, GoalRevision, EvalSignal, SyntheticPurpose│
├─────────────────────────────────────────────────────────────┤
│                     AGENCY DOMAIN                           │
│  Agent, AgenticLoop, Perception, Deliberation, Action       │
├─────────────────────────────────────────────────────────────┤
│                  COORDINATION DOMAIN                        │
│  AgentNetwork, Negotiator, Message, SharedIntent            │
├─────────────────────────────────────────────────────────────┤
│                  MEASUREMENT DOMAIN                         │
│  MetricsEngine, Benchmark, Report, TimeSeriesLog            │
├─────────────────────────────────────────────────────────────┤
│                  INFRASTRUCTURE                             │
│  LLMProvider, EventBus, Registry, Serialization, CLI        │
└─────────────────────────────────────────────────────────────┘
```

**Design Patterns**: Strategy, Observer, Composite, Builder, Factory, Chain of Responsibility, Mediator, Decorator, Template Method, State Machine, Repository, Value Object.

## Installation

```bash
# Core framework (numpy only)
pip install -e .

# With visualization (Rich + Matplotlib)
pip install -e ".[viz]"

# With LLM providers
pip install -e ".[llm-anthropic]"    # Anthropic Claude
pip install -e ".[llm-openai]"       # OpenAI GPT
pip install -e ".[llm-huggingface]"  # Local HuggingFace models
pip install -e ".[llm-generic]"      # Any OpenAI-compatible API

# Everything
pip install -e ".[all]"

# Development
pip install -e ".[dev]"
```

**Requires:** Python >= 3.11, numpy >= 1.24

## Quick Start

```python
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import ObjectiveVector, StateSnapshot
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.services.evaluation import NumericEvaluator

# Define a goal
objective = ObjectiveVector(
    values=(10.0, 20.0),
    directions=(Direction.APPROACH, Direction.APPROACH),
)
goal = Goal(name="reach_target", objective=objective)

# Evaluate current state
evaluator = NumericEvaluator(max_distance=50.0)
state = StateSnapshot(timestamp=0.0, values=(8.0, 18.0))
signal = evaluator.evaluate(goal, state)

print(f"Score: {signal.score:.2f}")       # 0.84
print(f"Confidence: {signal.confidence}")  # 1.0
```

### Running a Full Agentic Loop

```python
from synthetic_teleology.agents.factory import AgentBuilder
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.services.loop import SyncAgenticLoop
from synthetic_teleology.infrastructure.event_bus import EventBus

# Build an agent
bus = EventBus()
agent = (
    AgentBuilder("agent-1")
    .with_goal(goal)
    .with_evaluator(NumericEvaluator(max_distance=50.0))
    .with_event_bus(bus)
    .build()
)

# Create environment and run
env = NumericEnvironment(dimensions=2)
loop = SyncAgenticLoop(agent=agent, environment=env, event_bus=bus)
result = loop.run(max_steps=100)
print(f"Stopped: {result.stop_reason}, Steps: {result.steps_completed}")
```

## Examples

| Example | Description |
|---------|-------------|
| `01_basic_loop.py` | End-to-end SyncAgenticLoop with NumericEnvironment |
| `02_hierarchical_goals.py` | GoalTree with subgoals and coherence validation |
| `03_reflective_critic.py` | ReflectiveEvaluator confidence adjustment |
| `04_multi_agent_negotiation.py` | SharedEnvironment with consensus negotiation |
| `05_benchmark_suite.py` | MetricsEngine with synthetic agent logs |
| `06_llm_research_agent.py` | LLMAgent configuration with mock provider |
| `07_distribution_shift.py` | DistributionShiftBenchmark end-to-end |
| `08_constraint_integration.py` | Constraint pipeline with safety and budget checks |

Run any example:

```bash
PYTHONPATH=src python examples/01_basic_loop.py
```

## Metrics

The framework implements 7 metrics from the Synthetic Teleology theory:

| Metric | Abbreviation | Measures |
|--------|-------------|----------|
| Goal Persistence | GP | Fraction of steps maintaining the same goal |
| Teleological Coherence | TC | Correlation between score improvement and goal stability |
| Reflective Efficiency | RE | Cost-weighted score improvement per step |
| Adaptivity | AD | Post-perturbation recovery rate |
| Normative Fidelity | NF | Fraction of steps without constraint violations |
| Innovation Yield | IY | Score improvement from novel actions |
| Lyapunov Stability | LS | Score variance convergence over time |

## CLI

```bash
# Run a basic agent loop
synthetic-teleology run --steps 100 --dimensions 3

# Run benchmarks
synthetic-teleology benchmark --all

# Show framework info
synthetic-teleology info
```

## Testing

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -v
```

**380 tests** covering all modules.

## Package Structure

```
src/synthetic_teleology/
├── domain/          # Value objects, entities, aggregates, events, enums
├── services/        # Evaluation, goal revision, planning, constraints, loop
├── agents/          # Base, teleological, BDI, LLM agents + factory
├── environments/    # Numeric, resource, research, shared environments
├── measurement/     # Collector, 7 metrics, engine, reports, 4 benchmarks
├── infrastructure/  # EventBus, registry, config, serialization, LLM providers
├── presentation/    # Console dashboard, plots, export (JSON/CSV/HTML)
└── cli.py           # Command-line interface
```

## License

MIT
