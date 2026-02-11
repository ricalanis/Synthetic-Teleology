# Synthetic Teleology — LangGraph Toolkit

LangGraph toolkit for building **goal-directed agents** implementing Haidemariam (2026) *"From the logic of coordination to goal-directed reasoning"* — the theory of **Synthetic Teleology** in Agentic AI.

## Overview

Synthetic Teleology provides a LangGraph-native architecture for building agents that can:

- **Evaluate** their progress toward objectives using pluggable evaluation strategies
- **Revise** goals dynamically based on feedback, constraints, and environmental changes
- **Plan** and execute actions through configurable planning strategies
- **Reflect** on their own performance to adjust confidence and detect drift
- **Coordinate** with other agents through negotiation protocols
- **Measure** their behavior against 7 teleological metrics from the theory

The teleological loop runs as a **LangGraph StateGraph**: **Perceive → Evaluate → Revise → Plan → Filter → Act → Reflect**.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  LANGGRAPH LAYER (v0.2.0)                    │
│  StateGraph, Nodes, Edges, Builder, Streaming, Multi-Agent   │
├─────────────────────────────────────────────────────────────┤
│                    TELEOLOGY DOMAIN                          │
│  Goal, GoalTree, GoalRevision, EvalSignal, Constraints       │
├─────────────────────────────────────────────────────────────┤
│                   STRATEGY SERVICES                          │
│  Evaluators, GoalUpdaters, Planners, ConstraintPipeline      │
├─────────────────────────────────────────────────────────────┤
│                  COORDINATION DOMAIN                         │
│  ConsensusNegotiator, VotingNegotiator, AuctionNegotiator    │
├─────────────────────────────────────────────────────────────┤
│                  MEASUREMENT DOMAIN                          │
│  MetricsEngine, Benchmark, Report, TimeSeriesLog             │
├─────────────────────────────────────────────────────────────┤
│                  INFRASTRUCTURE                              │
│  LLMProvider, EventBus, Registry, Config, CLI                │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Core (numpy + langgraph + langchain-core)
pip install -e .

# With LangChain LLM providers
pip install -e ".[llm-anthropic-lc]"   # langchain-anthropic
pip install -e ".[llm-openai-lc]"      # langchain-openai

# With visualization (Rich + Matplotlib)
pip install -e ".[viz]"

# Everything
pip install -e ".[all]"

# Development
pip install -e ".[dev]"
```

**Requires:** Python >= 3.11

## Quick Start

```python
from synthetic_teleology.graph import GraphBuilder
from synthetic_teleology.environments.numeric import NumericEnvironment

# Create environment
env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))

# Build a teleological graph with one line
app, initial_state = (
    GraphBuilder("my-agent")
    .with_objective((5.0, 5.0))            # target values
    .with_max_steps(20)                     # loop limit
    .with_goal_achieved_threshold(0.9)      # early stop
    .with_environment(
        perceive_fn=lambda: env.observe(),
        act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
        transition_fn=lambda a: env.step(a) if a else None,
    )
    .build()
)

# Run the teleological loop
result = app.invoke(initial_state)

print(f"Steps: {result['step']}")
print(f"Stop reason: {result.get('stop_reason')}")
print(f"Final eval score: {result['eval_signal'].score:.4f}")
```

### One-liner Constructors

```python
from synthetic_teleology.graph import create_teleological_agent

app, state = create_teleological_agent(
    target_values=(10.0, 20.0),
    perceive_fn=lambda: env.observe(),
    transition_fn=lambda a: env.step(a) if a else None,
    max_steps=50,
)
result = app.invoke(state)
```

### Streaming

```python
from synthetic_teleology.graph import collect_stream_events

stream = app.stream(initial_state, stream_mode="updates")
events = collect_stream_events(stream)
for ev in events:
    print(f"[{ev['node']}] step={ev.get('step', '?')}")
```

### Multi-Agent Coordination

```python
from synthetic_teleology.graph import AgentConfig, build_multi_agent_graph

configs = [
    AgentConfig(agent_id="explorer", goal=goal_a, perceive_fn=obs_a, ...),
    AgentConfig(agent_id="conservator", goal=goal_b, perceive_fn=obs_b, ...),
]
app = build_multi_agent_graph(configs, max_rounds=5)
result = app.invoke({...})
```

## Examples

| # | File | Demonstrates |
|---|------|-------------|
| 01 | `01_langgraph_basic_loop.py` | Basic teleological loop as StateGraph, `.invoke()`, inspecting state |
| 02 | `02_llm_goal_directed_agent.py` | LLM-powered evaluation/planning with streaming |
| 03 | `03_multi_agent_negotiation.py` | Two agents with subgraphs + ConsensusNegotiator |
| 04 | `04_human_in_the_loop.py` | Custom review node for human approval at revision |
| 05 | `05_hierarchical_goals.py` | GoalTree + nested subgraphs, revision propagation |
| 06 | `06_benchmark_measurement.py` | Stream events → measurement bridge |
| 07 | `07_react_research_agent.py` | `create_react_teleological_agent()` with tool metadata |
| 08 | `08_constraint_aware_planning.py` | SafetyChecker + BudgetChecker, streaming constraint checks |

Run any example:

```bash
PYTHONPATH=src python examples/01_langgraph_basic_loop.py
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

## Testing

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -v
```

**449 tests** covering all modules (380 original + 69 graph layer).

## Package Structure

```
src/synthetic_teleology/
├── graph/           # LangGraph teleological loop (NEW in v0.2.0)
│   ├── state.py     # TeleologicalState TypedDict
│   ├── nodes.py     # 8 node functions
│   ├── edges.py     # Conditional routing
│   ├── graph.py     # build_teleological_graph()
│   ├── builder.py   # GraphBuilder fluent API
│   ├── prebuilt.py  # One-liner constructors
│   ├── multi_agent.py  # Multi-agent coordination
│   └── streaming.py # Stream event formatters
├── domain/          # Value objects, entities, aggregates, events, enums
├── services/        # Evaluation, goal revision, planning, constraints, loop (legacy)
├── agents/          # Base, teleological, BDI, LLM agents + factory
├── environments/    # Numeric, resource, research, shared environments
├── measurement/     # Collector, 7 metrics, engine, reports, 4 benchmarks
├── infrastructure/  # EventBus, registry, config, serialization, LLM providers
├── presentation/    # Console dashboard, plots, export (JSON/CSV/HTML)
└── cli.py           # Command-line interface
```

## Legacy API

The v0.1.0 `SyncAgenticLoop` and `AsyncAgenticLoop` classes still work but emit a `DeprecationWarning`. Prefer the LangGraph API:

```python
# Old (deprecated)
from synthetic_teleology.services.loop import SyncAgenticLoop
loop = SyncAgenticLoop(evaluator=..., planner=..., ...)
result = loop.run(goal)

# New (recommended)
from synthetic_teleology.graph import GraphBuilder
app, state = GraphBuilder("agent").with_objective(...).with_environment(...).build()
result = app.invoke(state)
```

## License

MIT
