# Synthetic Teleology — LLM-First LangGraph Toolkit

LangGraph toolkit for building **LLM-powered goal-directed agents** implementing Haidemariam (2026) *"From the logic of coordination to goal-directed reasoning"* — the theory of **Synthetic Teleology** in Agentic AI.

## Overview

Synthetic Teleology provides an **LLM-first, probabilistic architecture** for building agents that can:

- **Evaluate** progress using LLM reasoning with structured output (or pluggable numeric evaluators)
- **Revise** goals dynamically — LLM reasons about whether and how to adapt goals
- **Plan** with multi-hypothesis generation — N candidate plans scored by confidence
- **Check constraints** via soft reasoning with severity scores (not just boolean predicates)
- **Reflect** on full reasoning traces to decide whether to continue or adjust strategy
- **Coordinate** multiple agents through LLM-powered negotiation
- **Measure** behavior against 8 teleological metrics (including LLM-specific ReasoningQuality)

The teleological loop runs as a **LangGraph StateGraph**: **Perceive → Evaluate → Revise → Plan → Filter → Act → Reflect**.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     LANGGRAPH LAYER (v1.0)                        │
│  StateGraph, LLM Nodes, Probabilistic Edges, Builder, Streaming  │
├──────────────────────────────────────────────────────────────────┤
│                  LLM REASONING SERVICES (NEW)                     │
│  LLMEvaluator, LLMPlanner, LLMReviser, LLMConstraintChecker      │
│  (structured output via with_structured_output / tool-calling)    │
├──────────────────────────────────────────────────────────────────┤
│                 TELEOLOGY DOMAIN (adapted)                        │
│  Goal (text+optional vector), EvalSignal, Hypothesis, Events      │
├──────────────────────────────────────────────────────────────────┤
│                 NUMERIC STRATEGY SERVICES (preserved)             │
│  NumericEvaluator, GreedyPlanner, ThresholdUpdater, etc.          │
├──────────────────────────────────────────────────────────────────┤
│                 MEASUREMENT DOMAIN                                │
│  8 Metrics (incl. ReasoningQuality), Benchmarks, Reports          │
├──────────────────────────────────────────────────────────────────┤
│                 INFRASTRUCTURE                                    │
│  LangChain BaseChatModel, EventBus, Registry, Config              │
└──────────────────────────────────────────────────────────────────┘
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

## Quick Start — LLM Mode (v1.0)

```python
from langchain_anthropic import ChatAnthropic  # or ChatOpenAI, ChatOllama, etc.
from synthetic_teleology.graph import GraphBuilder

# Build an LLM-powered teleological agent
app, initial_state = (
    GraphBuilder("my-agent")
    .with_model(ChatAnthropic(model="claude-sonnet-4-5-20250929"))
    .with_goal(
        "Increase team productivity by 25%",
        criteria=["Identify bottlenecks", "Propose actionable solutions"],
    )
    .with_constraints("No layoffs", "Implement within 1 quarter")
    .with_max_steps(10)
    .with_num_hypotheses(3)             # multi-hypothesis planning
    .build()
)

# Run the teleological loop
result = app.invoke(initial_state)

print(f"Steps: {result['step']}")
print(f"Stop reason: {result.get('stop_reason')}")
print(f"Eval score: {result['eval_signal'].score:.4f}")
print(f"Reasoning: {result['eval_signal'].reasoning[:200]}")
```

### One-liner Constructors

```python
from synthetic_teleology.graph import create_llm_agent

app, state = create_llm_agent(
    model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
    goal="Analyze market trends and propose investment strategy",
    criteria=["Risk-adjusted returns > 8%"],
    constraints=["Max drawdown < 15%"],
    max_steps=10,
)
result = app.invoke(state)
```

### Numeric Mode (backward compatible)

```python
from synthetic_teleology.graph import GraphBuilder
from synthetic_teleology.environments.numeric import NumericEnvironment

env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))

app, state = (
    GraphBuilder("numeric-agent")
    .with_objective((5.0, 5.0))
    .with_max_steps(20)
    .with_environment(
        perceive_fn=lambda: env.observe(),
        act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
        transition_fn=lambda a: env.step(a) if a else None,
    )
    .build()
)
result = app.invoke(state)
```

### Multi-Agent Coordination

```python
from synthetic_teleology.graph import AgentConfig, build_multi_agent_graph

configs = [
    AgentConfig(
        agent_id="growth", goal="Maximize revenue growth",
        model=model, criteria=["Growth rate > 10%"],
    ),
    AgentConfig(
        agent_id="retention", goal="Reduce churn below 5%",
        model=model, criteria=["Churn < 5%"],
    ),
]
app = build_multi_agent_graph(configs, max_rounds=3)
result = app.invoke({...})
```

## Examples

### LLM Mode (NEW in v1.0)

| # | File | Demonstrates |
|---|------|-------------|
| 11 | `conceptual/11_llm_quickstart.py` | LLM agent with natural language goals, multi-hypothesis planning |
| 12 | `conceptual/12_llm_tools.py` | LLM agent with LangChain tools for actions |
| 13 | `conceptual/13_llm_multi_agent.py` | Multi-agent LLM coordination with per-agent goals |
| 14 | `conceptual/14_llm_metrics.py` | ReasoningQuality metric + LLM log analysis (no API key needed) |

### Numeric Mode (Conceptual)

| # | File | Demonstrates |
|---|------|-------------|
| 01 | `conceptual/01_basic_loop.py` | Basic teleological loop as StateGraph, `.invoke()`, inspecting state |
| 02 | `conceptual/02_multi_agent.py` | Two agents with subgraphs + ConsensusNegotiator |
| 03 | `conceptual/03_constraints.py` | SafetyChecker + BudgetChecker, streaming constraint checks |
| 04 | `conceptual/04_evaluation_strategies.py` | NumericEvaluator, CompositeEvaluator, ReflectiveEvaluator (drift detection) |
| 05 | `conceptual/05_goal_revision.py` | ThresholdUpdater, GradientUpdater, UncertaintyAwareUpdater, GoalUpdaterChain |
| 06 | `conceptual/06_planning_strategies.py` | GreedyPlanner, StochasticPlanner, HierarchicalPlanner decomposition |
| 07 | `conceptual/07_hierarchical_goals.py` | GoalTree, coherence validation, revision propagation, HierarchicalUpdater |
| 08 | `conceptual/08_environments.py` | ResourceEnvironment (scarcity), ResearchEnvironment (knowledge synthesis) |
| 09 | `conceptual/09_metrics_measurement.py` | All 7 teleological metrics, MetricsEngine, AgentLog, MetricsReport |
| 10 | `conceptual/10_ethical_constraints.py` | EthicalChecker predicates, ConstraintPipeline (fail_fast), PolicyFilter |

```bash
PYTHONPATH=src python examples/conceptual/14_llm_metrics.py   # no API key needed
PYTHONPATH=src python examples/conceptual/11_llm_quickstart.py # requires API key
PYTHONPATH=src python examples/conceptual/01_basic_loop.py     # numeric mode
```

### Production

Full-featured agents that demonstrate real-world usage with custom evaluators, planners, and constraint checkers:

| Agent | Description | Run |
|-------|-------------|-----|
| **Polymarket Trader** | Goal-directed prediction market trading. Aligns portfolio positions with conviction-based probability estimates using Kelly criterion sizing, risk limits, and capital constraints. | `PYTHONPATH=src python -m examples.production.polymarket_trader.main` |
| **Sales SDR** | Goal-directed sales development. Manages lead qualification, selects outreach channels, and tracks conversion funnel metrics with contact frequency and daily limit constraints. | `PYTHONPATH=src python -m examples.production.sales_sdr.main` |

Both agents run in simulated mode by default. Pass `--live` to use real APIs (requires `POLYMARKET_API_KEY` or `HUBSPOT_API_KEY`).

## Metrics

The framework implements 8 metrics (7 from theory + 1 LLM-specific):

| Metric | Abbreviation | Measures |
|--------|-------------|----------|
| Goal Persistence | GP | Fraction of steps maintaining the same goal |
| Teleological Coherence | TC | Correlation between score improvement and goal stability |
| Reflective Efficiency | RE | Cost-weighted score improvement per step |
| Adaptivity | AD | Post-perturbation recovery rate |
| Normative Fidelity | NF | Fraction of steps without constraint violations |
| Innovation Yield | IY | Score improvement from novel actions |
| Lyapunov Stability | LS | Score variance convergence over time |
| **Reasoning Quality** | **RQ** | **Coherence and diversity of LLM reasoning traces (NEW)** |

## Testing

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -v
```

**498 tests** covering all modules (380 original + 69 graph + 49 LLM services).

## Dual Mode Architecture

The framework supports **two modes** detected automatically by the builder:

| Mode | Trigger | Services | Requires API Key |
|------|---------|----------|-----------------|
| **LLM Mode** (new) | `.with_model(model)` | LLMEvaluator, LLMPlanner, LLMReviser, LLMConstraintChecker | Yes |
| **Numeric Mode** (legacy) | `.with_objective(values)` | NumericEvaluator, GreedyPlanner, ThresholdUpdater | No |

## Package Structure

```
src/synthetic_teleology/
├── graph/               # LangGraph teleological loop
│   ├── state.py         # TeleologicalState TypedDict
│   ├── nodes.py         # 8 node functions (LLM + numeric dual-mode)
│   ├── edges.py         # Conditional routing
│   ├── graph.py         # build_teleological_graph()
│   ├── builder.py       # GraphBuilder fluent API (LLM-first + numeric)
│   ├── prebuilt.py      # create_llm_agent(), create_numeric_agent()
│   ├── multi_agent.py   # Multi-agent coordination (LLM + numeric)
│   └── streaming.py     # Stream event formatters
├── services/
│   ├── llm_evaluation.py   # LLMEvaluator (structured output) — NEW
│   ├── llm_planning.py     # LLMPlanner (multi-hypothesis) — NEW
│   ├── llm_revision.py     # LLMReviser (LLM goal revision) — NEW
│   ├── llm_constraints.py  # LLMConstraintChecker (soft reasoning) — NEW
│   ├── evaluation.py       # NumericEvaluator, CompositeEvaluator, etc.
│   ├── planning.py         # GreedyPlanner, StochasticPlanner, etc.
│   ├── goal_revision.py    # ThresholdUpdater, GradientUpdater, etc.
│   └── constraint_engine.py # ConstraintPipeline, PolicyFilter
├── domain/              # Goal (text+vector), EvalSignal, Hypothesis, events
├── environments/        # Numeric, resource, research, shared environments
├── measurement/         # Collector, 8 metrics, engine, reports, benchmarks
├── infrastructure/      # EventBus, registry, config, LangChain bridge
├── agents/              # Base, teleological, BDI, LLM agents + factory
├── presentation/        # Console dashboard, plots, export
└── cli.py               # Command-line interface
```

## License

MIT
