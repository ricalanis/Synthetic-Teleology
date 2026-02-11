# Synthetic Teleology

LangGraph implementation of Haidemariam (2026) **Synthetic Teleology** for building goal-directed AI agents with recursive self-evaluation.

## What is Synthetic Teleology?

Synthetic Teleology is the **engineered capacity of AI systems to generate, pursue, and revise goals through recursive self-evaluation** (Haidemariam, 2026). Unlike reactive or purely reward-driven agents, teleological agents maintain an internal goal-directed loop that continuously evaluates progress, adapts strategy, and revises objectives when needed.

The core of the theory is a **recursive goal maintenance loop** (Equations 1–6):

```
Perceive → Evaluate → Revise → Plan → Filter → Act → Transition
    ↑                                                      │
    └──────────────────────────────────────────────────────┘
```

Each cycle, the agent:
1. **Perceives** the environment state *S_t*
2. **Evaluates** goal progress via scoring function *E(G_t, S_t)*
3. **Revises** the goal *G_t → G_{t+1}* if evaluation warrants change
4. **Plans** a policy *π_t* mapping states to actions
5. **Filters** the plan through constraints *C(π_t)*
6. **Acts** in the environment to produce *S_{t+1}*
7. **Transitions** to the next cycle with updated state

The framework grounds agent design in four **pillars of agency**: Intentionality (goal-directedness), Autonomy (self-governance), Adaptivity (goal revision under uncertainty), and Sociality (multi-agent coordination).

> Haidemariam, Y. A. (2026). From the logic of coordination to goal-directed reasoning: Synthetic teleology in agentic AI. *Frontiers in Artificial Intelligence*, 9, 1592432.

## Features

- **LLM-powered evaluation, planning, revision, and constraint checking** — structured output via Pydantic schemas
- **Multi-hypothesis planning** with softmax selection over confidence scores
- **Soft constraint reasoning** with severity scores and suggested mitigations
- **Goal revision** with provenance tracking and audit trail
- **Multi-agent coordination** with negotiation protocols (propose → critique → synthesize)
- **Parallel agent execution** via LangGraph Send API
- **8 teleological metrics** — 7 from the paper + ReasoningQuality for LLM traces
- **Dual-mode architecture** — LLM reasoning or numeric computation, auto-detected by the builder
- **Self-contained examples** — all run without API keys using built-in mock models
- **Active inference** via expected free energy decomposition
- **Self-modeling evaluation** with surprise detection and confidence adjustment
- **Evolving constraints** and distributed intentional grounding
- **BDI bridge** — classical Belief-Desire-Intention mapping to LangGraph nodes
- **LangGraph checkpointing support** — strategies injected via closures, not stored in state
- **Configurable timeouts** on all LLM services with fail-safe error handling
- **Constraint severity model** — opt-in `ConstraintResult` with severity scores and mitigations
- **Co-evolutionary constraint wiring** — `EvolvingConstraintManager` integrated into the graph loop

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       LANGGRAPH LAYER                            │
│  StateGraph, LLM Nodes, Probabilistic Edges, Builder, Streaming │
├──────────────────────────────────────────────────────────────────┤
│                    LLM REASONING SERVICES                        │
│  LLMEvaluator, LLMPlanner, LLMReviser, LLMConstraintChecker     │
│  (structured output via with_structured_output / tool-calling)   │
├──────────────────────────────────────────────────────────────────┤
│                    TELEOLOGY DOMAIN                               │
│  Goal (text + optional vector), EvalSignal, Hypothesis, Events   │
├──────────────────────────────────────────────────────────────────┤
│                    NUMERIC STRATEGY SERVICES                      │
│  NumericEvaluator, GreedyPlanner, ThresholdUpdater, etc.         │
├──────────────────────────────────────────────────────────────────┤
│                    MEASUREMENT                                    │
│  8 Metrics, MetricsEngine, Benchmarks, Reports                   │
├──────────────────────────────────────────────────────────────────┤
│                    INFRASTRUCTURE                                 │
│  LangChain BaseChatModel, EventBus, Registry, Config             │
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

## Quick Start

### LLM Agent

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

#### One-liner Constructor

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

### Numeric Agent

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

All examples are self-contained and run without API keys using built-in mock models. When a real API key is detected, examples automatically switch to the live LLM.

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
| 11 | `conceptual/11_llm_quickstart.py` | LLM agent with natural language goals, multi-hypothesis planning |
| 12 | `conceptual/12_llm_tools.py` | LLM agent with LangChain tools for actions |
| 13 | `conceptual/13_llm_multi_agent.py` | Multi-agent LLM coordination with per-agent goals |
| 14 | `conceptual/14_llm_metrics.py` | ReasoningQuality metric + LLM log analysis |

```bash
PYTHONPATH=src python examples/conceptual/14_llm_metrics.py
PYTHONPATH=src python examples/conceptual/11_llm_quickstart.py
PYTHONPATH=src python examples/conceptual/01_basic_loop.py
```

### Production

Full-featured agents with custom evaluators, planners, and constraint checkers:

| Agent | Description | Run |
|-------|-------------|-----|
| **Polymarket Trader** | Goal-directed prediction market trading. Aligns portfolio positions with conviction-based probability estimates using Kelly criterion sizing, risk limits, and capital constraints. | `PYTHONPATH=src python -m examples.production.polymarket_trader.main` |
| **Sales SDR** | Goal-directed sales development. Manages lead qualification, selects outreach channels, and tracks conversion funnel metrics with contact frequency and daily limit constraints. | `PYTHONPATH=src python -m examples.production.sales_sdr.main` |

Both agents run in simulated mode by default. Pass `--live` to use real APIs (requires `POLYMARKET_API_KEY` or `HUBSPOT_API_KEY`).

## Metrics

The framework implements 8 teleological metrics (7 from the paper + ReasoningQuality):

| Metric | Abbreviation | Measures |
|--------|-------------|----------|
| Goal Persistence | GP | Fraction of steps maintaining the same goal |
| Teleological Coherence | TC | Correlation between score improvement and goal stability |
| Reflective Efficiency | RE | Cost-weighted score improvement per step |
| Adaptivity | AD | Post-perturbation recovery rate |
| Normative Fidelity | NF | Fraction of steps without constraint violations |
| Innovation Yield | IY | Score improvement from novel actions |
| Lyapunov Stability | LS | Score variance convergence over time |
| Reasoning Quality | RQ | Coherence and diversity of LLM reasoning traces |

## Testing

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -v
```

**656 tests** covering all modules.

## Dual-Mode Architecture

The framework supports two modes detected automatically by the builder:

| Mode | Trigger | Services | Requires API Key |
|------|---------|----------|-----------------|
| **LLM Mode** | `.with_model(model)` | LLMEvaluator, LLMPlanner, LLMReviser, LLMConstraintChecker | Yes |
| **Numeric Mode** | `.with_objective(values)` | NumericEvaluator, GreedyPlanner, ThresholdUpdater | No |

## Package Structure

```
src/synthetic_teleology/
├── graph/               # LangGraph teleological loop
│   ├── state.py         # TeleologicalState TypedDict
│   ├── nodes.py         # 8 node functions (LLM + numeric dual-mode)
│   ├── edges.py         # Conditional routing
│   ├── graph.py         # build_teleological_graph()
│   ├── builder.py       # GraphBuilder fluent API
│   ├── prebuilt.py      # create_llm_agent(), create_numeric_agent()
│   ├── multi_agent.py   # Multi-agent coordination
│   └── streaming.py     # Stream event formatters
├── services/
│   ├── llm_evaluation.py   # LLMEvaluator (structured output)
│   ├── llm_planning.py     # LLMPlanner (multi-hypothesis)
│   ├── llm_revision.py     # LLMReviser (LLM goal revision)
│   ├── llm_constraints.py  # LLMConstraintChecker (soft reasoning)
│   ├── evaluation.py       # NumericEvaluator, CompositeEvaluator, etc.
│   ├── planning.py         # GreedyPlanner, StochasticPlanner, etc.
│   ├── goal_revision.py    # ThresholdUpdater, GradientUpdater, etc.
│   └── constraint_engine.py # ConstraintPipeline, PolicyFilter
├── domain/              # Goal (text + vector), EvalSignal, Hypothesis, events
├── environments/        # Numeric, resource, research, shared environments
├── measurement/         # Collector, 8 metrics, engine, reports, benchmarks
├── infrastructure/      # EventBus, registry, config, LangChain bridge
├── agents/              # Base, teleological, BDI, LLM agents + factory
├── presentation/        # Console dashboard, plots, export
└── cli.py               # Command-line interface
```

## License

MIT
