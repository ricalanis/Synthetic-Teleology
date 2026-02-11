# Investment Thesis Builder

Goal-directed investment analysis agent that builds comprehensive investment theses using simulated financial research tools. Demonstrates the teleological loop's ability to handle mid-analysis goal revision when material information (a lawsuit) is discovered.

## Architecture

The agent uses **LLM mode** (`with_model` + `with_goal`) with a **hybrid mock** pattern:

- **Custom evaluator** (`ThesisEvaluator`) reads simulated environment state (`ThesisState`) for deterministic scoring
- **LLM planner + reviser** served by `MockStructuredChatModel` which provides `PlanningOutput` and `RevisionOutput` responses
- **Simulated tools** with `.name`, `.description`, `.invoke(params)` that mutate the shared `ThesisState`

### Tools

| Tool | Purpose |
|------|---------|
| `FinancialDataTool` | Retrieve revenue, margins, growth rates |
| `NewsSearchTool` | Financial news search; triggers lawsuit at call 3+ |
| `FilingAnalyzerTool` | Analyze SEC filings (10-K, 10-Q, proxy) |
| `SentimentTool` | Text sentiment analysis; detects litigation language |

### Goal Revision Trigger

At **step 8**, the `NewsSearchTool` (on its 3rd call) returns lawsuit news and sets `lawsuit_discovered=True`. On **step 9**, the `ThesisEvaluator` detects this condition (lawsuit discovered but not yet analyzed) and returns a score of **-0.5**, which is below the revision threshold of **-0.3**. This triggers the `LLMReviser` which:

1. Adds "Analyze litigation risk and potential financial impact" to success criteria
2. Updates the goal description to include "with litigation risk assessment"
3. The agent continues research with the expanded scope

## Running

```bash
# Simulated mode (no API key required)
PYTHONPATH=src python -m examples.production.investment_thesis.main

# With verbose step-by-step output
PYTHONPATH=src python -m examples.production.investment_thesis.main --verbose

# Custom step limit
PYTHONPATH=src python -m examples.production.investment_thesis.main --steps 20

# Real LLM mode (set one of these)
ANTHROPIC_API_KEY=sk-... PYTHONPATH=src python -m examples.production.investment_thesis.main
OPENAI_API_KEY=sk-... PYTHONPATH=src python -m examples.production.investment_thesis.main
```

## File Structure

```
investment_thesis/
  __init__.py       # Package docstring
  models.py         # Domain dataclasses: FinancialMetric, NewsItem, ThesisState
  tools.py          # Simulated LangChain-compatible research tools
  strategies.py     # ThesisEvaluator, SourceDiversityChecker
  agent.py          # Graph wiring, mock model, build_thesis_agent()
  main.py           # CLI entry point with argparse
  README.md         # This file
```

## Architecture

```
FinancialDataTool -----+
NewsSearchTool ---------+---> ThesisState (shared mutable env)
FilingAnalyzerTool -----+         |
SentimentTool ----------+         v
                            ThesisEvaluator (custom, deterministic)
                                     |
                               eval score <= -0.3?
                                /            \
                              no              yes
                              |                |
                           LLMPlanner      LLMReviser -> goal revision
                              |                |
                              +--- act_node ---+
```

## Key Patterns Demonstrated

- **Hybrid mock**: custom evaluator for deterministic scoring + mock LLM for planning/revision
- **Goal revision**: material information discovery triggers automatic goal expansion
- **Tool integration**: LangChain-compatible tools with `.name`, `.description`, `.invoke()`
- **Working memory**: `WorkingMemory` accumulates tool results for progressive context
- **Knowledge store**: shared metacognitive commons for cross-referencing
- **Audit trail**: serializable revision history tracking goal evolution
