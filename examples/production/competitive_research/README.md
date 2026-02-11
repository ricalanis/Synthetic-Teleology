# Competitive Research Analyst

Goal-directed competitive intelligence agent built with Synthetic Teleology.

## What It Does

Builds a comprehensive competitive analysis of a target company (TechRival Inc)
using simulated research tools. Partway through the analysis, a major strategic
pivot is discovered, triggering automatic goal revision to incorporate the new
intelligence.

## Why It Matters

Demonstrates how a teleological agent adapts its goals in real time when the
environment changes. Unlike static research pipelines, the agent revises its
success criteria mid-execution and redirects its research plan accordingly.

## How to Run

```bash
# Simulated mode (no API key needed)
PYTHONPATH=src python -m examples.production.competitive_research.main

# With a real LLM
ANTHROPIC_API_KEY=sk-... PYTHONPATH=src python -m examples.production.competitive_research.main

# Options
PYTHONPATH=src python -m examples.production.competitive_research.main --steps 25 --verbose
```

## Features Exercised

- **Goal Revision**: Evaluator detects pivot, score drops below -0.3, LLM reviser
  adds new criteria ("analyze AI pivot"), planner generates post-revision actions.
- **Tool Routing**: Three simulated tools (web_search, document_reader, data_extractor)
  with `tool_name` in `ActionProposal` mapped to LangChain-compatible tool objects.
- **Knowledge Store**: Seed context and audit cross-references stored in
  `KnowledgeStore` for metacognitive commons.
- **Audit Trail**: `GoalAuditTrail` records every goal revision with provenance,
  eval score, and reasoning for post-hoc analysis.
- **Custom Evaluator + LLM Planner**: Hybrid pattern where evaluation is
  deterministic (reads simulated state) while planning uses LLM reasoning.
- **Constraint Checking**: `SourceDiversityChecker` ensures no single source
  type dominates the research findings.

## Architecture

```
WebSearchTool --------+
DocumentReaderTool ----+---> ResearchState (shared mutable env)
DataExtractorTool -----+         |
                                 v
                        ResearchEvaluator (custom, deterministic)
                                 |
                           eval score <= -0.3?
                            /            \
                          no              yes
                          |                |
                       LLMPlanner      LLMReviser -> goal revision
                          |                |
                          +--- act_node ---+
```
