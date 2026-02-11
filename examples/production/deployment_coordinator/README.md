# Deployment Coordinator

Multi-agent deployment orchestration using Synthetic Teleology's
`build_multi_agent_graph()` with LLM-powered negotiation.

## Scenario

Three agents with competing priorities coordinate a software deployment:

| Agent | Goal | Conflict |
|-------|------|----------|
| **Release Agent** | Ship release v2.5 on schedule | Wants speed |
| **Security Agent** | Zero critical vulnerabilities | Wants thoroughness |
| **SRE Agent** | Maintain 99.9% uptime | Wants stability |

At round 2, the Security agent discovers **CVE-2024-31337** (a critical RCE
in the api-gateway). This triggers:

1. Security agent **revises its goal** to prioritize immediate patching
2. **Negotiation** produces consensus: pause production, patch, verify
3. SRE agent **scales capacity** to maintain SLA during the incident
4. Release agent **delays production promotion** until security clears

## Architecture

```
build_multi_agent_graph()
    |
    v
+-- agent_0 (release) -----> subgraph: perceive -> eval -> plan -> act
+-- agent_1 (security) ----> subgraph: perceive -> eval -> revise -> plan -> act
+-- agent_2 (sre) ----------> subgraph: perceive -> eval -> plan -> act
    |
    v
negotiate (LLM: propose x3 -> critique -> synthesize)
    |
    v
loop back to agents with shared_direction
```

All three agents share a single `DeploymentState` instance. Each agent's
perceive function formats the state from its own perspective. Actions taken
by one agent (e.g., security patching a CVE) are visible to all others on
the next perceive cycle.

## Running

```bash
# Simulated mode (no API key required)
PYTHONPATH=src python -m examples.production.deployment_coordinator.main

# With real LLM
ANTHROPIC_API_KEY=sk-... PYTHONPATH=src python -m examples.production.deployment_coordinator.main

# Options
PYTHONPATH=src python -m examples.production.deployment_coordinator.main --rounds 3 --verbose
```

## Files

| File | Purpose |
|------|---------|
| `models.py` | Domain dataclasses: CVEReport, ServiceHealth, DeploymentState |
| `tools.py` | Per-agent tools that mutate the shared DeploymentState |
| `strategies.py` | Per-agent observation formatters |
| `agent.py` | Mock setup, environment wiring, `build_deployment_coordinator()` |
| `main.py` | CLI entry point with formatted output |

## Key Patterns

- **Shared mutable state**: All agents read/write the same `DeploymentState`
- **Custom perceive_fn**: Each agent sees the state through its own lens
- **CVE-triggered revision**: Score <= -0.3 triggers LLM goal revision
- **Negotiation consensus**: Propose-critique-synthesize protocol coordinates response
- **No `from __future__ import annotations`**: Required by LangGraph TypedDict resolution
