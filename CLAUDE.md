# Synthetic Teleology — Project Rules

## Documentation Style
- README.md is a complete descriptive document of the tool. Never changeloggy or version-oriented.
- Version-specific changes go in docs/changelog.md only.
- Architecture decisions go in docs/decisions.md.
- Known issues go in docs/known_issues.md.

## Code Conventions
- Never use `from __future__ import annotations` in TypedDict files consumed by LangGraph.
- Always use `PYTHONPATH=src` for tests and examples.
- Use `MockStructuredChatModel` from tests/helpers/mock_llm.py (not FakeListChatModel).
- Goal.revise() returns tuple[Goal, GoalRevision] — always unpack.
- StateSnapshot requires timestamp field.
- BDI terminology replaced with "Intentional State Mapping". Use IntentionalStateAgent, not BDIAgent.

## Architecture
- Graph API (GraphBuilder) is primary. Agent class hierarchy is reference implementation.
- Environments follow PyTorch philosophy: state_dict()/load_state_dict(), EnvironmentWrapper.
- Strategies injected via closures, not stored in state (enables checkpointing).
- Accumulation channels can be bounded via `make_bounded_state(max_history)` / `with_max_history()`.
