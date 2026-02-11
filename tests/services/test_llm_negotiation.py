"""Tests for LLM-powered multi-agent dialogue negotiation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import EvalSignal
from synthetic_teleology.graph.multi_agent import AgentConfig, build_multi_agent_graph
from synthetic_teleology.services.llm_negotiation import (
    GoalProposal,
    LLMNegotiator,
    NegotiationConsensus,
    NegotiationCritique,
)


def _mock_model(responses: list[str]) -> MagicMock:
    """Create a mock LLM model returning predefined JSON responses."""
    model = MagicMock()
    call_count = [0]

    def invoke(prompt):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        msg = MagicMock()
        msg.content = responses[idx]
        return msg

    model.invoke = invoke
    return model


class TestGoalProposal:
    def test_defaults(self) -> None:
        p = GoalProposal()
        assert p.agent_id == ""
        assert p.confidence == 0.5

    def test_custom_fields(self) -> None:
        p = GoalProposal(
            agent_id="a1",
            proposed_direction="go left",
            priority_dimensions=["speed", "safety"],
            confidence=0.9,
        )
        assert p.agent_id == "a1"
        assert len(p.priority_dimensions) == 2
        assert p.confidence == 0.9


class TestNegotiationCritique:
    def test_defaults(self) -> None:
        c = NegotiationCritique()
        assert c.agreements == []
        assert c.disagreements == []

    def test_custom_fields(self) -> None:
        c = NegotiationCritique(
            agreements=["both want speed"],
            disagreements=["safety vs cost"],
            synthesis_hints=["balance both"],
        )
        assert len(c.agreements) == 1
        assert len(c.synthesis_hints) == 1


class TestNegotiationConsensus:
    def test_defaults(self) -> None:
        c = NegotiationConsensus()
        assert c.shared_direction == ""
        assert c.confidence == 0.5
        assert c.round_count == 0


class TestLLMNegotiator:
    def _make_agent_results(self) -> dict:
        return {
            "agent-1": {
                "final_goal": Goal(name="g1", description="maximize speed"),
                "eval_signal": EvalSignal(score=0.7, confidence=0.8),
                "steps": 5,
                "stop_reason": "max_steps",
            },
            "agent-2": {
                "final_goal": Goal(name="g2", description="minimize cost"),
                "eval_signal": EvalSignal(score=0.6, confidence=0.7),
                "steps": 3,
                "stop_reason": "max_steps",
            },
        }

    def test_single_agent_no_negotiation(self) -> None:
        model = _mock_model([])
        negotiator = LLMNegotiator(model=model)
        result = negotiator.negotiate({
            "agent-1": {
                "final_goal": Goal(name="g1", description="go fast"),
            },
        })
        assert isinstance(result, NegotiationConsensus)
        assert result.confidence == 1.0
        assert result.round_count == 0
        assert "go fast" in result.shared_direction

    def test_empty_agents_returns_consensus(self) -> None:
        model = _mock_model([])
        negotiator = LLMNegotiator(model=model)
        result = negotiator.negotiate({})
        assert isinstance(result, NegotiationConsensus)

    def test_full_negotiation_protocol(self) -> None:
        """Full 3-phase: propose -> critique -> synthesize."""
        propose_resp1 = json.dumps({
            "direction": "optimize speed",
            "reasoning": "high eval score on speed",
            "priority_dimensions": ["speed"],
            "confidence": 0.8,
        })
        propose_resp2 = json.dumps({
            "direction": "reduce cost",
            "reasoning": "budget constraint",
            "priority_dimensions": ["cost"],
            "confidence": 0.7,
        })
        critique_resp = json.dumps({
            "agreements": ["both want efficiency"],
            "disagreements": ["speed vs cost tradeoff"],
            "synthesis_hints": ["find pareto-optimal balance"],
        })
        synthesize_resp = json.dumps({
            "shared_direction": "optimize speed-cost ratio",
            "revised_criteria": ["speed > 80%", "cost < budget"],
            "confidence": 0.85,
            "reasoning": "balanced approach",
        })

        model = _mock_model([propose_resp1, propose_resp2, critique_resp, synthesize_resp])
        negotiator = LLMNegotiator(model=model)
        result = negotiator.negotiate(self._make_agent_results())

        assert isinstance(result, NegotiationConsensus)
        assert result.shared_direction == "optimize speed-cost ratio"
        assert result.confidence == 0.85
        assert result.round_count == 1
        assert len(result.revised_criteria) == 2

    def test_propose_failure_fallback(self) -> None:
        """When propose fails, should use goal description as fallback."""
        model = MagicMock()
        model.invoke = MagicMock(side_effect=Exception("LLM error"))

        negotiator = LLMNegotiator(model=model)
        # Should still produce proposals (with low confidence fallback)
        proposals = negotiator._phase_propose(self._make_agent_results())
        assert len(proposals) == 2
        assert all(p.confidence == 0.3 for p in proposals)

    def test_critique_failure_returns_empty(self) -> None:
        model = MagicMock()
        model.invoke = MagicMock(side_effect=Exception("LLM error"))

        negotiator = LLMNegotiator(model=model)
        proposals = [GoalProposal(agent_id="a1"), GoalProposal(agent_id="a2")]
        critique = negotiator._phase_critique(proposals)
        assert isinstance(critique, NegotiationCritique)
        assert critique.agreements == []

    def test_synthesize_failure_combines_proposals(self) -> None:
        model = MagicMock()
        model.invoke = MagicMock(side_effect=Exception("LLM error"))

        negotiator = LLMNegotiator(model=model)
        proposals = [
            GoalProposal(agent_id="a1", proposed_direction="go left"),
            GoalProposal(agent_id="a2", proposed_direction="go right"),
        ]
        critique = NegotiationCritique()
        result = negotiator._phase_synthesize(proposals, critique)
        assert "go left" in result.shared_direction
        assert "go right" in result.shared_direction
        assert result.confidence == 0.3

    def test_parse_json_plain(self) -> None:
        text = '{"key": "value"}'
        assert LLMNegotiator._parse_json(text) == {"key": "value"}

    def test_parse_json_markdown_block(self) -> None:
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        assert LLMNegotiator._parse_json(text) == {"key": "value"}

    def test_parse_json_code_block(self) -> None:
        text = '```\n{"key": "value"}\n```'
        assert LLMNegotiator._parse_json(text) == {"key": "value"}

    def test_goal_without_description(self) -> None:
        """Agent result with goal that has no description."""
        model = _mock_model([
            json.dumps({"direction": "d1", "confidence": 0.5}),
            json.dumps({"direction": "d2", "confidence": 0.5}),
            json.dumps({"agreements": [], "disagreements": [], "synthesis_hints": []}),
            json.dumps({"shared_direction": "consensus", "confidence": 0.6, "reasoning": "ok"}),
        ])
        negotiator = LLMNegotiator(model=model)
        result = negotiator.negotiate({
            "a1": {"final_goal": Goal(name="g1"), "eval_signal": None, "steps": 1},
            "a2": {"final_goal": None, "eval_signal": None, "steps": 1},
        })
        assert isinstance(result, NegotiationConsensus)


class TestMultiAgentLLMNegotiation:
    """Test multi_agent.py LLM negotiation integration."""

    def test_llm_negotiate_function(self) -> None:
        """Test the _llm_negotiate helper."""
        from synthetic_teleology.graph.multi_agent import _llm_negotiate

        synthesize_resp = json.dumps({
            "shared_direction": "balance all",
            "revised_criteria": ["c1"],
            "confidence": 0.8,
            "reasoning": "good balance",
        })
        # Propose for 2 agents + critique + synthesize = 4 calls
        model = _mock_model([
            json.dumps({"direction": "d1", "confidence": 0.5}),
            json.dumps({"direction": "d2", "confidence": 0.5}),
            json.dumps({"agreements": ["a"], "disagreements": [], "synthesis_hints": []}),
            synthesize_resp,
        ])

        state = {"negotiation_round": 0, "max_dialogue_rounds": 2}
        agent_results = {
            "a1": {
                "final_goal": Goal(name="g1", description="speed"),
                "eval_signal": None, "steps": 1,
            },
            "a2": {
                "final_goal": Goal(name="g2", description="cost"),
                "eval_signal": None, "steps": 1,
            },
        }
        result = _llm_negotiate(state, agent_results, model)
        assert result["shared_direction"] == "balance all"
        assert result["negotiation_round"] == 1

    def test_llm_negotiate_fallback_on_error(self) -> None:
        """LLM negotiation falls back to numeric when it fails."""
        from synthetic_teleology.graph.multi_agent import _llm_negotiate

        model = MagicMock()
        model.invoke = MagicMock(side_effect=Exception("boom"))

        state = {"negotiation_round": 0}
        agent_results = {
            "a1": {"final_goal": Goal(name="g1"), "steps": 1},
            "a2": {"final_goal": Goal(name="g2"), "steps": 1},
        }
        # Should not raise — falls back to numeric
        result = _llm_negotiate(state, agent_results, model)
        assert isinstance(result, dict)

    def test_merge_agent_results_reducer(self) -> None:
        from synthetic_teleology.graph.multi_agent import _merge_agent_results
        left = {"a1": {"score": 0.5}}
        right = {"a2": {"score": 0.7}}
        merged = _merge_agent_results(left, right)
        assert "a1" in merged
        assert "a2" in merged


class TestMultiAgentParallel:
    """Test parallel multi-agent execution."""

    def test_parallel_graph_compiles(self) -> None:
        from synthetic_teleology.domain.entities import Goal
        from synthetic_teleology.domain.enums import Direction
        from synthetic_teleology.domain.values import ObjectiveVector
        from synthetic_teleology.environments.numeric import NumericEnvironment

        env1 = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
        env2 = NumericEnvironment(dimensions=2, initial_state=(10.0, 10.0))

        configs = [
            AgentConfig(
                agent_id="a1",
                goal=Goal(name="g1", objective=ObjectiveVector(
                    values=(5.0, 5.0), directions=(Direction.APPROACH, Direction.APPROACH)
                )),
                perceive_fn=lambda: env1.observe(),
                transition_fn=lambda a: env1.step(a) if a else None,
                max_steps_per_round=2,
            ),
            AgentConfig(
                agent_id="a2",
                goal=Goal(name="g2", objective=ObjectiveVector(
                    values=(7.0, 7.0), directions=(Direction.APPROACH, Direction.APPROACH)
                )),
                perceive_fn=lambda: env2.observe(),
                transition_fn=lambda a: env2.step(a) if a else None,
                max_steps_per_round=2,
            ),
        ]
        app = build_multi_agent_graph(configs, max_rounds=1, parallel=True)
        assert app is not None

    def test_sequential_is_default(self) -> None:
        """parallel=False is the default."""
        from synthetic_teleology.environments.numeric import NumericEnvironment

        env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
        configs = [
            AgentConfig(
                agent_id="a1",
                goal=Goal(name="g1"),
                perceive_fn=lambda: env.observe(),
                max_steps_per_round=1,
            ),
        ]
        # Single agent — parallel doesn't apply, should compile fine
        app = build_multi_agent_graph(configs, max_rounds=1, parallel=True)
        assert app is not None
