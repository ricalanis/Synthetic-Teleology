"""Graph wiring for the Deployment Coordinator.

Uses ``build_multi_agent_graph()`` with 3 agents (Release, Security, SRE)
coordinating a deployment.  LLM-powered negotiation resolves conflicts
when a CVE is discovered mid-deployment.

Self-contained: runs with mock LLMs by default (no API key required).
Set ANTHROPIC_API_KEY or OPENAI_API_KEY for real LLM mode.
"""

import json
import os
import time

from synthetic_teleology.domain.values import StateSnapshot
from synthetic_teleology.graph.multi_agent import AgentConfig, build_multi_agent_graph

from .models import DeploymentState
from .strategies import release_observation, security_observation, sre_observation
from .tools import (
    make_apply_patch_tool,
    make_check_sla_tool,
    make_promote_stage_tool,
    make_run_canary_tool,
    make_scale_capacity_tool,
    make_scan_vulnerabilities_tool,
)

# ---------------------------------------------------------------------------
# Environment callbacks per agent
# ---------------------------------------------------------------------------


def _make_release_env(state: DeploymentState):
    """Create perceive/transition closures for the Release agent."""
    run_canary = make_run_canary_tool(state)
    promote = make_promote_stage_tool(state)
    step_counter = [0]

    def perceive() -> StateSnapshot:
        return StateSnapshot(
            timestamp=time.time(),
            observation=release_observation(state),
        )

    def transition(action, constraints_context=None) -> None:
        step_counter[0] += 1
        name = getattr(action, "name", str(action))
        params = getattr(action, "parameters", {})

        if name == "run_canary":
            run_canary(params.get("service", "api-gateway"))
        elif name == "promote_stage":
            promote(params.get("target_stage", "staging"))
        # Other actions are no-ops (the planner may propose informational actions)

    return perceive, transition


def _make_security_env(state: DeploymentState):
    """Create perceive/transition closures for the Security agent.

    The perceive function runs a background scan on every call, advancing
    the scan counter.  This ensures the CVE is discovered during perceive
    (pass 2) so the LLM evaluator sees it in its observation, matching
    the mock's bad score on that pass.
    """
    scan = make_scan_vulnerabilities_tool(state)
    patch = make_apply_patch_tool(state)

    def perceive() -> StateSnapshot:
        # Run a passive scan each time we perceive (advances scan counter)
        scan("background-scan")
        return StateSnapshot(
            timestamp=time.time(),
            observation=security_observation(state),
        )

    def transition(action, constraints_context=None) -> None:
        name = getattr(action, "name", str(action))
        params = getattr(action, "parameters", {})

        if name == "scan_vulnerabilities":
            scan(params.get("target", "all"))
        elif name == "apply_patch":
            patch(params.get("cve_id", ""))

    return perceive, transition


def _make_sre_env(state: DeploymentState):
    """Create perceive/transition closures for the SRE agent."""
    check_sla = make_check_sla_tool(state)
    scale = make_scale_capacity_tool(state)

    def perceive() -> StateSnapshot:
        return StateSnapshot(
            timestamp=time.time(),
            observation=sre_observation(state),
        )

    def transition(action, constraints_context=None) -> None:
        name = getattr(action, "name", str(action))
        params = getattr(action, "parameters", {})

        if name == "check_sla":
            check_sla(params.get("service", "api-gateway"))
        elif name == "scale_capacity":
            scale(params.get("service", "api-gateway"), params.get("replicas", 3))

    return perceive, transition


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _get_model():
    """Get a real LLM or None for fallback to mocks."""
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.5)
        except ImportError:
            pass
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o", temperature=0.5)
        except ImportError:
            pass
    return None


def _build_release_mock():
    """Mock LLM for the Release agent.

    3 passes x 1 step each = 3 (EvaluationOutput + PlanningOutput) pairs.
    Scores stay above -0.3 so no revision is triggered.
    """
    from synthetic_teleology.services.llm_evaluation import EvaluationOutput
    from synthetic_teleology.services.llm_planning import (
        ActionProposal,
        PlanHypothesis,
        PlanningOutput,
    )
    from synthetic_teleology.testing import MockStructuredChatModel

    return MockStructuredChatModel(structured_responses=[
        # --- Pass 1 (initial): run canary ---
        EvaluationOutput(
            score=0.15,
            confidence=0.60,
            reasoning=(
                "Deployment is in planning stage with 0% progress. "
                "Canary not yet started. Need to begin deployment pipeline."
            ),
            criteria_scores={
                "Deployment progress >= 100%": -0.5,
                "All stages passed": -0.3,
                "Zero rollbacks": 1.0,
            },
        ),
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="run_canary",
                            description="Start canary deployment for api-gateway",
                            parameters={"service": "api-gateway"},
                        ),
                    ],
                    reasoning="Begin deployment by running canary on the primary service",
                    expected_outcome="Canary deployment validates code on 5% of traffic",
                    confidence=0.85,
                ),
            ],
            selected_index=0,
            selection_reasoning="Canary is the safest first step in the deployment pipeline",
        ),
        # --- Pass 2 (post-negotiation 1): promote to staging ---
        EvaluationOutput(
            score=0.40,
            confidence=0.70,
            reasoning=(
                "Canary passed. Progress at 15%. "
                "Need to advance to staging. Shared direction says to proceed with "
                "security monitoring alongside."
            ),
            criteria_scores={
                "Deployment progress >= 100%": -0.1,
                "All stages passed": 0.0,
                "Zero rollbacks": 1.0,
            },
        ),
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="promote_stage",
                            description="Promote to staging environment",
                            parameters={"target_stage": "staging"},
                        ),
                    ],
                    reasoning=(
                        "Canary passed successfully. Promote to staging for broader "
                        "validation. Security team is monitoring in parallel."
                    ),
                    expected_outcome="Staging validation with 25% traffic",
                    confidence=0.75,
                ),
            ],
            selected_index=0,
            selection_reasoning="Staging promotion follows successful canary",
        ),
        # --- Pass 3 (post-negotiation 2): promote to production ---
        EvaluationOutput(
            score=0.70,
            confidence=0.80,
            reasoning=(
                "Staging passed. CVE has been patched. "
                "Progress at 40%. Shared direction from negotiation: "
                "proceed to production now that security is clear."
            ),
            criteria_scores={
                "Deployment progress >= 100%": 0.4,
                "All stages passed": 0.6,
                "Zero rollbacks": 1.0,
            },
        ),
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="promote_stage",
                            description="Promote to production — final deployment step",
                            parameters={"target_stage": "production"},
                        ),
                    ],
                    reasoning=(
                        "All blockers cleared: canary passed, staging passed, "
                        "CVE-2024-31337 patched, SRE confirms healthy infrastructure."
                    ),
                    expected_outcome="Production deployment completes release v2.5",
                    confidence=0.90,
                ),
            ],
            selected_index=0,
            selection_reasoning="All gates passed — safe to promote to production",
        ),
    ])


def _build_security_mock():
    """Mock LLM for the Security agent.

    3 passes x 1 step each.  Pass 2 discovers CVE -> score <= -0.3 -> revision.
    Sequence: Eval + Plan, Eval(bad) + Revision + Plan, Eval + Plan.
    """
    from synthetic_teleology.services.llm_evaluation import EvaluationOutput
    from synthetic_teleology.services.llm_planning import (
        ActionProposal,
        PlanHypothesis,
        PlanningOutput,
    )
    from synthetic_teleology.services.llm_revision import RevisionOutput
    from synthetic_teleology.testing import MockStructuredChatModel

    return MockStructuredChatModel(structured_responses=[
        # --- Pass 1: initial scan (clean) ---
        EvaluationOutput(
            score=0.50,
            confidence=0.65,
            reasoning=(
                "No vulnerabilities detected in initial scan. "
                "Security posture is baseline-acceptable for canary stage."
            ),
            criteria_scores={
                "Zero critical CVEs": 1.0,
                "All patches applied": 1.0,
                "Security scan passed": 0.5,
            },
        ),
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="scan_vulnerabilities",
                            description="Run comprehensive vulnerability scan on all services",
                            parameters={"target": "all-services"},
                        ),
                    ],
                    reasoning="Proactive scanning during canary catches issues early",
                    expected_outcome="Clean scan or early CVE detection",
                    confidence=0.80,
                ),
            ],
            selected_index=0,
            selection_reasoning="Continuous scanning is best practice during deployment",
        ),
        # --- Pass 2: CVE discovered (score triggers revision) ---
        EvaluationOutput(
            score=-0.60,
            confidence=0.90,
            reasoning=(
                "CRITICAL: CVE-2024-31337 discovered in api-gateway. "
                "Remote code execution vulnerability. "
                "Deployment must be halted until patched."
            ),
            criteria_scores={
                "Zero critical CVEs": -1.0,
                "All patches applied": -0.5,
                "Security scan passed": -0.8,
            },
        ),
        # Revision triggered because score (-0.60) <= -0.3
        RevisionOutput(
            should_revise=True,
            reasoning=(
                "Critical CVE discovered. Goal must be revised to prioritize "
                "immediate patching before any further deployment progress."
            ),
            new_description=(
                "URGENT: Patch CVE-2024-31337 in api-gateway before allowing "
                "production promotion. Then verify security posture."
            ),
            new_criteria=[
                "CVE-2024-31337 patched",
                "Post-patch vulnerability scan clean",
                "No new critical CVEs introduced",
            ],
        ),
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="apply_patch",
                            description="Apply emergency patch for CVE-2024-31337",
                            parameters={"cve_id": "CVE-2024-31337"},
                        ),
                    ],
                    reasoning=(
                        "Critical CVE requires immediate patching. "
                        "Patch is available — apply before any further deployment."
                    ),
                    expected_outcome="CVE patched, security posture restored",
                    confidence=0.95,
                ),
            ],
            selected_index=0,
            selection_reasoning="Immediate patching is the only acceptable action",
        ),
        # --- Pass 3: verification scan (post-patch) ---
        EvaluationOutput(
            score=0.75,
            confidence=0.85,
            reasoning=(
                "CVE-2024-31337 has been patched. "
                "No open critical vulnerabilities. "
                "Security posture is clear for production promotion."
            ),
            criteria_scores={
                "CVE-2024-31337 patched": 1.0,
                "Post-patch vulnerability scan clean": 0.8,
                "No new critical CVEs introduced": 1.0,
            },
        ),
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="scan_vulnerabilities",
                            description="Final verification scan before production",
                            parameters={"target": "all-services"},
                        ),
                    ],
                    reasoning="Post-patch verification confirms security is clear",
                    expected_outcome="Clean final scan, clearance for production",
                    confidence=0.90,
                ),
            ],
            selected_index=0,
            selection_reasoning="Final scan provides confidence for production go-ahead",
        ),
    ])


def _build_sre_mock():
    """Mock LLM for the SRE agent.

    3 passes x 1 step each.  No revisions (scores stay above -0.3).
    """
    from synthetic_teleology.services.llm_evaluation import EvaluationOutput
    from synthetic_teleology.services.llm_planning import (
        ActionProposal,
        PlanHypothesis,
        PlanningOutput,
    )
    from synthetic_teleology.testing import MockStructuredChatModel

    return MockStructuredChatModel(structured_responses=[
        # --- Pass 1: baseline monitoring ---
        EvaluationOutput(
            score=0.60,
            confidence=0.75,
            reasoning=(
                "All 3 services healthy. Latency and error rates within SLA. "
                "No violations. Infrastructure is stable for deployment."
            ),
            criteria_scores={
                "Uptime >= 99.9%": 0.8,
                "Latency < 200ms": 0.9,
                "Error rate < 0.1%": 0.8,
            },
        ),
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="check_sla",
                            description="Verify SLA metrics for all services",
                            parameters={"service": "api-gateway"},
                        ),
                    ],
                    reasoning="Baseline SLA check before deployment changes traffic patterns",
                    expected_outcome="Confirmed SLA compliance across all services",
                    confidence=0.85,
                ),
            ],
            selected_index=0,
            selection_reasoning="Monitoring is the priority during canary phase",
        ),
        # --- Pass 2: scale up during CVE response ---
        EvaluationOutput(
            score=0.30,
            confidence=0.70,
            reasoning=(
                "api-gateway degraded due to CVE impact. "
                "Latency elevated. Need to scale capacity to maintain SLA "
                "while security team patches the vulnerability."
            ),
            criteria_scores={
                "Uptime >= 99.9%": 0.4,
                "Latency < 200ms": 0.2,
                "Error rate < 0.1%": 0.3,
            },
        ),
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="scale_capacity",
                            description="Scale api-gateway to absorb degraded performance",
                            parameters={"service": "api-gateway", "replicas": 5},
                        ),
                    ],
                    reasoning=(
                        "Scale up api-gateway to compensate for CVE-induced "
                        "latency spike while patch is applied"
                    ),
                    expected_outcome="Latency reduced, SLA maintained during incident",
                    confidence=0.80,
                ),
            ],
            selected_index=0,
            selection_reasoning="Scaling is fastest mitigation for latency spikes",
        ),
        # --- Pass 3: verify stability ---
        EvaluationOutput(
            score=0.80,
            confidence=0.85,
            reasoning=(
                "All services healthy post-patch. Latency normalized after "
                "scaling. No SLA violations in the last monitoring window. "
                "Infrastructure ready for production promotion."
            ),
            criteria_scores={
                "Uptime >= 99.9%": 0.95,
                "Latency < 200ms": 0.9,
                "Error rate < 0.1%": 0.85,
            },
        ),
        PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name="check_sla",
                            description="Final SLA verification before production",
                            parameters={"service": "api-gateway"},
                        ),
                    ],
                    reasoning="Confirm stability post-incident before production go-ahead",
                    expected_outcome="All services within SLA, safe to promote",
                    confidence=0.90,
                ),
            ],
            selected_index=0,
            selection_reasoning="Final stability check gives green light for production",
        ),
    ])


def _build_negotiation_mock():
    """Mock LLM for the negotiation protocol.

    2 negotiation rounds x (3 proposals + 1 critique + 1 synthesis) = 10 responses.

    The ``LLMNegotiator`` uses raw ``model.invoke(prompt)`` (not
    ``with_structured_output``), so responses are JSON strings returned
    as ``AIMessage`` content.
    """
    from synthetic_teleology.testing import MockStructuredChatModel

    return MockStructuredChatModel(structured_responses=[
        # ===== Negotiation Round 1 (after pass 1) =====

        # Propose: release-agent
        json.dumps({
            "direction": (
                "Proceed with canary and stage promotion. "
                "All signals green. Aim for staging by next round."
            ),
            "reasoning": "Canary passed, no blockers. Move fast to staging.",
            "priority_dimensions": ["deployment velocity", "stage completion"],
            "confidence": 0.80,
        }),
        # Propose: security-agent
        json.dumps({
            "direction": (
                "Continue deployment with continuous vulnerability scanning. "
                "Security baseline is clean. Monitor for emerging threats."
            ),
            "reasoning": "Initial scan clean, but continuous monitoring is essential.",
            "priority_dimensions": ["vulnerability detection", "continuous scanning"],
            "confidence": 0.75,
        }),
        # Propose: sre-agent
        json.dumps({
            "direction": (
                "Proceed with deployment. Infrastructure is stable. "
                "Keep monitoring SLA during stage transitions."
            ),
            "reasoning": "All services within SLA. Scaling plan ready if needed.",
            "priority_dimensions": ["uptime", "latency monitoring"],
            "confidence": 0.80,
        }),
        # Critique
        json.dumps({
            "agreements": [
                "All agents agree deployment should proceed to staging",
                "Continuous monitoring is a shared priority",
            ],
            "disagreements": [
                "Release prioritizes speed while security wants thorough scanning",
            ],
            "synthesis_hints": [
                "Run security scans in parallel with stage promotion",
                "Set automated gates that can halt promotion if scan fails",
            ],
        }),
        # Synthesize
        json.dumps({
            "shared_direction": (
                "Proceed to staging with security scanning in parallel. "
                "SRE monitors infrastructure. Automated gate blocks production "
                "if any critical CVE is found."
            ),
            "revised_criteria": [
                "Canary passed",
                "Security scan concurrent with staging",
                "SLA within bounds during transition",
            ],
            "confidence": 0.82,
            "reasoning": (
                "Balanced approach: deployment velocity with security guardrails. "
                "All three agents aligned on proceed-with-monitoring strategy."
            ),
        }),

        # ===== Negotiation Round 2 (after pass 2 — CVE discovered) =====

        # Propose: release-agent
        json.dumps({
            "direction": (
                "Staging passed but CVE blocks production. "
                "Hold production promotion until security clears. "
                "Prepare rollback plan as contingency."
            ),
            "reasoning": (
                "Cannot risk production with a critical CVE. "
                "Timeline delayed but safety is non-negotiable."
            ),
            "priority_dimensions": ["safe deployment", "patch verification"],
            "confidence": 0.70,
        }),
        # Propose: security-agent
        json.dumps({
            "direction": (
                "HALT all deployment progression. CVE-2024-31337 is critical. "
                "Patch has been applied. Need post-patch verification scan "
                "before any further advancement."
            ),
            "reasoning": (
                "Critical RCE vulnerability was found and patched. "
                "Must verify patch effectiveness before production."
            ),
            "priority_dimensions": ["patch verification", "zero critical CVEs"],
            "confidence": 0.90,
        }),
        # Propose: sre-agent
        json.dumps({
            "direction": (
                "Scale up affected services to maintain SLA during patching. "
                "api-gateway latency spiked. Additional capacity absorbs "
                "degradation while security team verifies patch."
            ),
            "reasoning": (
                "SLA at risk from CVE-induced degradation. "
                "Scaling provides buffer during incident response."
            ),
            "priority_dimensions": ["SLA compliance", "capacity headroom"],
            "confidence": 0.80,
        }),
        # Critique
        json.dumps({
            "agreements": [
                "All agents agree production promotion must wait for CVE resolution",
                "Patch has been applied — verification is the next step",
                "SRE scaling helps maintain stability during incident",
            ],
            "disagreements": [
                "Security wants full halt while release wants "
                "rapid promotion post-patch",
            ],
            "synthesis_hints": [
                "Run verification scan immediately",
                "SRE scales capacity to maintain SLA",
                "Release prepares promotion pipeline to execute "
                "once security gives all-clear",
            ],
        }),
        # Synthesize
        json.dumps({
            "shared_direction": (
                "Pause production promotion until CVE-2024-31337 patch is verified. "
                "SRE scales api-gateway capacity to maintain SLA. "
                "Once security scan confirms clean posture, release proceeds "
                "to production immediately."
            ),
            "revised_criteria": [
                "CVE-2024-31337 patch verified",
                "Post-patch scan clean",
                "SLA maintained during incident",
                "Production promotion after security clearance",
            ],
            "confidence": 0.88,
            "reasoning": (
                "Critical CVE required immediate halt. Patch applied, "
                "now in verification phase. Coordinated response: "
                "security verifies, SRE stabilizes, release prepares."
            ),
        }),
    ])


# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------


def build_deployment_coordinator(max_rounds: int = 2):
    """Build the multi-agent deployment coordinator graph.

    Returns ``(app, initial_input, deployment_state)`` tuple.

    Parameters
    ----------
    max_rounds:
        Number of negotiation rounds (agents run max_rounds + 1 times).
    """
    deployment_state = DeploymentState()

    # --- Resolve models ---
    real_model = _get_model()
    if real_model is not None:
        release_model = real_model
        security_model = real_model
        sre_model = real_model
        negotiation_model = real_model
        mode = "real LLM"
    else:
        release_model = _build_release_mock()
        security_model = _build_security_mock()
        sre_model = _build_sre_mock()
        negotiation_model = _build_negotiation_mock()
        mode = "simulated (no API key)"

    # --- Per-agent environments ---
    release_perceive, release_transition = _make_release_env(deployment_state)
    security_perceive, security_transition = _make_security_env(deployment_state)
    sre_perceive, sre_transition = _make_sre_env(deployment_state)

    # --- Agent configs ---
    configs = [
        AgentConfig(
            agent_id="release-agent",
            goal="Ship release v2.5 on schedule with zero rollbacks",
            model=release_model,
            criteria=[
                "Deployment progress >= 100%",
                "All stages passed",
                "Zero rollbacks",
            ],
            perceive_fn=release_perceive,
            transition_fn=release_transition,
            max_steps_per_round=1,
        ),
        AgentConfig(
            agent_id="security-agent",
            goal="Ensure zero critical vulnerabilities before production",
            model=security_model,
            criteria=[
                "Zero critical CVEs",
                "All patches applied",
                "Security scan passed",
            ],
            perceive_fn=security_perceive,
            transition_fn=security_transition,
            max_steps_per_round=1,
        ),
        AgentConfig(
            agent_id="sre-agent",
            goal="Maintain 99.9% uptime during deployment",
            model=sre_model,
            criteria=[
                "Uptime >= 99.9%",
                "Latency < 200ms",
                "Error rate < 0.1%",
            ],
            perceive_fn=sre_perceive,
            transition_fn=sre_transition,
            max_steps_per_round=1,
        ),
    ]

    # --- Build graph ---
    app = build_multi_agent_graph(
        agent_configs=configs,
        negotiation_model=negotiation_model,
    )

    initial_input = {
        "max_rounds": max_rounds,
        "agent_results": {},
        "events": [],
        "reasoning_trace": [],
    }

    return app, initial_input, deployment_state, mode
