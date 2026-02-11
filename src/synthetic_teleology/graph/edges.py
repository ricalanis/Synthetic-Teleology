"""Conditional edge functions for the teleological LangGraph.

These functions determine routing between nodes based on the current state.
"""

from typing import Any, Literal


def should_continue(state: dict[str, Any]) -> Literal["perceive", "__end__"]:
    """After reflect, decide whether to loop back or terminate.

    Returns ``"perceive"`` to continue the loop or ``"__end__"`` to stop.
    """
    if state.get("stop_reason"):
        return "__end__"
    return "perceive"


def should_revise(state: dict[str, Any]) -> Literal["revise", "check_constraints"]:
    """After evaluate, decide whether to attempt goal revision.

    Revise when the evaluation signal indicates poor performance
    (score <= -0.3). Good scores indicate the agent is on track and
    skip revision.
    """
    signal = state.get("eval_signal")
    if signal is None:
        return "check_constraints"

    score = signal.score
    if score <= -0.3:
        return "revise"
    return "check_constraints"
