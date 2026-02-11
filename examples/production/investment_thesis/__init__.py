"""Investment Thesis Builder — Goal-directed investment analysis agent.

Builds comprehensive investment thesis using simulated financial research tools.
Demonstrates goal revision when negative news (lawsuit) is discovered mid-analysis.
"""

from .agent import build_thesis_agent

__all__ = ["build_thesis_agent"]
