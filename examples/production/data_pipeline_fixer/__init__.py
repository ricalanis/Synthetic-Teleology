"""Data Pipeline Fixer — Autonomous pipeline monitoring and repair agent.

Maintains data pipeline health using simulated diagnostic and repair tools.
Demonstrates goal revision when schema drift is detected, plus evolving constraints.
"""

from .agent import build_pipeline_agent

__all__ = ["build_pipeline_agent"]
