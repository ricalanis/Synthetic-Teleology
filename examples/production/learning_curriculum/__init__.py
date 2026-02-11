"""Adaptive Learning Curriculum — Goal-directed curriculum optimization agent.

Adapts teaching curriculum based on simulated learner performance.
Demonstrates goal revision when quiz failures reveal knowledge gaps.
"""

from .agent import build_curriculum_agent

__all__ = ["build_curriculum_agent"]
