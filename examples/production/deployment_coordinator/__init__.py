"""Deployment Coordinator — Multi-agent deployment orchestration.

Three agents (Release, Security, SRE) coordinate a software deployment.
Demonstrates multi-agent negotiation when a CVE is discovered mid-deployment.
"""

from .agent import build_deployment_coordinator

__all__ = ["build_deployment_coordinator"]
