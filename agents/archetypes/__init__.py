"""
Agent Archetypes - Pre-built agent types for common tasks.

Available archetypes:
- Orchestrator: Coordinates multi-agent workflows
- Researcher: Gathers and analyzes information
- Coder: Writes and tests code
- Reviewer: Reviews work for quality
"""

from agents.archetypes.orchestrator import OrchestratorAgent
from agents.archetypes.researcher import ResearcherAgent
from agents.archetypes.coder import CoderAgent
from agents.archetypes.reviewer import ReviewerAgent

__all__ = [
    "OrchestratorAgent",
    "ResearcherAgent",
    "CoderAgent",
    "ReviewerAgent",
]
