"""
Agentic Titan - Agent Framework

This module provides the core agent abstractions and archetypes for building
multi-agent systems that can self-organize into different topologies.
"""

from agents.framework.base_agent import BaseAgent, AgentContext, AgentState
from agents.framework.errors import (
    TitanError,
    AgentError,
    HiveMindError,
    TopologyError,
    LLMAdapterError,
)
from agents.personas import Persona, ORCHESTRATOR, RESEARCHER, CODER, REVIEWER, say

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentContext",
    "AgentState",
    # Errors
    "TitanError",
    "AgentError",
    "HiveMindError",
    "TopologyError",
    "LLMAdapterError",
    # Personas
    "Persona",
    "ORCHESTRATOR",
    "RESEARCHER",
    "CODER",
    "REVIEWER",
    "say",
]
