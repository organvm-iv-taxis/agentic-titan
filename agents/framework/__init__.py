"""
Agent Framework - Core abstractions for building agents.

Ported and extended from:
- metasystem-core/agent_utils (BaseAgent lifecycle)
- agent--claude-smith (session management patterns)
"""

from agents.framework.base_agent import BaseAgent, AgentContext, AgentState, AgentResult
from agents.framework.errors import (
    TitanError,
    AgentError,
    HiveMindError,
    TopologyError,
    LLMAdapterError,
    CircuitBreakerError,
)
from agents.framework.resilience import CircuitBreaker, CircuitState

__all__ = [
    "BaseAgent",
    "AgentContext",
    "AgentState",
    "AgentResult",
    "TitanError",
    "AgentError",
    "HiveMindError",
    "TopologyError",
    "LLMAdapterError",
    "CircuitBreakerError",
    "CircuitBreaker",
    "CircuitState",
]
