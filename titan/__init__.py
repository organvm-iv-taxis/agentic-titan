"""
Agentic Titan - Polymorphic Agent Swarm Architecture

A model-agnostic, polyglot, self-organizing multi-agent system that:
- Shapeshifts between organizational topologies based on task requirements
- Deploys across local, container, and serverless runtimes dynamically
- Shares a collective intelligence layer (Hive Mind)
- Scales from 2 to 100+ agents
"""

from titan.spec import AgentSpec, SpecRegistry

__version__ = "0.1.0"

__all__ = [
    "AgentSpec",
    "SpecRegistry",
    "__version__",
]
