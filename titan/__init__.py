"""
Agentic Titan - Polymorphic Agent Swarm Architecture

A model-agnostic, polyglot, self-organizing multi-agent system that:
- Shapeshifts between organizational topologies based on task requirements
- Deploys across local, container, and serverless runtimes dynamically
- Shares a collective intelligence layer (Hive Mind)
- Scales from 2 to 100+ agents

Sub-packages:
- titan.core: Project context loading (TITAN.md)
- titan.memory: Pluggable memory backends (ChromaDB, Memori)
- titan.persistence: State checkpointing
- titan.observability: Langfuse tracing
- titan.learning: Local style learning
- titan.tools: Advanced tools (Image Gen, M365)
"""

from titan.spec import AgentSpec, SpecRegistry

__version__ = "0.1.0"

__all__ = [
    "AgentSpec",
    "SpecRegistry",
    "__version__",
]
