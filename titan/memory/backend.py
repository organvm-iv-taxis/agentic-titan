"""
Memory Backend Protocol - Abstract interface for memory backends.

Defines the contract that all memory backends must implement:
- Store/retrieve memories with semantic search
- Working memory (key-value)
- Entity extraction support
- Multi-tenant isolation

Reference: vendor/tools/memori/ architecture patterns
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger("titan.memory")


# ============================================================================
# Enums and Types
# ============================================================================


class BackendType(str, Enum):
    """Supported memory backend types."""

    CHROMADB = "chromadb"      # Vector store (default)
    MEMORI = "memori"          # SQL-native
    REDIS = "redis"            # Key-value
    IN_MEMORY = "in_memory"    # Development/testing


class MemoryCategory(str, Enum):
    """Categories for memory classification."""

    ESSENTIAL = "essential"        # Core facts and preferences
    CONTEXTUAL = "contextual"      # Project/work context
    CONVERSATIONAL = "conversational"  # Regular discussions
    REFERENCE = "reference"        # Technical references
    PERSONAL = "personal"          # Life events, relationships
    CONSCIOUS = "conscious"        # Auto-promote to short-term


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class MemoryConfig:
    """Configuration for memory backend."""

    backend_type: BackendType = BackendType.CHROMADB

    # Connection settings
    connection_string: str = ""
    database_path: str = "titan_memory.db"

    # ChromaDB settings
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    collection_name: str = "titan_memories"

    # Multi-tenant settings
    user_id: str = "default"
    assistant_id: str | None = None
    session_id: str = "default"

    # Memory settings
    max_short_term_items: int = 100
    memory_ttl_seconds: int = 3600 * 24 * 7  # 7 days default
    embedding_model: str = "hash"  # "hash", "sentence-transformers", etc.


@dataclass
class MemoryEntry:
    """
    A memory entry stored in the backend.

    Inspired by Memori's memory model with enhanced classification.
    """

    id: str
    content: str
    agent_id: str

    # Classification
    category: MemoryCategory = MemoryCategory.CONVERSATIONAL
    importance: float = 0.5
    novelty_score: float = 0.5
    relevance_score: float = 0.5

    # Metadata
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Entity extraction
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    # Relationships
    related_memories: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime | None = None
    expires_at: datetime | None = None

    # Multi-tenant
    user_id: str = "default"
    session_id: str = "default"

    # Embedding
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "agent_id": self.agent_id,
            "category": self.category.value,
            "importance": self.importance,
            "tags": self.tags,
            "entities": self.entities,
            "keywords": self.keywords,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            agent_id=data.get("agent_id", "unknown"),
            category=MemoryCategory(data.get("category", "conversational")),
            importance=data.get("importance", 0.5),
            tags=data.get("tags", []),
            entities=data.get("entities", []),
            keywords=data.get("keywords", []),
            metadata=data.get("metadata", {}),
            user_id=data.get("user_id", "default"),
            session_id=data.get("session_id", "default"),
        )


@dataclass
class SearchResult:
    """Result from memory search."""

    entry: MemoryEntry
    score: float
    distance: float | None = None


# ============================================================================
# Memory Backend Protocol
# ============================================================================


class MemoryBackend(ABC):
    """
    Abstract base class for memory backends.

    All memory implementations must provide these methods.
    """

    backend_type: BackendType

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the backend connection.

        Should:
        - Connect to database/service
        - Create tables/collections if needed
        - Validate configuration
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown and cleanup connections.
        """
        pass

    # =========================================================================
    # Long-term Memory Operations
    # =========================================================================

    @abstractmethod
    async def store(
        self,
        content: str,
        agent_id: str,
        *,
        importance: float = 0.5,
        category: MemoryCategory = MemoryCategory.CONVERSATIONAL,
        tags: list[str] | None = None,
        entities: list[str] | None = None,
        keywords: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store content in long-term memory.

        Args:
            content: Content to store
            agent_id: ID of the storing agent
            importance: Importance score (0.0 to 1.0)
            category: Memory category
            tags: Tags for categorization
            entities: Extracted entities
            keywords: Key terms
            metadata: Additional metadata

        Returns:
            Memory entry ID
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        k: int = 10,
        min_score: float = 0.0,
        category: MemoryCategory | None = None,
        tags: list[str] | None = None,
        agent_id: str | None = None,
    ) -> list[SearchResult]:
        """
        Search for relevant memories.

        Args:
            query: Search query (semantic)
            k: Number of results
            min_score: Minimum relevance score
            category: Filter by category
            tags: Filter by tags
            agent_id: Filter by agent

        Returns:
            List of search results
        """
        pass

    @abstractmethod
    async def get(self, memory_id: str) -> MemoryEntry | None:
        """
        Get a specific memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            MemoryEntry or None if not found
        """
        pass

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory ID to delete

        Returns:
            True if deleted
        """
        pass

    @abstractmethod
    async def update(
        self,
        memory_id: str,
        *,
        importance: float | None = None,
        category: MemoryCategory | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Update a memory entry.

        Args:
            memory_id: Memory ID to update
            importance: New importance score
            category: New category
            tags: New tags
            metadata: Additional metadata to merge

        Returns:
            True if updated
        """
        pass

    # =========================================================================
    # Working Memory (Key-Value)
    # =========================================================================

    @abstractmethod
    async def set_working(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """
        Set a value in working memory.

        Args:
            key: Key
            value: Value (will be serialized)
            ttl: Time-to-live in seconds
        """
        pass

    @abstractmethod
    async def get_working(self, key: str, default: Any = None) -> Any:
        """
        Get a value from working memory.

        Args:
            key: Key
            default: Default if not found

        Returns:
            Value or default
        """
        pass

    @abstractmethod
    async def delete_working(self, key: str) -> bool:
        """Delete a key from working memory."""
        pass

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @abstractmethod
    async def count(
        self,
        category: MemoryCategory | None = None,
        agent_id: str | None = None,
    ) -> int:
        """
        Count memories.

        Args:
            category: Filter by category
            agent_id: Filter by agent

        Returns:
            Count of matching memories
        """
        pass

    @abstractmethod
    async def clear(
        self,
        category: MemoryCategory | None = None,
        agent_id: str | None = None,
    ) -> int:
        """
        Clear memories.

        Args:
            category: Clear only this category
            agent_id: Clear only this agent's memories

        Returns:
            Number of memories cleared
        """
        pass

    async def health_check(self) -> dict[str, Any]:
        """
        Check backend health.

        Returns:
            Health status dict
        """
        return {
            "backend_type": self.backend_type.value,
            "initialized": self._initialized,
            "user_id": self.config.user_id,
        }


# ============================================================================
# Backend Factory
# ============================================================================

_backend_registry: dict[BackendType, type[MemoryBackend]] = {}


def register_backend(
    backend_type: BackendType,
    backend_class: type[MemoryBackend],
) -> None:
    """Register a memory backend implementation."""
    _backend_registry[backend_type] = backend_class


def create_backend(config: MemoryConfig) -> MemoryBackend:
    """
    Create a memory backend from configuration.

    Args:
        config: Memory configuration

    Returns:
        Configured MemoryBackend instance
    """
    backend_class = _backend_registry.get(config.backend_type)
    if not backend_class:
        raise ValueError(f"Unknown backend type: {config.backend_type}")
    return backend_class(config)


def list_backends() -> list[str]:
    """List available backend types."""
    return [bt.value for bt in _backend_registry.keys()]
