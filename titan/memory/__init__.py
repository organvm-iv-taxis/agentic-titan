"""
Titan Memory - Pluggable memory backend system.

Provides a unified interface for different memory backends:
- ChromaDB (vector store)
- Memori (SQL-native)
- Redis (key-value)

Reference: vendor/tools/memori/ SQL-native patterns
"""

from titan.memory.backend import (
    MemoryBackend,
    MemoryEntry,
    MemoryConfig,
    BackendType,
)
from titan.memory.chromadb_backend import ChromaDBBackend
from titan.memory.memori_backend import MemoriBackend

__all__ = [
    "MemoryBackend",
    "MemoryEntry",
    "MemoryConfig",
    "BackendType",
    "ChromaDBBackend",
    "MemoriBackend",
]
