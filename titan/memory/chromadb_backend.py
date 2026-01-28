"""
ChromaDB Backend - Vector store memory implementation.

Provides semantic search using ChromaDB embeddings.
This is the default backend for Titan.

Uses the existing HiveMind ChromaDB integration pattern.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import Any

from titan.memory.backend import (
    BackendType,
    MemoryBackend,
    MemoryCategory,
    MemoryConfig,
    MemoryEntry,
    SearchResult,
    register_backend,
)

logger = logging.getLogger("titan.memory.chromadb")


# ============================================================================
# Hash Embedding (Fallback)
# ============================================================================


def hash_embedding(text: str, dim: int = 384) -> list[float]:
    """
    Fast hash-based embedding for text.

    Uses character n-grams hashed to a fixed dimension.
    """
    ngrams: list[str] = []
    text_lower = text.lower()
    for n in [2, 3, 4]:
        for i in range(len(text_lower) - n + 1):
            ngrams.append(text_lower[i : i + n])

    embedding = [0.0] * dim
    for ngram in ngrams:
        h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1 if (h // dim) % 2 == 0 else -1
        embedding[idx] += sign * 1.0

    # Normalize
    magnitude = sum(x * x for x in embedding) ** 0.5
    if magnitude > 0:
        embedding = [x / magnitude for x in embedding]

    return embedding


# ============================================================================
# ChromaDB Backend Implementation
# ============================================================================


class ChromaDBBackend(MemoryBackend):
    """
    ChromaDB vector store backend.

    Provides semantic search through embeddings.
    Falls back to in-memory storage if ChromaDB is unavailable.
    """

    backend_type = BackendType.CHROMADB

    def __init__(self, config: MemoryConfig) -> None:
        super().__init__(config)
        self._client: Any = None
        self._collection: Any = None

        # Fallback in-memory storage
        self._memory_store: dict[str, MemoryEntry] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._working_memory: dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize ChromaDB connection."""
        try:
            import chromadb

            self._client = chromadb.HttpClient(
                host=self.config.chroma_host,
                port=self.config.chroma_port,
            )
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"Connected to ChromaDB at "
                f"{self.config.chroma_host}:{self.config.chroma_port}"
            )
        except Exception as e:
            logger.warning(f"ChromaDB unavailable, using in-memory fallback: {e}")
            self._client = None
            self._collection = None

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown ChromaDB connection."""
        self._client = None
        self._collection = None
        self._initialized = False
        logger.info("ChromaDB backend shutdown")

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
        """Store content in ChromaDB."""
        memory_id = f"mem-{uuid.uuid4().hex[:12]}"
        embedding = hash_embedding(content)
        now = datetime.now()

        entry = MemoryEntry(
            id=memory_id,
            content=content,
            agent_id=agent_id,
            category=category,
            importance=importance,
            tags=tags or [],
            entities=entities or [],
            keywords=keywords or [],
            metadata=metadata or {},
            embedding=embedding,
            created_at=now,
            user_id=self.config.user_id,
            session_id=self.config.session_id,
        )

        if self._collection is not None:
            self._collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[
                    {
                        "agent_id": agent_id,
                        "category": category.value,
                        "importance": importance,
                        "tags": json.dumps(tags or []),
                        "entities": json.dumps(entities or []),
                        "keywords": json.dumps(keywords or []),
                        "metadata": json.dumps(metadata or {}),
                        "created_at": now.isoformat(),
                        "user_id": self.config.user_id,
                        "session_id": self.config.session_id,
                    }
                ],
            )
        else:
            # In-memory fallback
            self._memory_store[memory_id] = entry
            self._embeddings[memory_id] = embedding

        logger.debug(f"Stored memory: {memory_id}")
        return memory_id

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
        """Search for relevant memories."""
        query_embedding = hash_embedding(query)

        if self._collection is not None:
            # Build where filter
            where_filter: dict[str, Any] = {}
            if agent_id:
                where_filter["agent_id"] = agent_id
            if category:
                where_filter["category"] = category.value

            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter if where_filter else None,
            )

            search_results = []
            docs = results.get("documents", [[]])[0]
            ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            for i, doc in enumerate(docs):
                meta = metadatas[i] if i < len(metadatas) else {}
                distance = distances[i] if i < len(distances) else 0

                # Convert distance to score (cosine: lower distance = higher score)
                score = 1.0 - distance if distance <= 1.0 else 0.0

                if score >= min_score:
                    entry = MemoryEntry(
                        id=ids[i] if i < len(ids) else f"mem-{i}",
                        content=doc,
                        agent_id=meta.get("agent_id", "unknown"),
                        category=MemoryCategory(meta.get("category", "conversational")),
                        importance=meta.get("importance", 0.5),
                        tags=json.loads(meta.get("tags", "[]")),
                        entities=json.loads(meta.get("entities", "[]")),
                        keywords=json.loads(meta.get("keywords", "[]")),
                        user_id=meta.get("user_id", "default"),
                        session_id=meta.get("session_id", "default"),
                    )
                    search_results.append(SearchResult(
                        entry=entry,
                        score=score,
                        distance=distance,
                    ))

            return search_results

        # In-memory fallback
        results = []
        for mem_id, entry in self._memory_store.items():
            # Apply filters
            if agent_id and entry.agent_id != agent_id:
                continue
            if category and entry.category != category:
                continue
            if tags and not any(t in entry.tags for t in tags):
                continue

            # Compute similarity
            if mem_id in self._embeddings:
                emb = self._embeddings[mem_id]
                similarity = sum(a * b for a, b in zip(query_embedding, emb))
            else:
                similarity = 0

            if similarity >= min_score:
                results.append((similarity, entry))

        # Sort by similarity
        results.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResult(entry=entry, score=score)
            for score, entry in results[:k]
        ]

    async def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a specific memory by ID."""
        if self._collection is not None:
            try:
                result = self._collection.get(ids=[memory_id])
                if result["documents"]:
                    meta = result["metadatas"][0] if result["metadatas"] else {}
                    return MemoryEntry(
                        id=memory_id,
                        content=result["documents"][0],
                        agent_id=meta.get("agent_id", "unknown"),
                        category=MemoryCategory(meta.get("category", "conversational")),
                        importance=meta.get("importance", 0.5),
                        tags=json.loads(meta.get("tags", "[]")),
                    )
            except Exception:
                pass
            return None

        return self._memory_store.get(memory_id)

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        if self._collection is not None:
            try:
                self._collection.delete(ids=[memory_id])
                return True
            except Exception:
                return False

        if memory_id in self._memory_store:
            del self._memory_store[memory_id]
            self._embeddings.pop(memory_id, None)
            return True
        return False

    async def update(
        self,
        memory_id: str,
        *,
        importance: float | None = None,
        category: MemoryCategory | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update a memory entry."""
        entry = await self.get(memory_id)
        if not entry:
            return False

        # Update fields
        if importance is not None:
            entry.importance = importance
        if category is not None:
            entry.category = category
        if tags is not None:
            entry.tags = tags
        if metadata is not None:
            entry.metadata.update(metadata)

        # Re-store (ChromaDB doesn't support partial updates well)
        if self._collection is not None:
            self._collection.update(
                ids=[memory_id],
                metadatas=[
                    {
                        "agent_id": entry.agent_id,
                        "category": entry.category.value,
                        "importance": entry.importance,
                        "tags": json.dumps(entry.tags),
                        "entities": json.dumps(entry.entities),
                        "keywords": json.dumps(entry.keywords),
                        "metadata": json.dumps(entry.metadata),
                        "user_id": entry.user_id,
                        "session_id": entry.session_id,
                    }
                ],
            )
        else:
            self._memory_store[memory_id] = entry

        return True

    async def set_working(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Set a value in working memory."""
        # Working memory is always in-memory for ChromaDB backend
        self._working_memory[key] = {
            "value": value,
            "expires": (datetime.now().timestamp() + ttl) if ttl else None,
        }

    async def get_working(self, key: str, default: Any = None) -> Any:
        """Get a value from working memory."""
        entry = self._working_memory.get(key)
        if entry:
            if entry["expires"] and datetime.now().timestamp() > entry["expires"]:
                del self._working_memory[key]
                return default
            return entry["value"]
        return default

    async def delete_working(self, key: str) -> bool:
        """Delete a key from working memory."""
        if key in self._working_memory:
            del self._working_memory[key]
            return True
        return False

    async def count(
        self,
        category: MemoryCategory | None = None,
        agent_id: str | None = None,
    ) -> int:
        """Count memories."""
        if self._collection is not None:
            # ChromaDB doesn't have efficient count with filters
            return self._collection.count()

        count = 0
        for entry in self._memory_store.values():
            if category and entry.category != category:
                continue
            if agent_id and entry.agent_id != agent_id:
                continue
            count += 1
        return count

    async def clear(
        self,
        category: MemoryCategory | None = None,
        agent_id: str | None = None,
    ) -> int:
        """Clear memories."""
        if self._collection is not None and not category and not agent_id:
            count = self._collection.count()
            # Delete all - need to recreate collection
            self._client.delete_collection(self.config.collection_name)
            self._collection = self._client.create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return count

        # Selective delete
        to_delete = []
        for mem_id, entry in self._memory_store.items():
            if category and entry.category != category:
                continue
            if agent_id and entry.agent_id != agent_id:
                continue
            to_delete.append(mem_id)

        for mem_id in to_delete:
            await self.delete(mem_id)

        return len(to_delete)

    async def health_check(self) -> dict[str, Any]:
        """Check backend health."""
        base = await super().health_check()
        base.update({
            "chromadb_connected": self._collection is not None,
            "memory_count": len(self._memory_store) if not self._collection else await self.count(),
            "working_memory_count": len(self._working_memory),
        })
        return base


# Register the backend
register_backend(BackendType.CHROMADB, ChromaDBBackend)
