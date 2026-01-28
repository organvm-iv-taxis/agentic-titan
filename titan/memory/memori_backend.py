"""
Memori Backend - SQL-native memory implementation.

Provides memory storage using SQLite/PostgreSQL:
- No expensive vector DB dependency
- Full-text search via FTS5
- Entity extraction support
- Multi-tenant isolation
- Cost optimization (80-90% cheaper than vector DBs)

Reference: vendor/tools/memori/ SQL-native patterns
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
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

logger = logging.getLogger("titan.memory.memori")


# ============================================================================
# SQL Schema
# ============================================================================

SQLITE_SCHEMA = """
-- Long-term memory storage
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    category TEXT DEFAULT 'conversational',
    importance REAL DEFAULT 0.5,
    novelty_score REAL DEFAULT 0.5,
    relevance_score REAL DEFAULT 0.5,
    tags TEXT DEFAULT '[]',
    entities TEXT DEFAULT '[]',
    keywords TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    related_memories TEXT DEFAULT '[]',
    supersedes TEXT DEFAULT '[]',
    user_id TEXT DEFAULT 'default',
    session_id TEXT DEFAULT 'default',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP,
    expires_at TIMESTAMP
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);

-- Full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    id,
    content,
    entities,
    keywords,
    content=memories,
    content_rowid=rowid
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memory_fts(rowid, id, content, entities, keywords)
    VALUES (new.rowid, new.id, new.content, new.entities, new.keywords);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, id, content, entities, keywords)
    VALUES ('delete', old.rowid, old.id, old.content, old.entities, old.keywords);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, id, content, entities, keywords)
    VALUES ('delete', old.rowid, old.id, old.content, old.entities, old.keywords);
    INSERT INTO memory_fts(rowid, id, content, entities, keywords)
    VALUES (new.rowid, new.id, new.content, new.entities, new.keywords);
END;

-- Working memory (short-term)
CREATE TABLE IF NOT EXISTS working_memory (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    user_id TEXT DEFAULT 'default',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_working_user ON working_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_working_expires ON working_memory(expires_at);
"""


# ============================================================================
# Memori Backend Implementation
# ============================================================================


class MemoriBackend(MemoryBackend):
    """
    SQL-native memory backend using SQLite.

    Provides:
    - Full-text search via FTS5
    - Efficient queries with proper indexing
    - Multi-tenant isolation
    - Lower cost than vector databases
    """

    backend_type = BackendType.MEMORI

    def __init__(self, config: MemoryConfig) -> None:
        super().__init__(config)
        self._conn: Any = None
        self._db_path: Path | None = None

    async def initialize(self) -> None:
        """Initialize SQLite connection."""
        try:
            import aiosqlite

            self._db_path = Path(self.config.database_path)

            # Create database and schema
            self._conn = await aiosqlite.connect(str(self._db_path))

            # Enable WAL mode for better concurrency
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA synchronous=NORMAL")

            # Create schema
            await self._conn.executescript(SQLITE_SCHEMA)
            await self._conn.commit()

            logger.info(f"Memori backend initialized: {self._db_path}")
            self._initialized = True

        except ImportError:
            raise ImportError(
                "aiosqlite is required for Memori backend. "
                "Install with: pip install 'agentic-titan[memori]'"
            )

    async def shutdown(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
        self._initialized = False
        logger.info("Memori backend shutdown")

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
        """Store content in SQLite."""
        if not self._conn:
            raise RuntimeError("Backend not initialized")

        memory_id = f"mem-{uuid.uuid4().hex[:12]}"
        now = datetime.now()

        await self._conn.execute(
            """
            INSERT INTO memories (
                id, content, agent_id, category, importance,
                tags, entities, keywords, metadata,
                user_id, session_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                content,
                agent_id,
                category.value,
                importance,
                json.dumps(tags or []),
                json.dumps(entities or []),
                json.dumps(keywords or []),
                json.dumps(metadata or {}),
                self.config.user_id,
                self.config.session_id,
                now.isoformat(),
            ),
        )
        await self._conn.commit()

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
        """Search for relevant memories using FTS5."""
        if not self._conn:
            raise RuntimeError("Backend not initialized")

        # Build query
        # Use FTS5 for text matching with BM25 ranking
        sql = """
            SELECT
                m.id, m.content, m.agent_id, m.category, m.importance,
                m.tags, m.entities, m.keywords, m.metadata,
                m.user_id, m.session_id, m.created_at,
                bm25(memory_fts) as score
            FROM memories m
            JOIN memory_fts ON m.id = memory_fts.id
            WHERE memory_fts MATCH ?
        """
        params: list[Any] = [query]

        # Add filters
        if category:
            sql += " AND m.category = ?"
            params.append(category.value)

        if agent_id:
            sql += " AND m.agent_id = ?"
            params.append(agent_id)

        # User isolation
        sql += " AND m.user_id = ?"
        params.append(self.config.user_id)

        # Order by relevance
        sql += " ORDER BY score LIMIT ?"
        params.append(k)

        try:
            cursor = await self._conn.execute(sql, params)
            rows = await cursor.fetchall()

            results = []
            for row in rows:
                entry = MemoryEntry(
                    id=row[0],
                    content=row[1],
                    agent_id=row[2],
                    category=MemoryCategory(row[3]),
                    importance=row[4],
                    tags=json.loads(row[5]),
                    entities=json.loads(row[6]),
                    keywords=json.loads(row[7]),
                    metadata=json.loads(row[8]),
                    user_id=row[9],
                    session_id=row[10],
                )

                # BM25 scores are negative (closer to 0 is better)
                # Convert to positive score
                score = abs(row[12]) if row[12] else 0

                if score >= min_score:
                    results.append(SearchResult(entry=entry, score=score))

            # Filter by tags if specified
            if tags:
                results = [
                    r for r in results
                    if any(t in r.entry.tags for t in tags)
                ]

            return results

        except Exception as e:
            logger.warning(f"FTS search failed, falling back to LIKE: {e}")

            # Fallback to simple LIKE search
            sql = """
                SELECT
                    id, content, agent_id, category, importance,
                    tags, entities, keywords, metadata,
                    user_id, session_id, created_at
                FROM memories
                WHERE content LIKE ? AND user_id = ?
            """
            params = [f"%{query}%", self.config.user_id]

            if category:
                sql += " AND category = ?"
                params.append(category.value)
            if agent_id:
                sql += " AND agent_id = ?"
                params.append(agent_id)

            sql += " ORDER BY importance DESC LIMIT ?"
            params.append(k)

            cursor = await self._conn.execute(sql, params)
            rows = await cursor.fetchall()

            results = []
            for row in rows:
                entry = MemoryEntry(
                    id=row[0],
                    content=row[1],
                    agent_id=row[2],
                    category=MemoryCategory(row[3]),
                    importance=row[4],
                    tags=json.loads(row[5]),
                    entities=json.loads(row[6]),
                    keywords=json.loads(row[7]),
                    metadata=json.loads(row[8]),
                    user_id=row[9],
                    session_id=row[10],
                )
                results.append(SearchResult(entry=entry, score=entry.importance))

            return results

    async def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a specific memory by ID."""
        if not self._conn:
            raise RuntimeError("Backend not initialized")

        cursor = await self._conn.execute(
            """
            SELECT
                id, content, agent_id, category, importance,
                tags, entities, keywords, metadata,
                user_id, session_id, created_at
            FROM memories
            WHERE id = ? AND user_id = ?
            """,
            (memory_id, self.config.user_id),
        )
        row = await cursor.fetchone()

        if row:
            # Update accessed_at
            await self._conn.execute(
                "UPDATE memories SET accessed_at = ? WHERE id = ?",
                (datetime.now().isoformat(), memory_id),
            )
            await self._conn.commit()

            return MemoryEntry(
                id=row[0],
                content=row[1],
                agent_id=row[2],
                category=MemoryCategory(row[3]),
                importance=row[4],
                tags=json.loads(row[5]),
                entities=json.loads(row[6]),
                keywords=json.loads(row[7]),
                metadata=json.loads(row[8]),
                user_id=row[9],
                session_id=row[10],
            )

        return None

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        if not self._conn:
            raise RuntimeError("Backend not initialized")

        cursor = await self._conn.execute(
            "DELETE FROM memories WHERE id = ? AND user_id = ?",
            (memory_id, self.config.user_id),
        )
        await self._conn.commit()

        return cursor.rowcount > 0

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
        if not self._conn:
            raise RuntimeError("Backend not initialized")

        updates = []
        params: list[Any] = []

        if importance is not None:
            updates.append("importance = ?")
            params.append(importance)

        if category is not None:
            updates.append("category = ?")
            params.append(category.value)

        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))

        if metadata is not None:
            # Merge with existing metadata
            entry = await self.get(memory_id)
            if entry:
                entry.metadata.update(metadata)
                updates.append("metadata = ?")
                params.append(json.dumps(entry.metadata))

        if not updates:
            return False

        params.extend([memory_id, self.config.user_id])

        cursor = await self._conn.execute(
            f"UPDATE memories SET {', '.join(updates)} WHERE id = ? AND user_id = ?",
            params,
        )
        await self._conn.commit()

        return cursor.rowcount > 0

    async def set_working(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Set a value in working memory."""
        if not self._conn:
            raise RuntimeError("Backend not initialized")

        expires_at = None
        if ttl:
            expires_at = (datetime.now() + timedelta(seconds=ttl)).isoformat()

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO working_memory (key, value, user_id, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (key, json.dumps(value), self.config.user_id, expires_at),
        )
        await self._conn.commit()

    async def get_working(self, key: str, default: Any = None) -> Any:
        """Get a value from working memory."""
        if not self._conn:
            raise RuntimeError("Backend not initialized")

        cursor = await self._conn.execute(
            """
            SELECT value, expires_at FROM working_memory
            WHERE key = ? AND user_id = ?
            """,
            (key, self.config.user_id),
        )
        row = await cursor.fetchone()

        if row:
            # Check expiration
            if row[1]:
                expires = datetime.fromisoformat(row[1])
                if datetime.now() > expires:
                    await self.delete_working(key)
                    return default
            return json.loads(row[0])

        return default

    async def delete_working(self, key: str) -> bool:
        """Delete a key from working memory."""
        if not self._conn:
            raise RuntimeError("Backend not initialized")

        cursor = await self._conn.execute(
            "DELETE FROM working_memory WHERE key = ? AND user_id = ?",
            (key, self.config.user_id),
        )
        await self._conn.commit()

        return cursor.rowcount > 0

    async def count(
        self,
        category: MemoryCategory | None = None,
        agent_id: str | None = None,
    ) -> int:
        """Count memories."""
        if not self._conn:
            raise RuntimeError("Backend not initialized")

        sql = "SELECT COUNT(*) FROM memories WHERE user_id = ?"
        params: list[Any] = [self.config.user_id]

        if category:
            sql += " AND category = ?"
            params.append(category.value)
        if agent_id:
            sql += " AND agent_id = ?"
            params.append(agent_id)

        cursor = await self._conn.execute(sql, params)
        row = await cursor.fetchone()

        return row[0] if row else 0

    async def clear(
        self,
        category: MemoryCategory | None = None,
        agent_id: str | None = None,
    ) -> int:
        """Clear memories."""
        if not self._conn:
            raise RuntimeError("Backend not initialized")

        sql = "DELETE FROM memories WHERE user_id = ?"
        params: list[Any] = [self.config.user_id]

        if category:
            sql += " AND category = ?"
            params.append(category.value)
        if agent_id:
            sql += " AND agent_id = ?"
            params.append(agent_id)

        cursor = await self._conn.execute(sql, params)
        await self._conn.commit()

        return cursor.rowcount

    async def health_check(self) -> dict[str, Any]:
        """Check backend health."""
        base = await super().health_check()
        base.update({
            "database_path": str(self._db_path) if self._db_path else None,
            "connected": self._conn is not None,
            "memory_count": await self.count() if self._conn else 0,
        })
        return base


# Register the backend
register_backend(BackendType.MEMORI, MemoriBackend)
