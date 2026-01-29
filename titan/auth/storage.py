"""
Titan Authentication - Storage Backend

PostgreSQL storage for users and API keys.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

logger = logging.getLogger("titan.auth.storage")


def _get_passlib():
    """Lazy import of passlib."""
    try:
        from passlib.context import CryptContext
        return CryptContext(schemes=["bcrypt"], deprecated="auto")
    except ImportError:
        raise ImportError(
            "passlib is required for password hashing. "
            "Install with: pip install 'agentic-titan[auth]'"
        )


class AuthStorage:
    """
    PostgreSQL storage for authentication data.

    Manages users and API keys with CRUD operations.
    """

    def __init__(self, postgres_client: Any) -> None:
        """
        Initialize auth storage.

        Args:
            postgres_client: PostgresClient instance
        """
        self._db = postgres_client
        self._pwd_context = _get_passlib()

    async def initialize(self) -> None:
        """Initialize database tables."""
        from titan.auth.models import USERS_TABLE_SQL, API_KEYS_TABLE_SQL

        if not self._db.is_connected:
            await self._db.connect()

        if self._db.is_connected:
            await self._db.execute(USERS_TABLE_SQL)
            await self._db.execute(API_KEYS_TABLE_SQL)
            logger.info("Auth tables initialized")

    # =========================================================================
    # User Operations
    # =========================================================================

    def hash_password(self, password: str) -> str:  # allow-secret
        """Hash a password for storage."""
        return self._pwd_context.hash(password)  # allow-secret

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:  # allow-secret
        """Verify a password against a hash."""
        return self._pwd_context.verify(plain_password, hashed_password)  # allow-secret

    async def create_user(
        self,
        user_id: UUID,
        username: str,
        hashed_password: str,  # allow-secret
        email: str | None = None,
        role: str = "user",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Create a new user.

        Returns:
            True if created successfully
        """
        if not self._db.is_connected:
            return False

        try:
            await self._db.execute(
                """
                INSERT INTO users (id, username, email, hashed_password, role, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                user_id,
                username,
                email,
                hashed_password,  # allow-secret
                role,
                json.dumps(metadata or {}),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            return False

    async def get_user(self, user_id: UUID | str) -> dict[str, Any] | None:
        """Get a user by ID."""
        if not self._db.is_connected:
            return None

        target = UUID(user_id) if isinstance(user_id, str) else user_id
        row = await self._db.fetchrow(
            "SELECT * FROM users WHERE id = $1",
            target,
        )

        if not row:
            return None

        result = dict(row)
        if "metadata" in result and isinstance(result["metadata"], str):
            result["metadata"] = json.loads(result["metadata"])
        return result

    async def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        """Get a user by username."""
        if not self._db.is_connected:
            return None

        row = await self._db.fetchrow(
            "SELECT * FROM users WHERE username = $1",
            username,
        )

        if not row:
            return None

        result = dict(row)
        if "metadata" in result and isinstance(result["metadata"], str):
            result["metadata"] = json.loads(result["metadata"])
        return result

    async def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        """Get a user by email."""
        if not self._db.is_connected:
            return None

        row = await self._db.fetchrow(
            "SELECT * FROM users WHERE email = $1",
            email,
        )

        if not row:
            return None

        result = dict(row)
        if "metadata" in result and isinstance(result["metadata"], str):
            result["metadata"] = json.loads(result["metadata"])
        return result

    async def update_user(
        self,
        user_id: UUID | str,
        updates: dict[str, Any],
    ) -> bool:
        """
        Update a user.

        Args:
            user_id: User ID
            updates: Dictionary of field updates

        Returns:
            True if updated successfully
        """
        if not self._db.is_connected:
            return False

        if not updates:
            return True

        target = UUID(user_id) if isinstance(user_id, str) else user_id

        # Always update updated_at
        updates["updated_at"] = datetime.now(timezone.utc)

        # Build dynamic update query
        set_clauses = []
        params: list[Any] = []
        param_idx = 1

        for field_name, value in updates.items():
            if field_name == "metadata":
                value = json.dumps(value)
            set_clauses.append(f"{field_name} = ${param_idx}")
            params.append(value)
            param_idx += 1

        params.append(target)
        query = f"""
            UPDATE users
            SET {", ".join(set_clauses)}
            WHERE id = ${param_idx}
        """

        try:
            await self._db.execute(query, *params)
            return True
        except Exception as e:
            logger.error(f"Failed to update user {user_id}: {e}")
            return False

    async def update_last_login(self, user_id: UUID | str) -> bool:
        """Update user's last login timestamp."""
        return await self.update_user(user_id, {"last_login": datetime.now(timezone.utc)})

    async def delete_user(self, user_id: UUID | str) -> bool:
        """Delete a user and their API keys (cascade)."""
        if not self._db.is_connected:
            return False

        target = UUID(user_id) if isinstance(user_id, str) else user_id
        try:
            await self._db.execute("DELETE FROM users WHERE id = $1", target)
            return True
        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            return False

    async def list_users(
        self,
        role: str | None = None,
        is_active: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List users with optional filtering."""
        if not self._db.is_connected:
            return []

        conditions = []
        params: list[Any] = []
        param_idx = 1

        if role:
            conditions.append(f"role = ${param_idx}")
            params.append(role)
            param_idx += 1

        if is_active is not None:
            conditions.append(f"is_active = ${param_idx}")
            params.append(is_active)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        query = f"""
            SELECT id, username, email, role, is_active, created_at, last_login
            FROM users
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        rows = await self._db.fetch(query, *params)
        return [dict(row) for row in rows]

    async def count_users(self, role: str | None = None) -> int:
        """Count users with optional role filter."""
        if not self._db.is_connected:
            return 0

        if role:
            count = await self._db.fetchval(
                "SELECT COUNT(*) FROM users WHERE role = $1",
                role,
            )
        else:
            count = await self._db.fetchval("SELECT COUNT(*) FROM users")

        return count or 0

    # =========================================================================
    # API Key Operations
    # =========================================================================

    async def create_api_key(
        self,
        key_id: UUID,
        key_hash: str,
        key_prefix: str,
        name: str,
        user_id: UUID,
        scopes: list[str] | None = None,
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Create a new API key.

        Returns:
            True if created successfully
        """
        if not self._db.is_connected:
            return False

        try:
            await self._db.execute(
                """
                INSERT INTO api_keys
                (id, key_hash, key_prefix, name, user_id, scopes, expires_at, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                key_id,
                key_hash,
                key_prefix,
                name,
                user_id,
                json.dumps(scopes or []),
                expires_at,
                json.dumps(metadata or {}),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            return False

    async def get_api_key(self, key_id: UUID | str) -> dict[str, Any] | None:
        """Get an API key by ID."""
        if not self._db.is_connected:
            return None

        target = UUID(key_id) if isinstance(key_id, str) else key_id
        row = await self._db.fetchrow(
            "SELECT * FROM api_keys WHERE id = $1",
            target,
        )

        if not row:
            return None

        result = dict(row)
        if "scopes" in result and isinstance(result["scopes"], str):
            result["scopes"] = json.loads(result["scopes"])
        if "metadata" in result and isinstance(result["metadata"], str):
            result["metadata"] = json.loads(result["metadata"])
        return result

    async def get_api_key_by_hash(self, key_hash: str) -> dict[str, Any] | None:
        """Get an API key by its hash."""
        if not self._db.is_connected:
            return None

        row = await self._db.fetchrow(
            "SELECT * FROM api_keys WHERE key_hash = $1",
            key_hash,
        )

        if not row:
            return None

        result = dict(row)
        if "scopes" in result and isinstance(result["scopes"], str):
            result["scopes"] = json.loads(result["scopes"])
        if "metadata" in result and isinstance(result["metadata"], str):
            result["metadata"] = json.loads(result["metadata"])
        return result

    async def get_api_keys_for_user(
        self,
        user_id: UUID | str,
        include_inactive: bool = False,
    ) -> list[dict[str, Any]]:
        """Get all API keys for a user."""
        if not self._db.is_connected:
            return []

        target = UUID(user_id) if isinstance(user_id, str) else user_id

        if include_inactive:
            query = "SELECT * FROM api_keys WHERE user_id = $1 ORDER BY created_at DESC"
            rows = await self._db.fetch(query, target)
        else:
            query = """
                SELECT * FROM api_keys
                WHERE user_id = $1 AND is_active = TRUE
                ORDER BY created_at DESC
            """
            rows = await self._db.fetch(query, target)

        results = []
        for row in rows:
            result = dict(row)
            if "scopes" in result and isinstance(result["scopes"], str):
                result["scopes"] = json.loads(result["scopes"])
            if "metadata" in result and isinstance(result["metadata"], str):
                result["metadata"] = json.loads(result["metadata"])
            results.append(result)

        return results

    async def update_api_key_last_used(self, key_id: UUID | str) -> bool:
        """Update API key's last used timestamp."""
        if not self._db.is_connected:
            return False

        target = UUID(key_id) if isinstance(key_id, str) else key_id
        try:
            await self._db.execute(
                "UPDATE api_keys SET last_used_at = $1 WHERE id = $2",
                datetime.now(timezone.utc),
                target,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update API key last used: {e}")
            return False

    async def deactivate_api_key(self, key_id: UUID | str) -> bool:
        """Deactivate an API key."""
        if not self._db.is_connected:
            return False

        target = UUID(key_id) if isinstance(key_id, str) else key_id
        try:
            await self._db.execute(
                "UPDATE api_keys SET is_active = FALSE WHERE id = $1",
                target,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to deactivate API key: {e}")
            return False

    async def delete_api_key(self, key_id: UUID | str) -> bool:
        """Delete an API key."""
        if not self._db.is_connected:
            return False

        target = UUID(key_id) if isinstance(key_id, str) else key_id
        try:
            await self._db.execute("DELETE FROM api_keys WHERE id = $1", target)
            return True
        except Exception as e:
            logger.error(f"Failed to delete API key: {e}")
            return False


# Singleton instance
_auth_storage: AuthStorage | None = None


async def get_auth_storage() -> AuthStorage:
    """Get or create the auth storage instance."""
    global _auth_storage
    if _auth_storage is None:
        from titan.persistence.postgres import get_postgres_client

        client = get_postgres_client()
        _auth_storage = AuthStorage(client)
        await _auth_storage.initialize()
    return _auth_storage
