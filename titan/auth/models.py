"""
Titan Authentication - Data Models

Defines the core authentication data models including users, API keys, and tokens.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, EmailStr


class UserRole(str, Enum):
    """User role enumeration for role-based access control."""

    ADMIN = "admin"      # Full system access
    USER = "user"        # Standard user access
    SERVICE = "service"  # Service account for automation
    READONLY = "readonly"  # Read-only access


class User(BaseModel):
    """
    User account model.

    Represents a user in the system with authentication credentials
    and role-based permissions.
    """

    id: UUID = Field(default_factory=uuid4)
    username: str = Field(..., min_length=3, max_length=100)
    email: str | None = Field(default=None)
    hashed_password: str = Field(...)
    role: UserRole = Field(default=UserRole.USER)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = Field(default=None)
    last_login: datetime | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self, include_password: bool = False) -> dict[str, Any]:
        """Convert to dictionary, optionally excluding password."""
        data = {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "metadata": self.metadata,
        }
        if include_password:
            data["hashed_password"] = self.hashed_password
        return data

    def to_public_dict(self) -> dict[str, Any]:
        """Convert to dictionary for public display (no sensitive fields)."""
        return {
            "id": str(self.id),
            "username": self.username,
            "role": self.role.value,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> User:
        """Create User from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            username=data["username"],
            email=data.get("email"),
            hashed_password=data["hashed_password"],
            role=UserRole(data["role"]) if isinstance(data["role"], str) else data["role"],
            is_active=data.get("is_active", True),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data["created_at"], str)
                else data["created_at"]
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data.get("updated_at") and isinstance(data["updated_at"], str)
                else data.get("updated_at")
            ),
            last_login=(
                datetime.fromisoformat(data["last_login"])
                if data.get("last_login") and isinstance(data["last_login"], str)
                else data.get("last_login")
            ),
            metadata=data.get("metadata", {}),
        )


class UserCreate(BaseModel):
    """Request model for creating a new user."""

    username: str = Field(..., min_length=3, max_length=100)
    email: str | None = Field(default=None)
    password: str = Field(..., min_length=8)  # allow-secret
    role: UserRole = Field(default=UserRole.USER)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserUpdate(BaseModel):
    """Request model for updating a user."""

    email: str | None = None
    password: str | None = None  # allow-secret
    role: UserRole | None = None
    is_active: bool | None = None
    metadata: dict[str, Any] | None = None


class APIKey(BaseModel):
    """
    API key model for programmatic access.

    API keys provide a way for services and scripts to authenticate
    without using JWT tokens.
    """

    id: UUID = Field(default_factory=uuid4)
    key_hash: str = Field(...)  # SHA256 hash of the key
    key_prefix: str = Field(...)  # First 8 characters for identification
    name: str = Field(..., min_length=1, max_length=100)
    user_id: UUID = Field(...)
    scopes: list[str] = Field(default_factory=list)
    expires_at: datetime | None = Field(default=None)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: datetime | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if the API key has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "key_hash": self.key_hash,
            "key_prefix": self.key_prefix,
            "name": self.name,
            "user_id": str(self.user_id),
            "scopes": self.scopes,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "metadata": self.metadata,
        }

    def to_public_dict(self) -> dict[str, Any]:
        """Convert to dictionary for public display (no hash)."""
        return {
            "id": str(self.id),
            "key_prefix": self.key_prefix,
            "name": self.name,
            "scopes": self.scopes,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> APIKey:
        """Create APIKey from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            key_hash=data["key_hash"],
            key_prefix=data["key_prefix"],
            name=data["name"],
            user_id=UUID(data["user_id"]) if isinstance(data["user_id"], str) else data["user_id"],
            scopes=data.get("scopes", []),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at") and isinstance(data["expires_at"], str)
                else data.get("expires_at")
            ),
            is_active=data.get("is_active", True),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data["created_at"], str)
                else data["created_at"]
            ),
            last_used_at=(
                datetime.fromisoformat(data["last_used_at"])
                if data.get("last_used_at") and isinstance(data["last_used_at"], str)
                else data.get("last_used_at")
            ),
            metadata=data.get("metadata", {}),
        )


class APIKeyCreate(BaseModel):
    """Request model for creating a new API key."""

    name: str = Field(..., min_length=1, max_length=100)
    scopes: list[str] = Field(default_factory=list)
    expires_in_days: int | None = Field(default=None, ge=1, le=365)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenPair(BaseModel):
    """JWT token pair (access + refresh)."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Access token expiry in seconds")


class TokenData(BaseModel):
    """Data extracted from a decoded JWT token."""

    user_id: str
    username: str
    role: UserRole
    token_type: str = "access"  # "access" or "refresh"
    exp: datetime
    iat: datetime


# PostgreSQL table definition
USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ,
    last_login TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
"""

API_KEYS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY,
    key_hash VARCHAR(64) NOT NULL,
    key_prefix VARCHAR(10) NOT NULL,
    name VARCHAR(100) NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    scopes JSONB DEFAULT '[]',
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_prefix ON api_keys(key_prefix);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
"""
