"""
Shared fixtures for API tests.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from titan.auth.models import User, UserRole


@pytest.fixture
def admin_user():
    """Create an admin user for testing."""
    return User(
        id=uuid4(),
        username="admin",
        email="admin@example.com",
        hashed_password="hashed",  # allow-secret
        role=UserRole.ADMIN,
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def regular_user():
    """Create a regular user for testing."""
    return User(
        id=uuid4(),
        username="user",
        email="user@example.com",
        hashed_password="hashed",  # allow-secret
        role=UserRole.USER,
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_auth_storage():
    """Create a mock auth storage."""
    storage = AsyncMock()
    storage.get_user.return_value = None
    storage.get_user_by_username.return_value = None
    storage.get_user_by_email.return_value = None
    storage.create_user.return_value = True
    storage.update_user.return_value = True
    storage.delete_user.return_value = True
    storage.list_users.return_value = []
    storage.count_users.return_value = 0
    storage.hash_password = MagicMock(return_value="hashed")  # allow-secret
    storage.verify_password = MagicMock(return_value=True)  # allow-secret
    return storage


@pytest.fixture
def mock_postgres_client():
    """Create a mock PostgreSQL client."""
    client = MagicMock()
    client.is_connected = True
    client.health_check = AsyncMock(return_value={"healthy": True, "status": "healthy"})
    client.get_audit_events = AsyncMock(return_value=[])
    return client
