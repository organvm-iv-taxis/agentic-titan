"""
Shared fixtures for auth tests.
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
def readonly_user():
    """Create a readonly user for testing."""
    return User(
        id=uuid4(),
        username="readonly",
        email="readonly@example.com",
        hashed_password="hashed",  # allow-secret
        role=UserRole.READONLY,
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def service_user():
    """Create a service account user for testing."""
    return User(
        id=uuid4(),
        username="service",
        hashed_password="hashed",  # allow-secret
        role=UserRole.SERVICE,
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
    storage.hash_password.return_value = "hashed"  # allow-secret
    storage.verify_password.return_value = True  # allow-secret
    storage.create_api_key.return_value = True
    storage.get_api_key.return_value = None
    storage.get_api_key_by_hash.return_value = None
    storage.get_api_keys_for_user.return_value = []
    storage.update_api_key_last_used.return_value = True
    storage.deactivate_api_key.return_value = True
    storage.delete_api_key.return_value = True
    return storage
