"""
Tests for titan.api.admin_routes module.

Tests admin API endpoints.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from titan.auth.models import User, UserRole


@pytest.fixture
def admin_user():
    """Create an admin user for testing."""
    return User(
        id=uuid4(),
        username="admin",
        hashed_password="hashed",  # allow-secret
        role=UserRole.ADMIN,
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )


class TestDetailedHealth:
    """Tests for detailed health endpoint."""

    @pytest.mark.asyncio
    async def test_detailed_health_all_healthy(self, admin_user):
        """Test detailed health when all components are healthy."""
        from titan.api.admin_routes import detailed_health

        with patch.dict("sys.modules", {"redis": MagicMock()}):
            import sys
            mock_redis = sys.modules["redis"]
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_redis.from_url.return_value = mock_client

            with patch("titan.persistence.postgres.get_postgres_client") as mock_pg:
                mock_pg_client = MagicMock()
                mock_pg_client.health_check = AsyncMock(return_value={"healthy": True, "status": "healthy"})
                mock_pg.return_value = mock_pg_client

                result = await detailed_health()

                assert result.status in ("healthy", "degraded")

    @pytest.mark.asyncio
    async def test_detailed_health_redis_down(self, admin_user):
        """Test detailed health when Redis is down."""
        from titan.api.admin_routes import detailed_health

        with patch.dict("sys.modules", {"redis": MagicMock()}):
            import sys
            mock_redis = sys.modules["redis"]
            mock_redis.from_url.side_effect = Exception("Connection refused")

            with patch("titan.persistence.postgres.get_postgres_client") as mock_pg:
                mock_pg_client = MagicMock()
                mock_pg_client.health_check = AsyncMock(return_value={"healthy": True, "status": "healthy"})
                mock_pg.return_value = mock_pg_client

                result = await detailed_health()

                assert result.components["redis"]["status"] == "unhealthy"


class TestMetricsSummary:
    """Tests for metrics summary endpoint."""

    @pytest.mark.asyncio
    async def test_metrics_summary(self, admin_user):
        """Test metrics summary endpoint."""
        from titan.api.admin_routes import metrics_summary

        with patch("titan.auth.storage.get_auth_storage") as mock_storage:
            storage = AsyncMock()
            storage.count_users.return_value = 10
            mock_storage.return_value = storage

            with patch("titan.batch.orchestrator.get_batch_orchestrator") as mock_orch:
                orch = MagicMock()
                orch.list_batches.return_value = []
                mock_orch.return_value = orch

                with patch("titan.workflows.inquiry_engine.get_inquiry_engine") as mock_engine:
                    engine = MagicMock()
                    engine.list_sessions.return_value = []
                    mock_engine.return_value = engine

                    result = await metrics_summary()

                    assert result.total_users == 10
                    assert isinstance(result.total_batches, int)


class TestUserManagement:
    """Tests for user management endpoints."""

    @pytest.mark.asyncio
    async def test_list_users(self, admin_user):
        """Test listing users."""
        from titan.api.admin_routes import list_users

        mock_users = [
            {
                "id": uuid4(),
                "username": "user1",
                "email": "user1@example.com",
                "role": "user",
                "is_active": True,
                "created_at": datetime.now(timezone.utc),
                "last_login": None,
            },
            {
                "id": uuid4(),
                "username": "user2",
                "email": None,
                "role": "admin",
                "is_active": True,
                "created_at": datetime.now(timezone.utc),
                "last_login": datetime.now(timezone.utc),
            },
        ]

        with patch("titan.auth.storage.get_auth_storage") as mock_storage:
            storage = AsyncMock()
            storage.list_users.return_value = mock_users
            mock_storage.return_value = storage

            result = await list_users()

            assert len(result) == 2
            assert result[0].username == "user1"
            assert result[1].role == "admin"

    @pytest.mark.asyncio
    async def test_create_user_success(self, admin_user):
        """Test creating a new user."""
        from titan.api.admin_routes import create_user, UserCreateRequest

        request = UserCreateRequest(
            username="newuser",
            email="new@example.com",
            password="secure_password",  # allow-secret
            role=UserRole.USER,
        )

        with patch("titan.auth.storage.get_auth_storage") as mock_storage:
            storage = AsyncMock()
            storage.get_user_by_username.return_value = None
            storage.get_user_by_email.return_value = None
            storage.hash_password.return_value = "hashed"  # allow-secret
            storage.create_user.return_value = True
            mock_storage.return_value = storage

            result = await create_user(request)

            assert result.username == "newuser"
            assert result.email == "new@example.com"
            assert result.role == "user"
            storage.create_user.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_user_duplicate_username(self, admin_user):
        """Test creating user with duplicate username."""
        from titan.api.admin_routes import create_user, UserCreateRequest
        from fastapi import HTTPException

        request = UserCreateRequest(
            username="existing",
            password="secure_password",  # allow-secret
        )

        with patch("titan.auth.storage.get_auth_storage") as mock_storage:
            storage = AsyncMock()
            storage.get_user_by_username.return_value = {"id": uuid4()}  # Exists
            mock_storage.return_value = storage

            with pytest.raises(HTTPException) as exc_info:
                await create_user(request)

            assert exc_info.value.status_code == 409
            assert "already exists" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_update_user_success(self, admin_user):
        """Test updating a user."""
        from titan.api.admin_routes import update_user, UserUpdateRequest

        user_id = str(uuid4())
        request = UserUpdateRequest(
            email="updated@example.com",
            role=UserRole.ADMIN,
        )

        user_data = {
            "id": user_id,
            "username": "testuser",
            "email": "updated@example.com",
            "role": "admin",
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
            "last_login": None,
        }

        with patch("titan.auth.storage.get_auth_storage") as mock_storage:
            storage = AsyncMock()
            storage.get_user.return_value = user_data
            storage.update_user.return_value = True
            mock_storage.return_value = storage

            result = await update_user(user_id, request)

            assert result.email == "updated@example.com"
            assert result.role == "admin"

    @pytest.mark.asyncio
    async def test_update_user_not_found(self, admin_user):
        """Test updating non-existent user."""
        from titan.api.admin_routes import update_user, UserUpdateRequest
        from fastapi import HTTPException

        request = UserUpdateRequest(email="new@example.com")

        with patch("titan.auth.storage.get_auth_storage") as mock_storage:
            storage = AsyncMock()
            storage.get_user.return_value = None
            mock_storage.return_value = storage

            with pytest.raises(HTTPException) as exc_info:
                await update_user("nonexistent", request)

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_user_success(self, admin_user):
        """Test deleting a user."""
        from titan.api.admin_routes import delete_user

        user_id = str(uuid4())

        with patch("titan.auth.storage.get_auth_storage") as mock_storage:
            storage = AsyncMock()
            storage.get_user.return_value = {"id": user_id, "username": "todelete"}
            storage.delete_user.return_value = True
            mock_storage.return_value = storage

            result = await delete_user(user_id)

            assert "deleted" in result["message"].lower()
            storage.delete_user.assert_called_once_with(user_id)


class TestConfiguration:
    """Tests for configuration endpoints."""

    @pytest.mark.asyncio
    async def test_get_config(self, admin_user):
        """Test getting all configuration."""
        from titan.api.admin_routes import get_config

        result = await get_config()

        assert len(result) > 0
        assert any(c.key == "rate_limit_default" for c in result)
        assert any(c.key == "batch_max_concurrent" for c in result)

    @pytest.mark.asyncio
    async def test_update_config_success(self, admin_user):
        """Test updating configuration."""
        from titan.api.admin_routes import update_config, ConfigUpdateRequest

        request = ConfigUpdateRequest(value="200/minute")

        result = await update_config("rate_limit_default", request)

        assert result.key == "rate_limit_default"
        assert result.value == "200/minute"

    @pytest.mark.asyncio
    async def test_update_config_not_found(self, admin_user):
        """Test updating non-existent config."""
        from titan.api.admin_routes import update_config, ConfigUpdateRequest
        from fastapi import HTTPException

        request = ConfigUpdateRequest(value="test")

        with pytest.raises(HTTPException) as exc_info:
            await update_config("nonexistent_key", request)

        assert exc_info.value.status_code == 404


class TestBatchManagement:
    """Tests for batch management endpoints."""

    @pytest.mark.asyncio
    async def test_get_stalled_batches(self, admin_user):
        """Test getting stalled batches."""
        from titan.api.admin_routes import get_stalled_batches

        with patch("titan.batch.orchestrator.get_batch_orchestrator") as mock_orch:
            orch = MagicMock()
            orch.get_stalled_batches = AsyncMock(return_value=[])
            mock_orch.return_value = orch

            result = await get_stalled_batches()

            assert result == []

    @pytest.mark.asyncio
    async def test_recover_batch_success(self, admin_user):
        """Test recovering a stalled batch."""
        from titan.api.admin_routes import recover_batch, RecoveryRequest

        request = RecoveryRequest(strategy="retry")

        with patch("titan.batch.orchestrator.get_batch_orchestrator") as mock_orch:
            orch = MagicMock()
            orch.recover_stalled_batch = AsyncMock(return_value={"recovered": 2, "failed": 0})
            mock_orch.return_value = orch

            result = await recover_batch("batch-123", request)

            assert result["recovered_sessions"] == 2
            assert result["strategy"] == "retry"

    @pytest.mark.asyncio
    async def test_recover_batch_invalid_strategy(self, admin_user):
        """Test recovering batch with invalid strategy."""
        from titan.api.admin_routes import recover_batch, RecoveryRequest
        from fastapi import HTTPException

        request = RecoveryRequest(strategy="invalid")

        with pytest.raises(HTTPException) as exc_info:
            await recover_batch("batch-123", request)

        assert exc_info.value.status_code == 400


class TestSystemOperations:
    """Tests for system operation endpoints."""

    @pytest.mark.asyncio
    async def test_flush_cache_pattern(self, admin_user):
        """Test flushing cache with pattern."""
        from titan.api.admin_routes import flush_cache

        with patch.dict("sys.modules", {"redis": MagicMock()}):
            import sys
            mock_redis = sys.modules["redis"]
            mock_client = MagicMock()
            mock_client.keys.return_value = [b"key1", b"key2"]
            mock_redis.from_url.return_value = mock_client

            result = await flush_cache(pattern="titan:*")

            assert result["keys_deleted"] == 2
            mock_client.keys.assert_called_with("titan:*")

    @pytest.mark.asyncio
    async def test_get_audit_events(self, admin_user):
        """Test getting audit events."""
        from titan.api.admin_routes import get_audit_events

        mock_events = [
            {
                "id": uuid4(),
                "timestamp": datetime.now(timezone.utc),
                "event_type": "agent.created",
                "action": "create",
                "agent_id": "agent-1",
                "session_id": None,
                "user_id": None,
            },
        ]

        with patch("titan.persistence.postgres.get_postgres_client") as mock_pg:
            client = MagicMock()
            client.get_audit_events = AsyncMock(return_value=mock_events)
            mock_pg.return_value = client

            result = await get_audit_events()

            assert len(result) == 1
            assert result[0].event_type == "agent.created"
