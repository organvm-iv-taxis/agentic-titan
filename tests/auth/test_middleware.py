"""
Tests for titan.auth.middleware module.

Tests FastAPI authentication dependencies.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from titan.auth.models import User, UserRole
from titan.auth.middleware import (
    AuthenticationError,
    AuthorizationError,
    require_role,
)


class TestAuthenticationError:
    """Tests for AuthenticationError."""

    def test_authentication_error_default(self):
        """Test default authentication error."""
        error = AuthenticationError()

        assert error.status_code == 401
        assert "credentials" in error.detail.lower()
        assert error.headers["WWW-Authenticate"] == "Bearer"

    def test_authentication_error_custom_message(self):
        """Test authentication error with custom message."""
        error = AuthenticationError("Custom error message")

        assert error.status_code == 401
        assert error.detail == "Custom error message"


class TestAuthorizationError:
    """Tests for AuthorizationError."""

    def test_authorization_error_default(self):
        """Test default authorization error."""
        error = AuthorizationError()

        assert error.status_code == 403
        assert "permissions" in error.detail.lower()

    def test_authorization_error_custom_message(self):
        """Test authorization error with custom message."""
        error = AuthorizationError("Admin only")

        assert error.status_code == 403
        assert error.detail == "Admin only"


class TestRequireRole:
    """Tests for require_role dependency factory."""

    @pytest.fixture
    def admin_user(self):
        """Create an admin user for testing."""
        return User(
            id=uuid4(),
            username="admin",
            hashed_password="hashed",  # allow-secret
            role=UserRole.ADMIN,
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def regular_user(self):
        """Create a regular user for testing."""
        return User(
            id=uuid4(),
            username="user",
            hashed_password="hashed",  # allow-secret
            role=UserRole.USER,
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def readonly_user(self):
        """Create a readonly user for testing."""
        return User(
            id=uuid4(),
            username="readonly",
            hashed_password="hashed",  # allow-secret
            role=UserRole.READONLY,
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_require_role_allowed(self, admin_user):
        """Test that allowed role passes through."""
        checker = require_role([UserRole.ADMIN])

        result = await checker(user=admin_user)

        assert result == admin_user

    @pytest.mark.asyncio
    async def test_require_role_multiple_allowed(self, regular_user):
        """Test that any of multiple allowed roles passes."""
        checker = require_role([UserRole.ADMIN, UserRole.USER])

        result = await checker(user=regular_user)

        assert result == regular_user

    @pytest.mark.asyncio
    async def test_require_role_denied(self, readonly_user):
        """Test that unauthorized role raises error."""
        checker = require_role([UserRole.ADMIN])

        with pytest.raises(AuthorizationError) as exc_info:
            await checker(user=readonly_user)

        assert "admin" in exc_info.value.detail.lower()


class TestUserModel:
    """Tests for User model integration with middleware."""

    def test_user_to_public_dict(self):
        """Test converting user to public dictionary."""
        user = User(
            id=uuid4(),
            username="testuser",
            email="test@example.com",
            hashed_password="secret_hash",  # allow-secret
            role=UserRole.USER,
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )

        public = user.to_public_dict()

        assert "id" in public
        assert public["username"] == "testuser"
        assert public["role"] == "user"
        assert "hashed_password" not in public
        assert "email" not in public  # Not in public dict

    def test_user_from_dict(self):
        """Test creating user from dictionary."""
        user_id = uuid4()
        now = datetime.now(timezone.utc)

        data = {
            "id": str(user_id),
            "username": "testuser",
            "email": "test@example.com",
            "hashed_password": "secret_hash",  # allow-secret
            "role": "admin",
            "is_active": True,
            "created_at": now.isoformat(),
            "updated_at": None,
            "last_login": None,
            "metadata": {"key": "value"},
        }

        user = User.from_dict(data)

        assert user.id == user_id
        assert user.username == "testuser"
        assert user.role == UserRole.ADMIN
        assert user.metadata == {"key": "value"}


class TestJWTAuthentication:
    """Tests for JWT authentication flow."""

    @pytest.mark.asyncio
    async def test_authenticate_jwt_valid(self):
        """Test authenticating with valid JWT."""
        from titan.auth.middleware import _authenticate_jwt

        user_id = uuid4()
        mock_user_data = {
            "id": str(user_id),
            "username": "testuser",
            "hashed_password": "hashed",  # allow-secret
            "role": "user",
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        with patch("titan.auth.jwt.verify_token") as mock_verify:
            mock_verify.return_value = {"sub": str(user_id)}

            with patch("titan.auth.storage.get_auth_storage") as mock_storage:
                storage = AsyncMock()
                storage.get_user.return_value = mock_user_data
                mock_storage.return_value = storage

                user = await _authenticate_jwt("valid_token")

                assert user.username == "testuser"
                mock_verify.assert_called_once_with("valid_token", expected_type="access")

    @pytest.mark.asyncio
    async def test_authenticate_jwt_invalid(self):
        """Test authenticating with invalid JWT."""
        from titan.auth.middleware import _authenticate_jwt
        from titan.auth.jwt import JWTError

        with patch("titan.auth.jwt.verify_token") as mock_verify:
            mock_verify.side_effect = JWTError("Invalid token")

            with pytest.raises(AuthenticationError, match="Invalid token"):
                await _authenticate_jwt("invalid_token")

    @pytest.mark.asyncio
    async def test_authenticate_jwt_user_not_found(self):
        """Test authenticating when user not found."""
        from titan.auth.middleware import _authenticate_jwt

        with patch("titan.auth.jwt.verify_token") as mock_verify:
            mock_verify.return_value = {"sub": str(uuid4())}

            with patch("titan.auth.storage.get_auth_storage") as mock_storage:
                storage = AsyncMock()
                storage.get_user.return_value = None
                mock_storage.return_value = storage

                with pytest.raises(AuthenticationError, match="User not found"):
                    await _authenticate_jwt("valid_token")

    @pytest.mark.asyncio
    async def test_authenticate_jwt_user_disabled(self):
        """Test authenticating when user is disabled."""
        from titan.auth.middleware import _authenticate_jwt

        mock_user_data = {
            "id": str(uuid4()),
            "username": "testuser",
            "hashed_password": "hashed",  # allow-secret
            "role": "user",
            "is_active": False,  # Disabled
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        with patch("titan.auth.jwt.verify_token") as mock_verify:
            mock_verify.return_value = {"sub": str(uuid4())}

            with patch("titan.auth.storage.get_auth_storage") as mock_storage:
                storage = AsyncMock()
                storage.get_user.return_value = mock_user_data
                mock_storage.return_value = storage

                with pytest.raises(AuthenticationError, match="disabled"):
                    await _authenticate_jwt("valid_token")


class TestAPIKeyAuthentication:
    """Tests for API key authentication flow."""

    @pytest.mark.asyncio
    async def test_authenticate_api_key_valid(self):
        """Test authenticating with valid API key."""
        from titan.auth.middleware import _authenticate_api_key

        user_id = uuid4()
        key_data = {
            "id": str(uuid4()),
            "key_hash": "test_hash",
            "user_id": str(user_id),
            "is_active": True,
            "expires_at": None,
        }
        user_data = {
            "id": str(user_id),
            "username": "apiuser",
            "hashed_password": "hashed",  # allow-secret
            "role": "service",
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        with patch("titan.auth.api_keys.hash_api_key") as mock_hash:
            mock_hash.return_value = "test_hash"

            with patch("titan.auth.api_keys.is_key_valid") as mock_valid:
                mock_valid.return_value = (True, None)

                with patch("titan.auth.storage.get_auth_storage") as mock_storage:
                    storage = AsyncMock()
                    storage.get_api_key_by_hash.return_value = key_data
                    storage.get_user.return_value = user_data
                    storage.update_api_key_last_used.return_value = True
                    mock_storage.return_value = storage

                    user = await _authenticate_api_key("titan_test_key")

                    assert user.username == "apiuser"
                    assert user.role == UserRole.SERVICE

    @pytest.mark.asyncio
    async def test_authenticate_api_key_not_found(self):
        """Test authenticating with non-existent API key."""
        from titan.auth.middleware import _authenticate_api_key

        with patch("titan.auth.api_keys.hash_api_key") as mock_hash:
            mock_hash.return_value = "unknown_hash"

            with patch("titan.auth.storage.get_auth_storage") as mock_storage:
                storage = AsyncMock()
                storage.get_api_key_by_hash.return_value = None
                mock_storage.return_value = storage

                with pytest.raises(AuthenticationError, match="Invalid API key"):
                    await _authenticate_api_key("titan_unknown_key")

    @pytest.mark.asyncio
    async def test_authenticate_api_key_expired(self):
        """Test authenticating with expired API key."""
        from titan.auth.middleware import _authenticate_api_key

        key_data = {
            "id": str(uuid4()),
            "key_hash": "test_hash",
            "user_id": str(uuid4()),
            "is_active": True,
            "expires_at": datetime.now(timezone.utc),  # Already expired
        }

        with patch("titan.auth.api_keys.hash_api_key") as mock_hash:
            mock_hash.return_value = "test_hash"

            with patch("titan.auth.api_keys.is_key_valid") as mock_valid:
                mock_valid.return_value = (False, "API key has expired")

                with patch("titan.auth.storage.get_auth_storage") as mock_storage:
                    storage = AsyncMock()
                    storage.get_api_key_by_hash.return_value = key_data
                    mock_storage.return_value = storage

                    with pytest.raises(AuthenticationError, match="expired"):
                        await _authenticate_api_key("titan_expired_key")
