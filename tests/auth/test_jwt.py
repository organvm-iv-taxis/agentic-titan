"""
Tests for titan.auth.jwt module.

Tests JWT token creation, verification, and refresh functionality.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

# Mock jose before importing jwt module
jose_mock = MagicMock()
jose_mock.jwt = MagicMock()


class TestJWTBasics:
    """Tests for basic JWT operations."""

    def test_import_error_without_jose(self):
        """Test that helpful error is raised when jose not installed."""
        with patch.dict("sys.modules", {"jose": None}):
            from titan.auth import jwt

            with pytest.raises(ImportError, match="python-jose"):
                jwt._get_jose()

    def test_create_access_token(self):
        """Test creating an access token."""
        with patch("titan.auth.jwt._get_jose") as mock_jose:
            mock_jwt = MagicMock()
            mock_jwt.encode.return_value = "test_access_token"
            mock_jose.return_value = (mock_jwt, Exception)

            from titan.auth.jwt import create_access_token

            token = create_access_token(  # allow-secret
                user_id="user-123",
                username="testuser",
                role="user",
            )

            assert token == "test_access_token"  # allow-secret
            mock_jwt.encode.assert_called_once()

            # Verify payload structure
            call_args = mock_jwt.encode.call_args
            payload = call_args[0][0]
            assert payload["sub"] == "user-123"
            assert payload["username"] == "testuser"
            assert payload["role"] == "user"
            assert payload["token_type"] == "access"
            assert "exp" in payload
            assert "iat" in payload

    def test_create_refresh_token(self):
        """Test creating a refresh token."""
        with patch("titan.auth.jwt._get_jose") as mock_jose:
            mock_jwt = MagicMock()
            mock_jwt.encode.return_value = "test_refresh_token"
            mock_jose.return_value = (mock_jwt, Exception)

            from titan.auth.jwt import create_refresh_token

            token = create_refresh_token(  # allow-secret
                user_id="user-123",
                username="testuser",
                role="user",
            )

            assert token == "test_refresh_token"  # allow-secret

            # Verify it's a refresh token
            call_args = mock_jwt.encode.call_args
            payload = call_args[0][0]
            assert payload["token_type"] == "refresh"

    def test_create_access_token_custom_expiry(self):
        """Test creating token with custom expiry."""
        with patch("titan.auth.jwt._get_jose") as mock_jose:
            mock_jwt = MagicMock()
            mock_jwt.encode.return_value = "test_token"
            mock_jose.return_value = (mock_jwt, Exception)

            from titan.auth.jwt import create_access_token

            token = create_access_token(  # allow-secret
                user_id="user-123",
                username="testuser",
                role="admin",
                expires_delta=timedelta(hours=2),
            )

            assert token == "test_token"  # allow-secret

    def test_decode_token(self):
        """Test decoding a valid token."""
        with patch("titan.auth.jwt._get_jose") as mock_jose:
            mock_jwt = MagicMock()
            expected_payload = {
                "sub": "user-123",
                "username": "testuser",
                "role": "user",
                "token_type": "access",
            }
            mock_jwt.decode.return_value = expected_payload
            mock_jose.return_value = (mock_jwt, Exception)

            from titan.auth.jwt import decode_token

            payload = decode_token("valid_token")

            assert payload == expected_payload
            mock_jwt.decode.assert_called_once()

    def test_decode_token_invalid(self):
        """Test decoding an invalid token raises JWTError."""
        with patch("titan.auth.jwt._get_jose") as mock_jose:
            mock_jwt = MagicMock()

            class MockJWTError(Exception):
                pass

            mock_jwt.decode.side_effect = MockJWTError("Invalid token")
            mock_jose.return_value = (mock_jwt, MockJWTError)

            from titan.auth.jwt import decode_token, JWTError

            with pytest.raises(JWTError, match="Invalid token"):
                decode_token("invalid_token")

    def test_verify_token_access(self):
        """Test verifying an access token."""
        with patch("titan.auth.jwt._get_jose") as mock_jose:
            mock_jwt = MagicMock()
            mock_jwt.decode.return_value = {
                "sub": "user-123",
                "token_type": "access",
            }
            mock_jose.return_value = (mock_jwt, Exception)

            from titan.auth.jwt import verify_token

            payload = verify_token("access_token", expected_type="access")

            assert payload["sub"] == "user-123"
            assert payload["token_type"] == "access"

    def test_verify_token_wrong_type(self):
        """Test verifying token with wrong type raises error."""
        with patch("titan.auth.jwt._get_jose") as mock_jose:
            mock_jwt = MagicMock()
            mock_jwt.decode.return_value = {
                "sub": "user-123",
                "token_type": "refresh",
            }
            mock_jose.return_value = (mock_jwt, Exception)

            from titan.auth.jwt import verify_token, JWTError

            with pytest.raises(JWTError, match="Invalid token type"):
                verify_token("refresh_token", expected_type="access")

    def test_create_token_pair(self):
        """Test creating both access and refresh tokens."""
        with patch("titan.auth.jwt._get_jose") as mock_jose:
            mock_jwt = MagicMock()
            mock_jwt.encode.side_effect = ["access_token", "refresh_token"]
            mock_jose.return_value = (mock_jwt, Exception)

            from titan.auth.jwt import create_token_pair

            access, refresh, expires_in = create_token_pair(
                user_id="user-123",
                username="testuser",
                role="user",
            )

            assert access == "access_token"  # allow-secret
            assert refresh == "refresh_token"  # allow-secret
            assert expires_in > 0
            assert mock_jwt.encode.call_count == 2

    def test_refresh_access_token(self):
        """Test refreshing an access token."""
        with patch("titan.auth.jwt._get_jose") as mock_jose:
            mock_jwt = MagicMock()

            # First decode returns refresh token payload
            mock_jwt.decode.return_value = {
                "sub": "user-123",
                "username": "testuser",
                "role": "user",
                "token_type": "refresh",
            }
            # Then encode creates new access token
            mock_jwt.encode.return_value = "new_access_token"
            mock_jose.return_value = (mock_jwt, Exception)

            from titan.auth.jwt import refresh_access_token

            access, expires_in = refresh_access_token("valid_refresh_token")

            assert access == "new_access_token"  # allow-secret
            assert expires_in > 0

    def test_refresh_access_token_invalid_payload(self):
        """Test refreshing with invalid payload raises error."""
        with patch("titan.auth.jwt._get_jose") as mock_jose:
            mock_jwt = MagicMock()
            mock_jwt.decode.return_value = {
                "token_type": "refresh",
                # Missing sub, username, role
            }
            mock_jose.return_value = (mock_jwt, Exception)

            from titan.auth.jwt import refresh_access_token, JWTError

            with pytest.raises(JWTError, match="Invalid refresh token payload"):
                refresh_access_token("incomplete_refresh_token")

    def test_additional_claims(self):
        """Test adding additional claims to token."""
        with patch("titan.auth.jwt._get_jose") as mock_jose:
            mock_jwt = MagicMock()
            mock_jwt.encode.return_value = "custom_token"
            mock_jose.return_value = (mock_jwt, Exception)

            from titan.auth.jwt import create_access_token

            token = create_access_token(  # allow-secret
                user_id="user-123",
                username="testuser",
                role="user",
                additional_claims={"custom_field": "custom_value"},
            )

            call_args = mock_jwt.encode.call_args
            payload = call_args[0][0]
            assert payload["custom_field"] == "custom_value"
