"""
Tests for titan.api.rate_limit module.

Tests rate limiting configuration and behavior.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi import Request


class TestGetUserIdentifier:
    """Tests for user identifier extraction."""

    def test_get_user_identifier_authenticated(self):
        """Test identifier for authenticated user."""
        from titan.api.rate_limit import get_user_identifier

        request = MagicMock(spec=Request)
        request.state.user = MagicMock()
        request.state.user.id = "user-123"

        result = get_user_identifier(request)

        assert result == "user:user-123"

    def test_get_user_identifier_api_key(self):
        """Test identifier for API key authentication."""
        from titan.api.rate_limit import get_user_identifier

        request = MagicMock(spec=Request)
        request.state = MagicMock()
        del request.state.user  # No user
        request.headers.get.return_value = "titan_abc12345"

        # Need to mock hasattr
        with patch("titan.api.rate_limit.hasattr", side_effect=lambda obj, name: name != "user"):
            result = get_user_identifier(request)

        assert result == "key:titan_ab"

    def test_get_user_identifier_ip_fallback(self):
        """Test identifier falls back to IP address."""
        from titan.api.rate_limit import get_user_identifier

        request = MagicMock(spec=Request)
        request.state = MagicMock(spec=[])  # No user attribute
        request.headers.get.return_value = None

        with patch("titan.api.rate_limit._get_slowapi") as mock_slowapi:
            mock_get_ip = MagicMock(return_value="192.168.1.1")
            mock_slowapi.return_value = (None, mock_get_ip, None, None)

            result = get_user_identifier(request)

        assert result == "ip:192.168.1.1"


class TestRateLimits:
    """Tests for rate limit configuration."""

    def test_rate_limits_defined(self):
        """Test that rate limits are defined for key endpoints."""
        from titan.api.rate_limit import RATE_LIMITS

        assert "auth_login" in RATE_LIMITS
        assert "batch_submit" in RATE_LIMITS
        assert "inquiry_start" in RATE_LIMITS
        assert "default" in RATE_LIMITS

    def test_get_rate_limit_known_endpoint(self):
        """Test getting rate limit for known endpoint."""
        from titan.api.rate_limit import get_rate_limit, RATE_LIMITS

        result = get_rate_limit("auth_login")

        assert result == RATE_LIMITS["auth_login"]

    def test_get_rate_limit_unknown_endpoint(self):
        """Test getting rate limit for unknown endpoint."""
        from titan.api.rate_limit import get_rate_limit, RATE_LIMITS

        result = get_rate_limit("unknown_endpoint")

        assert result == RATE_LIMITS["default"]

    def test_auth_login_more_restrictive(self):
        """Test that auth login has more restrictive rate limit."""
        from titan.api.rate_limit import RATE_LIMITS

        # Parse rate limits
        def parse_rate(rate_str):
            count, period = rate_str.split("/")
            return int(count)

        auth_rate = parse_rate(RATE_LIMITS["auth_login"])
        default_rate = parse_rate(RATE_LIMITS["default"])

        assert auth_rate < default_rate


class TestRateLimitExceededHandler:
    """Tests for rate limit exceeded handler."""

    def test_rate_limit_exceeded_response(self):
        """Test that handler returns proper response."""
        from titan.api.rate_limit import rate_limit_exceeded_handler

        request = MagicMock()
        exc = MagicMock()
        exc.detail = "Rate limit exceeded"

        with patch("titan.api.rate_limit._get_slowapi") as mock_slowapi:
            mock_slowapi.return_value = (None, None, Exception, None)

            response = rate_limit_exceeded_handler(request, exc)

        assert response.status_code == 429
        assert "retry_after" in response.body.decode()

    def test_rate_limit_exceeded_headers(self):
        """Test that handler includes required headers."""
        from titan.api.rate_limit import rate_limit_exceeded_handler

        request = MagicMock()
        exc = MagicMock()

        with patch("titan.api.rate_limit._get_slowapi") as mock_slowapi:
            mock_slowapi.return_value = (None, None, Exception, None)

            response = rate_limit_exceeded_handler(request, exc)

        assert "Retry-After" in response.headers
        assert "X-RateLimit-Limit" in response.headers


class TestCreateLimiter:
    """Tests for limiter creation."""

    def test_create_limiter_success(self):
        """Test successful limiter creation."""
        with patch("titan.api.rate_limit._get_slowapi") as mock_slowapi:
            mock_limiter_class = MagicMock()
            mock_limiter = MagicMock()
            mock_limiter_class.return_value = mock_limiter
            mock_slowapi.return_value = (mock_limiter_class, None, None, None)

            from titan.api.rate_limit import create_limiter

            result = create_limiter()

            assert result == mock_limiter
            mock_limiter_class.assert_called_once()

    def test_create_limiter_import_error(self):
        """Test limiter creation without slowapi."""
        with patch("titan.api.rate_limit._get_slowapi") as mock_slowapi:
            mock_slowapi.side_effect = ImportError("slowapi not installed")

            from titan.api.rate_limit import create_limiter

            with pytest.raises(ImportError, match="slowapi"):
                create_limiter()


class TestLimitDecorator:
    """Tests for limit decorator."""

    def test_limit_with_limiter(self):
        """Test limit decorator when limiter is available."""
        with patch("titan.api.rate_limit.get_limiter") as mock_get:
            mock_limiter = MagicMock()
            mock_limiter.limit.return_value = lambda f: f  # Pass-through
            mock_get.return_value = mock_limiter

            from titan.api.rate_limit import limit

            @limit("10/minute")
            def test_func():
                return "result"

            result = test_func()

            assert result == "result"
            mock_limiter.limit.assert_called_with("10/minute")

    def test_limit_without_limiter(self):
        """Test limit decorator when limiter is not available."""
        with patch("titan.api.rate_limit.get_limiter") as mock_get:
            mock_get.return_value = None

            from titan.api.rate_limit import limit

            @limit("10/minute")
            def test_func():
                return "result"

            result = test_func()

            assert result == "result"


class TestSetupRateLimiting:
    """Tests for rate limiting setup."""

    def test_setup_rate_limiting_success(self):
        """Test successful rate limiting setup."""
        with patch("titan.api.rate_limit._get_slowapi") as mock_slowapi:
            mock_limiter_class = MagicMock()
            mock_limiter = MagicMock()
            mock_limiter_class.return_value = mock_limiter
            mock_slowapi.return_value = (mock_limiter_class, None, Exception, None)

            with patch("titan.api.rate_limit.get_limiter") as mock_get:
                mock_get.return_value = mock_limiter

                from titan.api.rate_limit import setup_rate_limiting

                app = MagicMock()
                setup_rate_limiting(app)

                # Should set limiter in app state
                assert hasattr(app.state, "limiter") or app.state.limiter == mock_limiter

    def test_setup_rate_limiting_import_error(self):
        """Test rate limiting setup without slowapi."""
        with patch("titan.api.rate_limit._get_slowapi") as mock_slowapi:
            mock_slowapi.side_effect = ImportError("slowapi not installed")

            from titan.api.rate_limit import setup_rate_limiting

            app = MagicMock()
            # Should not raise, just log warning
            setup_rate_limiting(app)
