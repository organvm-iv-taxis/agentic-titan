"""
Titan API - Rate Limiting

Provides rate limiting for API endpoints using slowapi with Redis backend.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

from fastapi import Request, Response
from starlette.responses import JSONResponse

logger = logging.getLogger("titan.api.rate_limit")

# Configuration from environment
REDIS_URL = os.getenv("TITAN_REDIS_URL", "redis://localhost:6379")
DEFAULT_RATE_LIMIT = os.getenv("TITAN_DEFAULT_RATE_LIMIT", "100/minute")


def _get_slowapi():
    """Lazy import of slowapi."""
    try:
        from slowapi import Limiter
        from slowapi.util import get_remote_address
        from slowapi.errors import RateLimitExceeded
        from slowapi.middleware import SlowAPIMiddleware
        return Limiter, get_remote_address, RateLimitExceeded, SlowAPIMiddleware
    except ImportError:
        raise ImportError(
            "slowapi is required for rate limiting. "
            "Install with: pip install 'agentic-titan[ratelimit]'"
        )


def get_user_identifier(request: Request) -> str:
    """
    Get a unique identifier for rate limiting.

    Priority:
    1. Authenticated user ID
    2. API key prefix
    3. Remote IP address

    Args:
        request: The FastAPI request

    Returns:
        Unique identifier string
    """
    # Check for authenticated user
    if hasattr(request.state, "user") and request.state.user:
        return f"user:{request.state.user.id}"

    # Check for API key header
    api_key = request.headers.get("X-API-Key")  # allow-secret
    if api_key:
        return f"key:{api_key[:8]}"

    # Fall back to IP address
    _, get_remote_address, _, _ = _get_slowapi()
    return f"ip:{get_remote_address(request)}"


def create_limiter() -> "Limiter":
    """
    Create and configure the rate limiter.

    Returns:
        Configured Limiter instance
    """
    Limiter, _, _, _ = _get_slowapi()

    return Limiter(
        key_func=get_user_identifier,
        storage_uri=REDIS_URL,
        default_limits=[DEFAULT_RATE_LIMIT],
        strategy="fixed-window",  # Options: fixed-window, moving-window
        headers_enabled=True,  # Include X-RateLimit-* headers
    )


def rate_limit_exceeded_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handler for rate limit exceeded errors.

    Args:
        request: The FastAPI request
        exc: The RateLimitExceeded exception

    Returns:
        JSON response with rate limit error
    """
    _, _, RateLimitExceeded, _ = _get_slowapi()

    # Extract retry-after from exception if available
    retry_after = 60  # Default
    if hasattr(exc, "detail"):
        logger.warning(f"Rate limit exceeded: {exc.detail}")

    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please try again later.",
            "retry_after": retry_after,
        },
        headers={
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit": DEFAULT_RATE_LIMIT,
        },
    )


# Predefined rate limits for different endpoint types
RATE_LIMITS = {
    # Authentication endpoints (more restrictive to prevent brute force)
    "auth_login": "5/minute",
    "auth_refresh": "10/minute",
    "auth_api_keys": "10/minute",

    # Batch processing (expensive operations)
    "batch_submit": "10/minute",
    "batch_start": "20/minute",
    "batch_export": "5/minute",

    # Inquiry endpoints
    "inquiry_start": "20/minute",
    "inquiry_run_all": "10/minute",

    # Admin endpoints (more permissive for admins)
    "admin_users": "50/minute",
    "admin_config": "30/minute",

    # Default for unspecified endpoints
    "default": "100/minute",
}


def get_rate_limit(endpoint_key: str) -> str:
    """
    Get the rate limit for a specific endpoint.

    Args:
        endpoint_key: Key from RATE_LIMITS dict

    Returns:
        Rate limit string (e.g., "10/minute")
    """
    return RATE_LIMITS.get(endpoint_key, RATE_LIMITS["default"])


# Global limiter instance (lazy initialized)
_limiter: "Limiter" | None = None


def get_limiter() -> "Limiter":
    """Get or create the global limiter instance."""
    global _limiter
    if _limiter is None:
        try:
            _limiter = create_limiter()
        except ImportError:
            # Return a no-op limiter if slowapi not installed
            logger.warning("slowapi not installed, rate limiting disabled")
            return None
    return _limiter


def setup_rate_limiting(app) -> None:
    """
    Set up rate limiting for a FastAPI application.

    Args:
        app: FastAPI application instance
    """
    try:
        Limiter, _, RateLimitExceeded, SlowAPIMiddleware = _get_slowapi()

        limiter = get_limiter()
        if limiter is None:
            return

        # Store limiter in app state for access in routes
        app.state.limiter = limiter

        # Add rate limit exceeded handler
        app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

        logger.info("Rate limiting configured")

    except ImportError:
        logger.warning("slowapi not installed, rate limiting disabled")


def limit(rate: str) -> Callable:
    """
    Decorator factory for rate limiting individual endpoints.

    Args:
        rate: Rate limit string (e.g., "10/minute", "100/hour")

    Returns:
        Decorator function

    Usage:
        @router.post("/submit")
        @limit("10/minute")
        async def submit_batch(...):
            ...
    """
    limiter = get_limiter()
    if limiter is None:
        # Return no-op decorator if limiter not available
        def noop_decorator(func):
            return func
        return noop_decorator

    return limiter.limit(rate)


# Convenience decorators for common rate limits
def auth_limit(func: Callable) -> Callable:
    """Apply authentication endpoint rate limit."""
    return limit(RATE_LIMITS["auth_login"])(func)


def batch_limit(func: Callable) -> Callable:
    """Apply batch endpoint rate limit."""
    return limit(RATE_LIMITS["batch_submit"])(func)


def inquiry_limit(func: Callable) -> Callable:
    """Apply inquiry endpoint rate limit."""
    return limit(RATE_LIMITS["inquiry_start"])(func)
