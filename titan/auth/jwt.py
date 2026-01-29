"""
Titan Authentication - JWT Handling

Provides JWT token creation and verification for access and refresh tokens.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import ValidationError

logger = logging.getLogger("titan.auth.jwt")

# Configuration from environment
JWT_SECRET_KEY = os.getenv("TITAN_JWT_SECRET", "titan-dev-secret-change-in-production")  # allow-secret
JWT_ALGORITHM = os.getenv("TITAN_JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("TITAN_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("TITAN_REFRESH_TOKEN_EXPIRE_DAYS", "7"))


class JWTError(Exception):
    """JWT-related error."""

    pass


def _get_jose():
    """Lazy import of python-jose."""
    try:
        from jose import jwt, JWTError as JoseJWTError
        return jwt, JoseJWTError
    except ImportError:
        raise ImportError(
            "python-jose is required for JWT authentication. "
            "Install with: pip install 'agentic-titan[auth]'"
        )


def create_access_token(
    user_id: str,
    username: str,
    role: str,
    expires_delta: timedelta | None = None,
    additional_claims: dict[str, Any] | None = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        user_id: User's unique identifier
        username: User's username
        role: User's role
        expires_delta: Optional custom expiration time
        additional_claims: Optional additional JWT claims

    Returns:
        Encoded JWT access token
    """
    jwt_module, _ = _get_jose()

    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    now = datetime.now(timezone.utc)
    expire = now + expires_delta

    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "token_type": "access",
        "iat": now,
        "exp": expire,
    }

    if additional_claims:
        payload.update(additional_claims)

    return jwt_module.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def create_refresh_token(
    user_id: str,
    username: str,
    role: str,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a JWT refresh token.

    Args:
        user_id: User's unique identifier
        username: User's username
        role: User's role
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT refresh token
    """
    jwt_module, _ = _get_jose()

    if expires_delta is None:
        expires_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    now = datetime.now(timezone.utc)
    expire = now + expires_delta

    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "token_type": "refresh",
        "iat": now,
        "exp": expire,
    }

    return jwt_module.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:  # allow-secret
    """
    Decode and verify a JWT.

    Args:
        token: The JWT value to decode  # allow-secret

    Returns:
        Decoded payload

    Raises:
        JWTError: If token is invalid or expired
    """
    jwt_module, JoseJWTError = _get_jose()

    try:
        payload = jwt_module.decode(
            token,  # allow-secret
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
        )
        return payload
    except JoseJWTError as e:
        logger.warning(f"JWT decode error: {e}")
        raise JWTError(f"Invalid token: {e}")  # allow-secret


def verify_token(token: str, expected_type: str = "access") -> dict[str, Any]:  # allow-secret
    """
    Verify a JWT and check its type.

    Args:
        token: The JWT value to verify  # allow-secret
        expected_type: Expected type ("access" or "refresh")

    Returns:
        Decoded payload

    Raises:
        JWTError: If token is invalid, expired, or wrong type
    """
    payload = decode_token(token)  # allow-secret

    token_type = payload.get("token_type", "access")  # allow-secret
    if token_type != expected_type:
        raise JWTError(f"Invalid token type: expected {expected_type}, got {token_type}")

    return payload


def create_token_pair(
    user_id: str,
    username: str,
    role: str,
) -> tuple[str, str, int]:
    """
    Create both access and refresh tokens.

    Args:
        user_id: User's unique identifier
        username: User's username
        role: User's role

    Returns:
        Tuple of (access_token, refresh_token, expires_in_seconds)
    """
    access_token = create_access_token(user_id, username, role)
    refresh_token = create_refresh_token(user_id, username, role)
    expires_in = ACCESS_TOKEN_EXPIRE_MINUTES * 60

    return access_token, refresh_token, expires_in


def refresh_access_token(refresh_token: str) -> tuple[str, int]:
    """
    Create a new access token using a refresh token.

    Args:
        refresh_token: Valid refresh token

    Returns:
        Tuple of (new_access_token, expires_in_seconds)

    Raises:
        JWTError: If refresh token is invalid or expired
    """
    payload = verify_token(refresh_token, expected_type="refresh")

    user_id = payload.get("sub")
    username = payload.get("username")
    role = payload.get("role")

    if not all([user_id, username, role]):
        raise JWTError("Invalid refresh token payload")

    access_token = create_access_token(user_id, username, role)
    expires_in = ACCESS_TOKEN_EXPIRE_MINUTES * 60

    return access_token, expires_in
