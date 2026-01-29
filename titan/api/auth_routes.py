"""
Titan API - Authentication Routes

REST API endpoints for user authentication and API key management.

Endpoints:
    POST /api/auth/login           - Login with username/password
    POST /api/auth/refresh         - Refresh access token
    POST /api/auth/logout          - Invalidate tokens (placeholder)
    GET  /api/auth/me              - Get current user info
    POST /api/auth/api-keys        - Create a new API key
    GET  /api/auth/api-keys        - List user's API keys
    DELETE /api/auth/api-keys/{id} - Revoke an API key
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from titan.auth.models import (
    User,
    UserRole,
    TokenPair,
    APIKeyCreate,
)
from titan.auth.middleware import (
    get_current_user,
    AuthenticationError,
)

logger = logging.getLogger("titan.api.auth")

auth_router = APIRouter(prefix="/auth", tags=["auth"])


# =============================================================================
# Request/Response Models
# =============================================================================


class LoginResponse(BaseModel):
    """Response for successful login."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict[str, Any]


class RefreshRequest(BaseModel):
    """Request to refresh access token."""

    refresh_token: str


class RefreshResponse(BaseModel):
    """Response for token refresh."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


class APIKeyResponse(BaseModel):
    """Response for API key creation (includes the key only once)."""

    id: str
    key: str  # Full key, only shown on creation
    key_prefix: str
    name: str
    scopes: list[str]
    expires_at: str | None
    created_at: str


class APIKeyListResponse(BaseModel):
    """Response for API key listing (no full key)."""

    id: str
    key_prefix: str
    name: str
    scopes: list[str]
    expires_at: str | None
    is_active: bool
    created_at: str
    last_used_at: str | None


class UserResponse(BaseModel):
    """Response for user info."""

    id: str
    username: str
    email: str | None
    role: str
    is_active: bool
    created_at: str
    last_login: str | None


# =============================================================================
# Endpoints
# =============================================================================


@auth_router.post("/login", response_model=LoginResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> LoginResponse:
    """
    Authenticate with username and password.

    Returns JWT access and refresh tokens.
    """
    from titan.auth.storage import get_auth_storage
    from titan.auth.jwt import create_token_pair

    storage = await get_auth_storage()

    # Look up user
    user_data = await storage.get_user_by_username(form_data.username)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify password
    if not storage.verify_password(form_data.password, user_data["hashed_password"]):  # allow-secret
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active
    if not user_data.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    # Update last login
    await storage.update_last_login(user_data["id"])

    # Create tokens
    access_token, refresh_token, expires_in = create_token_pair(
        user_id=str(user_data["id"]),
        username=user_data["username"],
        role=user_data["role"],
    )

    logger.info(f"User logged in: {user_data['username']}")

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user={
            "id": str(user_data["id"]),
            "username": user_data["username"],
            "role": user_data["role"],
        },
    )


@auth_router.post("/refresh", response_model=RefreshResponse)
async def refresh_token(request: RefreshRequest) -> RefreshResponse:
    """
    Refresh an access token using a refresh token.
    """
    from titan.auth.jwt import refresh_access_token, JWTError

    try:
        access_token, expires_in = refresh_access_token(request.refresh_token)  # allow-secret
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid refresh token: {e}",  # allow-secret
            headers={"WWW-Authenticate": "Bearer"},
        )

    return RefreshResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@auth_router.post("/logout")
async def logout(user: User = Depends(get_current_user)) -> dict[str, str]:
    """
    Logout the current user.

    Note: JWT tokens are stateless, so this endpoint is a placeholder.
    In a full implementation, you would add the token to a blocklist.
    """
    logger.info(f"User logged out: {user.username}")
    return {"message": "Logged out successfully"}


@auth_router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)) -> UserResponse:
    """
    Get the current authenticated user's information.
    """
    return UserResponse(
        id=str(user.id),
        username=user.username,
        email=user.email,
        role=user.role.value,
        is_active=user.is_active,
        created_at=user.created_at.isoformat(),
        last_login=user.last_login.isoformat() if user.last_login else None,
    )


@auth_router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    request: APIKeyCreate,
    user: User = Depends(get_current_user),
) -> APIKeyResponse:
    """
    Create a new API key for the current user.

    The full key is only returned once - store it securely!
    """
    from titan.auth.api_keys import generate_api_key, calculate_expiry
    from titan.auth.storage import get_auth_storage

    # Generate the key
    full_key, key_hash, key_prefix = generate_api_key()
    key_id = uuid4()
    expires_at = calculate_expiry(request.expires_in_days)

    # Store the key
    storage = await get_auth_storage()
    success = await storage.create_api_key(
        key_id=key_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=request.name,
        user_id=user.id,
        scopes=request.scopes,
        expires_at=expires_at,
        metadata=request.metadata,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key",
        )

    logger.info(f"API key created: {key_prefix}... for user {user.username}")

    return APIKeyResponse(
        id=str(key_id),
        key=full_key,  # Only returned once!
        key_prefix=key_prefix,
        name=request.name,
        scopes=request.scopes,
        expires_at=expires_at.isoformat() if expires_at else None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@auth_router.get("/api-keys", response_model=list[APIKeyListResponse])
async def list_api_keys(
    user: User = Depends(get_current_user),
    include_inactive: bool = False,
) -> list[APIKeyListResponse]:
    """
    List all API keys for the current user.

    Note: Full keys are not returned - only prefixes for identification.
    """
    from titan.auth.storage import get_auth_storage

    storage = await get_auth_storage()
    keys = await storage.get_api_keys_for_user(user.id, include_inactive=include_inactive)

    return [
        APIKeyListResponse(
            id=str(k["id"]),
            key_prefix=k["key_prefix"],
            name=k["name"],
            scopes=k.get("scopes", []),
            expires_at=k["expires_at"].isoformat() if k.get("expires_at") else None,
            is_active=k.get("is_active", True),
            created_at=k["created_at"].isoformat() if isinstance(k["created_at"], datetime) else k["created_at"],
            last_used_at=k["last_used_at"].isoformat() if k.get("last_used_at") else None,
        )
        for k in keys
    ]


@auth_router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    user: User = Depends(get_current_user),
) -> dict[str, str]:
    """
    Revoke (deactivate) an API key.

    Users can only revoke their own keys.
    """
    from titan.auth.storage import get_auth_storage

    storage = await get_auth_storage()

    # Get the key to verify ownership
    key_data = await storage.get_api_key(key_id)
    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    # Check ownership (admins can revoke any key)
    if str(key_data["user_id"]) != str(user.id) and user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only revoke your own API keys",
        )

    # Deactivate the key
    success = await storage.deactivate_api_key(key_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key",
        )

    logger.info(f"API key revoked: {key_data['key_prefix']}... by user {user.username}")

    return {"message": "API key revoked successfully"}
