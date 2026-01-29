"""
Titan Authentication System

Provides JWT-based authentication and API key management for the Titan API.

Components:
- models: User, APIKey, TokenPair, UserRole data models
- jwt: JWT token creation and verification
- api_keys: API key generation and validation
- middleware: FastAPI authentication dependencies
- storage: PostgreSQL backend for users and API keys
"""

from titan.auth.models import (
    User,
    UserCreate,
    UserRole,
    APIKey,
    APIKeyCreate,
    TokenPair,
    TokenData,
)
from titan.auth.jwt import (
    create_access_token,
    create_refresh_token,
    verify_token,
    decode_token,
)
from titan.auth.api_keys import (
    generate_api_key,
    hash_api_key,
    verify_api_key,
)
from titan.auth.middleware import (
    get_current_user,
    get_current_user_optional,
    require_role,
    require_admin,
)

__all__ = [
    # Models
    "User",
    "UserCreate",
    "UserRole",
    "APIKey",
    "APIKeyCreate",
    "TokenPair",
    "TokenData",
    # JWT
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "decode_token",
    # API Keys
    "generate_api_key",
    "hash_api_key",
    "verify_api_key",
    # Middleware
    "get_current_user",
    "get_current_user_optional",
    "require_role",
    "require_admin",
]
