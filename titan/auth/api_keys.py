"""
Titan Authentication - API Key Management

Provides API key generation, hashing, and verification.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("titan.auth.api_keys")

# API key configuration
API_KEY_PREFIX = os.getenv("TITAN_API_KEY_PREFIX", "titan_")
API_KEY_LENGTH = int(os.getenv("TITAN_API_KEY_LENGTH", "32"))


class APIKeyError(Exception):
    """API key-related error."""

    pass


def generate_api_key() -> tuple[str, str, str]:
    """
    Generate a new API key.

    Returns:
        Tuple of (full_key, key_hash, key_prefix)
        - full_key: The complete API key to give to the user (only shown once)
        - key_hash: SHA256 hash of the key for storage
        - key_prefix: First 8 characters for identification
    """
    # Generate random bytes and convert to hex
    random_bytes = secrets.token_hex(API_KEY_LENGTH)
    full_key = f"{API_KEY_PREFIX}{random_bytes}"

    # Create hash for storage
    key_hash = hash_api_key(full_key)

    # Extract prefix for identification
    key_prefix = full_key[:8]

    return full_key, key_hash, key_prefix


def hash_api_key(key: str) -> str:
    """
    Hash an API key for secure storage.

    Args:
        key: The API key to hash

    Returns:
        SHA256 hash of the key
    """
    return hashlib.sha256(key.encode()).hexdigest()


def verify_api_key(provided_key: str, stored_hash: str) -> bool:
    """
    Verify an API key against a stored hash.

    Args:
        provided_key: The API key provided by the user
        stored_hash: The stored hash to verify against

    Returns:
        True if the key is valid, False otherwise
    """
    provided_hash = hash_api_key(provided_key)
    # Use secrets.compare_digest for timing-safe comparison
    return secrets.compare_digest(provided_hash, stored_hash)


def parse_api_key_header(header_value: str | None) -> str | None:
    """
    Parse an API key from a header value.

    Supports formats:
    - "titan_abc123..."  (raw key)
    - "Bearer titan_abc123..."  (with Bearer prefix)

    Args:
        header_value: The header value to parse

    Returns:
        The extracted API key or None if not found
    """
    if not header_value:
        return None

    # Strip whitespace
    header_value = header_value.strip()

    # Handle Bearer prefix
    if header_value.lower().startswith("bearer "):
        header_value = header_value[7:].strip()

    # Validate it looks like an API key
    if header_value.startswith(API_KEY_PREFIX):
        return header_value

    return None


def get_key_prefix(key: str) -> str:
    """
    Extract the prefix from an API key for identification.

    Args:
        key: The full API key

    Returns:
        The first 8 characters of the key
    """
    return key[:8] if len(key) >= 8 else key


def calculate_expiry(days: int | None) -> datetime | None:
    """
    Calculate expiry datetime from days.

    Args:
        days: Number of days until expiry, or None for no expiry

    Returns:
        Expiry datetime or None
    """
    if days is None:
        return None
    return datetime.now(timezone.utc) + timedelta(days=days)


def is_key_valid(
    key_hash: str,
    stored_hash: str,
    expires_at: datetime | None,
    is_active: bool,
) -> tuple[bool, str | None]:
    """
    Check if an API key is valid.

    Args:
        key_hash: Hash of the provided key
        stored_hash: Stored hash to compare against
        expires_at: Expiry datetime or None
        is_active: Whether the key is active

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if active
    if not is_active:
        return False, "API key is deactivated"

    # Check expiry
    if expires_at and datetime.now(timezone.utc) > expires_at:
        return False, "API key has expired"

    # Verify hash
    if not secrets.compare_digest(key_hash, stored_hash):
        return False, "Invalid API key"

    return True, None
