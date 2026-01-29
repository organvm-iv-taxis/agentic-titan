"""
Tests for titan.auth.api_keys module.

Tests API key generation, hashing, and validation.
"""

import pytest
from datetime import datetime, timedelta, timezone

from titan.auth.api_keys import (
    generate_api_key,
    hash_api_key,
    verify_api_key,
    parse_api_key_header,
    get_key_prefix,
    calculate_expiry,
    is_key_valid,
    API_KEY_PREFIX,
)


class TestAPIKeyGeneration:
    """Tests for API key generation."""

    def test_generate_api_key(self):
        """Test generating a new API key."""
        full_key, key_hash, key_prefix = generate_api_key()

        # Key should start with prefix
        assert full_key.startswith(API_KEY_PREFIX)

        # Hash should be 64 characters (SHA256)
        assert len(key_hash) == 64

        # Prefix should be first 8 characters
        assert key_prefix == full_key[:8]

    def test_generate_unique_keys(self):
        """Test that generated keys are unique."""
        keys = set()
        for _ in range(100):
            full_key, _, _ = generate_api_key()
            keys.add(full_key)

        # All keys should be unique
        assert len(keys) == 100

    def test_key_length(self):
        """Test that generated keys have expected length."""
        full_key, _, _ = generate_api_key()

        # Key should be prefix + 64 hex characters (32 bytes)
        expected_length = len(API_KEY_PREFIX) + 64
        assert len(full_key) == expected_length


class TestAPIKeyHashing:
    """Tests for API key hashing."""

    def test_hash_api_key(self):
        """Test hashing an API key."""
        key = "titan_abc123def456"
        key_hash = hash_api_key(key)

        # Hash should be 64 characters (SHA256)
        assert len(key_hash) == 64

        # Hash should be consistent
        assert hash_api_key(key) == key_hash

    def test_different_keys_different_hashes(self):
        """Test that different keys produce different hashes."""
        key1 = "titan_key1"
        key2 = "titan_key2"

        hash1 = hash_api_key(key1)
        hash2 = hash_api_key(key2)

        assert hash1 != hash2


class TestAPIKeyVerification:
    """Tests for API key verification."""

    def test_verify_api_key_valid(self):
        """Test verifying a valid API key."""
        full_key, key_hash, _ = generate_api_key()

        assert verify_api_key(full_key, key_hash) is True

    def test_verify_api_key_invalid(self):
        """Test verifying an invalid API key."""
        _, key_hash, _ = generate_api_key()

        assert verify_api_key("wrong_key", key_hash) is False

    def test_verify_api_key_timing_safe(self):
        """Test that verification uses timing-safe comparison."""
        # This test verifies the function works correctly
        # Actual timing attack resistance requires statistical analysis
        full_key, key_hash, _ = generate_api_key()

        # Should work with correct key
        assert verify_api_key(full_key, key_hash) is True

        # Should fail with similar but wrong key
        wrong_key = full_key[:-1] + "X"
        assert verify_api_key(wrong_key, key_hash) is False


class TestAPIKeyHeaderParsing:
    """Tests for API key header parsing."""

    def test_parse_raw_key(self):
        """Test parsing a raw API key."""
        key = f"{API_KEY_PREFIX}abc123"
        result = parse_api_key_header(key)
        assert result == key

    def test_parse_bearer_key(self):
        """Test parsing a Bearer-prefixed API key."""
        key = f"{API_KEY_PREFIX}abc123"
        result = parse_api_key_header(f"Bearer {key}")
        assert result == key

    def test_parse_bearer_lowercase(self):
        """Test parsing with lowercase bearer."""
        key = f"{API_KEY_PREFIX}abc123"
        result = parse_api_key_header(f"bearer {key}")
        assert result == key

    def test_parse_with_whitespace(self):
        """Test parsing with extra whitespace."""
        key = f"{API_KEY_PREFIX}abc123"
        result = parse_api_key_header(f"  Bearer   {key}  ")
        assert result == key

    def test_parse_invalid_key(self):
        """Test parsing an invalid key format."""
        result = parse_api_key_header("not_a_valid_key")
        assert result is None

    def test_parse_none(self):
        """Test parsing None."""
        result = parse_api_key_header(None)
        assert result is None

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = parse_api_key_header("")
        assert result is None


class TestKeyPrefix:
    """Tests for key prefix extraction."""

    def test_get_key_prefix(self):
        """Test extracting key prefix."""
        key = "titan_abc123def456"
        prefix = get_key_prefix(key)
        assert prefix == "titan_ab"

    def test_get_key_prefix_short_key(self):
        """Test extracting prefix from short key."""
        key = "short"
        prefix = get_key_prefix(key)
        assert prefix == "short"


class TestExpiryCalculation:
    """Tests for expiry calculation."""

    def test_calculate_expiry_days(self):
        """Test calculating expiry with days."""
        expiry = calculate_expiry(30)

        assert expiry is not None
        # Should be approximately 30 days from now
        expected = datetime.now(timezone.utc) + timedelta(days=30)
        assert abs((expiry - expected).total_seconds()) < 2

    def test_calculate_expiry_none(self):
        """Test calculating expiry with None."""
        expiry = calculate_expiry(None)
        assert expiry is None


class TestKeyValidation:
    """Tests for comprehensive key validation."""

    def test_is_key_valid_all_conditions_met(self):
        """Test validation when all conditions are met."""
        full_key, key_hash, _ = generate_api_key()
        provided_hash = hash_api_key(full_key)

        is_valid, error = is_key_valid(
            key_hash=provided_hash,
            stored_hash=key_hash,
            expires_at=None,
            is_active=True,
        )

        assert is_valid is True
        assert error is None

    def test_is_key_valid_inactive(self):
        """Test validation when key is inactive."""
        full_key, key_hash, _ = generate_api_key()
        provided_hash = hash_api_key(full_key)

        is_valid, error = is_key_valid(
            key_hash=provided_hash,
            stored_hash=key_hash,
            expires_at=None,
            is_active=False,
        )

        assert is_valid is False
        assert "deactivated" in error.lower()

    def test_is_key_valid_expired(self):
        """Test validation when key is expired."""
        full_key, key_hash, _ = generate_api_key()
        provided_hash = hash_api_key(full_key)

        expired = datetime.now(timezone.utc) - timedelta(days=1)

        is_valid, error = is_key_valid(
            key_hash=provided_hash,
            stored_hash=key_hash,
            expires_at=expired,
            is_active=True,
        )

        assert is_valid is False
        assert "expired" in error.lower()

    def test_is_key_valid_wrong_hash(self):
        """Test validation when hash doesn't match."""
        _, key_hash, _ = generate_api_key()
        wrong_hash = hash_api_key("wrong_key")

        is_valid, error = is_key_valid(
            key_hash=wrong_hash,
            stored_hash=key_hash,
            expires_at=None,
            is_active=True,
        )

        assert is_valid is False
        assert "invalid" in error.lower()

    def test_is_key_valid_not_yet_expired(self):
        """Test validation when key expires in the future."""
        full_key, key_hash, _ = generate_api_key()
        provided_hash = hash_api_key(full_key)

        future = datetime.now(timezone.utc) + timedelta(days=30)

        is_valid, error = is_key_valid(
            key_hash=provided_hash,
            stored_hash=key_hash,
            expires_at=future,
            is_active=True,
        )

        assert is_valid is True
        assert error is None
