"""Tests for Ray integration module."""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch, MagicMock

from titan.ray import is_ray_available, RAY_AVAILABLE
from titan.ray.config import RayConfig, get_ray_config, set_ray_config
from titan.ray.backend_selector import (
    ComputeBackend,
    select_backend,
    is_distributed,
    get_backend,
    reset_backend,
)


class TestRayAvailability:
    """Tests for Ray availability detection."""

    def test_is_ray_available_returns_bool(self):
        """Test that is_ray_available returns a boolean."""
        result = is_ray_available()
        assert isinstance(result, bool)

    def test_ray_available_constant(self):
        """Test that RAY_AVAILABLE is a boolean."""
        assert isinstance(RAY_AVAILABLE, bool)


class TestRayConfig:
    """Tests for Ray configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = RayConfig()
        assert config.address == "auto"
        assert config.namespace == "titan"
        assert config.num_replicas >= 1
        assert config.min_replicas >= 1
        assert config.max_replicas >= config.num_replicas

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(os.environ, {
            "RAY_ADDRESS": "ray://head:6379",
            "RAY_NAMESPACE": "test",
            "RAY_REPLICAS": "4",
        }):
            config = RayConfig()
            assert config.address == "ray://head:6379"
            assert config.namespace == "test"
            assert config.num_replicas == 4

    def test_to_deployment_config(self):
        """Test conversion to deployment configuration."""
        config = RayConfig()
        deployment_config = config.to_deployment_config()

        assert "num_replicas" in deployment_config
        assert isinstance(deployment_config["num_replicas"], int)

    def test_to_deployment_config_autoscaling(self):
        """Test deployment config with autoscaling enabled."""
        config = RayConfig()
        config.autoscaling_enabled = True
        deployment_config = config.to_deployment_config()

        assert "autoscaling_config" in deployment_config
        assert "min_replicas" in deployment_config["autoscaling_config"]
        assert "max_replicas" in deployment_config["autoscaling_config"]

    def test_to_deployment_config_resources(self):
        """Test deployment config with resource limits."""
        config = RayConfig()
        config.num_cpus = 2.0
        config.num_gpus = 1.0
        deployment_config = config.to_deployment_config()

        assert "ray_actor_options" in deployment_config
        assert deployment_config["ray_actor_options"]["num_cpus"] == 2.0
        assert deployment_config["ray_actor_options"]["num_gpus"] == 1.0

    def test_get_ray_config_singleton(self):
        """Test that get_ray_config returns singleton."""
        config1 = get_ray_config()
        config2 = get_ray_config()
        assert config1 is config2

    def test_set_ray_config(self):
        """Test setting custom configuration."""
        custom_config = RayConfig()
        custom_config.address = "custom"
        set_ray_config(custom_config)

        config = get_ray_config()
        assert config.address == "custom"

        # Reset for other tests
        set_ray_config(RayConfig())


class TestComputeBackend:
    """Tests for compute backend selection."""

    def setup_method(self):
        """Reset backend cache before each test."""
        reset_backend()

    def teardown_method(self):
        """Reset backend cache after each test."""
        reset_backend()

    def test_backend_enum_values(self):
        """Test that all backend values exist."""
        assert ComputeBackend.CELERY.value == "celery"
        assert ComputeBackend.RAY.value == "ray"
        assert ComputeBackend.LOCAL.value == "local"

    def test_select_backend_explicit(self):
        """Test explicit backend selection."""
        with patch.dict(os.environ, {"TITAN_COMPUTE_BACKEND": "local"}, clear=False):
            reset_backend()
            backend = select_backend()
            assert backend == ComputeBackend.LOCAL

    def test_select_backend_default_local(self):
        """Test default local backend selection."""
        with patch.dict(os.environ, {
            "TITAN_COMPUTE_BACKEND": "",
            "TITAN_USE_RAY": "",
            "TITAN_USE_CELERY": "",
        }, clear=False):
            reset_backend()
            backend = select_backend()
            assert backend == ComputeBackend.LOCAL

    def test_is_distributed_local(self):
        """Test is_distributed returns False for local."""
        with patch.dict(os.environ, {"TITAN_COMPUTE_BACKEND": "local"}, clear=False):
            reset_backend()
            assert is_distributed() is False

    def test_is_distributed_celery(self):
        """Test is_distributed returns True for Celery."""
        with patch.dict(os.environ, {"TITAN_COMPUTE_BACKEND": "celery"}, clear=False):
            reset_backend()
            assert is_distributed() is True

    def test_is_distributed_ray(self):
        """Test is_distributed returns True for Ray."""
        with patch.dict(os.environ, {"TITAN_COMPUTE_BACKEND": "ray"}, clear=False):
            reset_backend()
            assert is_distributed() is True

    def test_get_backend_cached(self):
        """Test that get_backend caches result."""
        with patch.dict(os.environ, {"TITAN_COMPUTE_BACKEND": "local"}, clear=False):
            reset_backend()
            backend1 = get_backend()
            backend2 = get_backend()
            assert backend1 is backend2

    def test_reset_backend(self):
        """Test reset_backend clears cache."""
        with patch.dict(os.environ, {"TITAN_COMPUTE_BACKEND": "local"}, clear=False):
            reset_backend()
            backend1 = get_backend()

        with patch.dict(os.environ, {"TITAN_COMPUTE_BACKEND": "celery"}, clear=False):
            reset_backend()
            backend2 = get_backend()

        assert backend1 != backend2


class TestRayBackendWhenUnavailable:
    """Tests for Ray backend when Ray is not installed."""

    def test_get_ray_backend_raises_when_unavailable(self):
        """Test that get_ray_backend raises ImportError when Ray unavailable."""
        with patch("titan.ray.RAY_AVAILABLE", False):
            from titan.ray import get_ray_backend
            with pytest.raises(ImportError, match="Ray is not installed"):
                get_ray_backend()
