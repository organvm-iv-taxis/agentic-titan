"""Ray configuration and environment detection.

Provides configuration for Ray clusters and deployments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RayConfig:
    """Configuration for Ray Serve deployments."""

    # Cluster configuration
    address: str = field(default_factory=lambda: os.getenv("RAY_ADDRESS", "auto"))
    namespace: str = field(default_factory=lambda: os.getenv("RAY_NAMESPACE", "titan"))

    # Serve configuration
    http_host: str = field(default_factory=lambda: os.getenv("RAY_HTTP_HOST", "0.0.0.0"))
    http_port: int = field(default_factory=lambda: int(os.getenv("RAY_HTTP_PORT", "8000")))

    # Deployment configuration
    num_replicas: int = field(default_factory=lambda: int(os.getenv("RAY_REPLICAS", "2")))
    max_replicas: int = field(default_factory=lambda: int(os.getenv("RAY_MAX_REPLICAS", "10")))
    min_replicas: int = field(default_factory=lambda: int(os.getenv("RAY_MIN_REPLICAS", "1")))

    # Resource configuration
    num_cpus: float = field(default_factory=lambda: float(os.getenv("RAY_NUM_CPUS", "1.0")))
    num_gpus: float = field(default_factory=lambda: float(os.getenv("RAY_NUM_GPUS", "0.0")))
    memory: int = field(default_factory=lambda: int(os.getenv("RAY_MEMORY", "0")))  # 0 = auto

    # Autoscaling configuration
    autoscaling_enabled: bool = field(default_factory=lambda: os.getenv("RAY_AUTOSCALING", "true").lower() == "true")
    target_num_ongoing_requests: int = field(default_factory=lambda: int(os.getenv("RAY_TARGET_REQUESTS", "5")))
    upscale_delay_s: float = field(default_factory=lambda: float(os.getenv("RAY_UPSCALE_DELAY", "30.0")))
    downscale_delay_s: float = field(default_factory=lambda: float(os.getenv("RAY_DOWNSCALE_DELAY", "300.0")))

    # Health check configuration
    health_check_period_s: float = field(default_factory=lambda: float(os.getenv("RAY_HEALTH_CHECK_PERIOD", "10.0")))
    health_check_timeout_s: float = field(default_factory=lambda: float(os.getenv("RAY_HEALTH_CHECK_TIMEOUT", "30.0")))

    def to_deployment_config(self) -> dict[str, Any]:
        """Convert to Ray Serve deployment configuration."""
        config: dict[str, Any] = {
            "num_replicas": self.num_replicas,
        }

        if self.autoscaling_enabled:
            config["autoscaling_config"] = {
                "min_replicas": self.min_replicas,
                "max_replicas": self.max_replicas,
                "target_num_ongoing_requests_per_replica": self.target_num_ongoing_requests,
                "upscale_delay_s": self.upscale_delay_s,
                "downscale_delay_s": self.downscale_delay_s,
            }

        if self.num_cpus > 0:
            config["ray_actor_options"] = config.get("ray_actor_options", {})
            config["ray_actor_options"]["num_cpus"] = self.num_cpus

        if self.num_gpus > 0:
            config["ray_actor_options"] = config.get("ray_actor_options", {})
            config["ray_actor_options"]["num_gpus"] = self.num_gpus

        if self.memory > 0:
            config["ray_actor_options"] = config.get("ray_actor_options", {})
            config["ray_actor_options"]["memory"] = self.memory

        config["health_check_period_s"] = self.health_check_period_s
        config["health_check_timeout_s"] = self.health_check_timeout_s

        return config


# Global configuration instance
_config: RayConfig | None = None


def get_ray_config() -> RayConfig:
    """Get the global Ray configuration."""
    global _config
    if _config is None:
        _config = RayConfig()
    return _config


def set_ray_config(config: RayConfig) -> None:
    """Set the global Ray configuration."""
    global _config
    _config = config
