"""Ray Serve integration for Titan.

This module provides optional Ray Serve integration for distributed
compute alongside Celery. Ray offers autoscaling, GPU support, and
better performance for certain workloads.

Components:
- config: Ray configuration and environment detection
- serve: Ray Serve deployments for inquiry and batch processing
- actors: Ray actors for orchestration

Usage:
    from titan.ray import get_ray_backend, is_ray_available

    if is_ray_available():
        backend = get_ray_backend()
        result = await backend.run_inquiry(session_id, stage)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

# Check for Ray availability
RAY_AVAILABLE = False
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    pass


def is_ray_available() -> bool:
    """Check if Ray is available."""
    return RAY_AVAILABLE


def get_ray_backend():
    """Get the Ray backend if available.

    Returns:
        RayBackend instance if Ray is available

    Raises:
        ImportError: If Ray is not installed
    """
    if not RAY_AVAILABLE:
        raise ImportError(
            "Ray is not installed. Install with: pip install 'agentic-titan[ray]'"
        )
    from titan.ray.serve import RayBackend
    return RayBackend()


__all__ = [
    "is_ray_available",
    "get_ray_backend",
    "RAY_AVAILABLE",
]
