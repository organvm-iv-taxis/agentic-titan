"""Compute backend selector for Titan.

Selects between Celery, Ray, and Local compute backends based on
environment configuration and availability.
"""

from __future__ import annotations

import os
import logging
from enum import Enum
from typing import Any

logger = logging.getLogger("titan.ray.backend_selector")


class ComputeBackend(str, Enum):
    """Available compute backends."""

    CELERY = "celery"
    RAY = "ray"
    LOCAL = "local"


def select_backend() -> ComputeBackend:
    """Select the compute backend based on environment configuration.

    Priority:
    1. Explicit TITAN_COMPUTE_BACKEND environment variable
    2. TITAN_USE_RAY=true -> Ray
    3. TITAN_USE_CELERY=true -> Celery
    4. Default -> Local

    Returns:
        Selected ComputeBackend
    """
    # Check explicit backend setting
    explicit = os.getenv("TITAN_COMPUTE_BACKEND", "").lower()
    if explicit in ("celery", "ray", "local"):
        backend = ComputeBackend(explicit)
        logger.info(f"Using explicit compute backend: {backend.value}")
        return backend

    # Check Ray preference
    if os.getenv("TITAN_USE_RAY", "").lower() == "true":
        from titan.ray import is_ray_available
        if is_ray_available():
            logger.info("Using Ray compute backend")
            return ComputeBackend.RAY
        else:
            logger.warning("Ray requested but not available, falling back to local")

    # Check Celery preference
    if os.getenv("TITAN_USE_CELERY", "").lower() == "true":
        try:
            import celery
            logger.info("Using Celery compute backend")
            return ComputeBackend.CELERY
        except ImportError:
            logger.warning("Celery requested but not available, falling back to local")

    # Default to local
    logger.info("Using local compute backend")
    return ComputeBackend.LOCAL


def is_distributed() -> bool:
    """Check if a distributed backend is selected.

    Returns:
        True if Celery or Ray is being used
    """
    backend = select_backend()
    return backend in (ComputeBackend.CELERY, ComputeBackend.RAY)


async def run_inquiry_stage(
    session_id: str,
    stage_name: str,
) -> dict[str, Any]:
    """Run an inquiry stage using the selected backend.

    Args:
        session_id: Inquiry session ID
        stage_name: Stage to run

    Returns:
        Stage result as dictionary
    """
    backend = select_backend()

    if backend == ComputeBackend.RAY:
        from titan.ray import get_ray_backend
        ray_backend = get_ray_backend()
        return await ray_backend.run_inquiry_stage(session_id, stage_name)

    elif backend == ComputeBackend.CELERY:
        from titan.batch.worker import process_topic_task
        # Celery is synchronous, wrap in task
        task = process_topic_task.delay(session_id, {"stage": stage_name})
        result = task.get(timeout=300)
        return result

    else:  # LOCAL
        from titan.workflows.inquiry_engine import get_inquiry_engine
        engine = get_inquiry_engine()
        result = await engine.run_stage(session_id, stage_name)
        return result.to_dict()


async def start_batch(batch_id: str) -> dict[str, Any]:
    """Start a batch job using the selected backend.

    Args:
        batch_id: Batch job ID

    Returns:
        Batch status as dictionary
    """
    backend = select_backend()

    if backend == ComputeBackend.RAY:
        from titan.ray import get_ray_backend
        ray_backend = get_ray_backend()
        return await ray_backend.start_batch(batch_id)

    else:  # CELERY or LOCAL
        from titan.batch.orchestrator import get_batch_orchestrator
        orchestrator = get_batch_orchestrator()
        batch = await orchestrator.start_batch(batch_id)
        return batch.to_dict()


# Global backend cache
_selected_backend: ComputeBackend | None = None


def get_backend() -> ComputeBackend:
    """Get the cached backend selection.

    Returns:
        Cached ComputeBackend
    """
    global _selected_backend
    if _selected_backend is None:
        _selected_backend = select_backend()
    return _selected_backend


def reset_backend() -> None:
    """Reset the backend cache (for testing)."""
    global _selected_backend
    _selected_backend = None
