"""
Titan Batch - Celery Application

Celery application instance configured for distributed batch processing.
Provides the main Celery app used by workers and task definitions.
"""

from __future__ import annotations

import logging
import os

from celery import Celery
from celery.signals import (
    celeryd_after_setup,
    task_failure,
    task_postrun,
    task_prerun,
    worker_ready,
    worker_shutdown,
)

from titan.batch.celery_config import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    get_celery_config,
)

logger = logging.getLogger("titan.batch.celery")

# =============================================================================
# Celery Application
# =============================================================================

celery_app = Celery(
    "titan.batch",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["titan.batch.worker"],
)

# Load configuration
celery_app.config_from_object(get_celery_config())


# =============================================================================
# Signal Handlers
# =============================================================================

@celeryd_after_setup.connect
def setup_worker_logging(sender, instance, **kwargs):
    """Configure logging after worker setup."""
    log_level = os.getenv("CELERY_WORKER_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info(f"Worker {sender} logging configured at {log_level}")


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    """Handle worker ready event."""
    worker_id = os.getenv("CELERY_WORKER_ID", sender.hostname if sender else "unknown")
    runtime_type = os.getenv("WORKER_RUNTIME_TYPE", "local")

    logger.info(
        f"Worker ready: {worker_id} (runtime: {runtime_type})"
    )

    # TODO: Register with HiveMind for coordination
    # This would publish worker capacity and metrics


@worker_shutdown.connect
def on_worker_shutdown(sender, **kwargs):
    """Handle worker shutdown event."""
    worker_id = os.getenv("CELERY_WORKER_ID", sender.hostname if sender else "unknown")
    logger.info(f"Worker shutting down: {worker_id}")

    # TODO: Deregister from HiveMind


@task_prerun.connect
def on_task_prerun(sender=None, task_id=None, task=None, args=None, kwargs=None, **kw):
    """Handle task start event."""
    logger.debug(f"Task starting: {task.name}[{task_id}]")


@task_postrun.connect
def on_task_postrun(
    sender=None,
    task_id=None,
    task=None,
    args=None,
    kwargs=None,
    retval=None,
    state=None,
    **kw,
):
    """Handle task completion event."""
    logger.debug(f"Task completed: {task.name}[{task_id}] -> {state}")


@task_failure.connect
def on_task_failure(
    sender=None,
    task_id=None,
    exception=None,
    args=None,
    kwargs=None,
    traceback=None,
    einfo=None,
    **kw,
):
    """Handle task failure event."""
    logger.error(
        f"Task failed: {sender.name}[{task_id}] - {exception}",
        exc_info=(type(exception), exception, traceback),
    )


# =============================================================================
# Celery App Configuration Methods
# =============================================================================

def configure_for_testing() -> None:
    """
    Configure Celery for testing (synchronous execution).

    Use in tests to avoid needing a running broker.
    """
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
        broker_url="memory://",
        result_backend="cache+memory://",
    )
    logger.info("Celery configured for testing (synchronous mode)")


def configure_for_production() -> None:
    """
    Configure Celery for production.

    Ensures all fault tolerance settings are active.
    """
    celery_app.conf.update(
        task_always_eager=False,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
    )
    logger.info("Celery configured for production")


def get_active_workers() -> list[dict]:
    """
    Get list of active Celery workers.

    Returns:
        List of worker info dictionaries
    """
    try:
        inspect = celery_app.control.inspect()
        active = inspect.active() or {}
        stats = inspect.stats() or {}

        workers = []
        for worker_name, worker_stats in stats.items():
            workers.append({
                "name": worker_name,
                "concurrency": worker_stats.get("pool", {}).get("max-concurrency", 0),
                "active_tasks": len(active.get(worker_name, [])),
                "broker": worker_stats.get("broker", {}),
            })

        return workers
    except Exception as e:
        logger.warning(f"Failed to get active workers: {e}")
        return []


def get_queue_lengths() -> dict[str, int]:
    """
    Get current lengths of task queues.

    Returns:
        Dictionary mapping queue names to lengths
    """
    try:
        inspect = celery_app.control.inspect()
        reserved = inspect.reserved() or {}

        # Count reserved tasks per queue
        queue_counts: dict[str, int] = {}
        for worker_name, tasks in reserved.items():
            for task in tasks:
                queue = task.get("delivery_info", {}).get("routing_key", "unknown")
                queue_counts[queue] = queue_counts.get(queue, 0) + 1

        return queue_counts
    except Exception as e:
        logger.warning(f"Failed to get queue lengths: {e}")
        return {}


def revoke_task(task_id: str, terminate: bool = False) -> bool:
    """
    Revoke a task by ID.

    Args:
        task_id: Celery task ID
        terminate: Whether to terminate running task

    Returns:
        True if revocation was sent
    """
    try:
        celery_app.control.revoke(task_id, terminate=terminate)
        logger.info(f"Revoked task {task_id} (terminate={terminate})")
        return True
    except Exception as e:
        logger.error(f"Failed to revoke task {task_id}: {e}")
        return False


# =============================================================================
# Application Entry Points
# =============================================================================

# For running with: celery -A titan.batch.celery_app worker
# The celery_app instance is already created above


if __name__ == "__main__":
    # Allow running directly for debugging
    celery_app.start()
