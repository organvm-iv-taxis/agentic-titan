"""
Titan Batch - Celery Configuration

Celery broker and result backend settings for distributed task execution.
Includes fault tolerance configuration for reliable job processing.
"""

from __future__ import annotations

import os


# =============================================================================
# Broker Configuration
# =============================================================================

# Redis broker URL - used for task queue
CELERY_BROKER_URL = os.getenv(
    "CELERY_BROKER_URL",
    "redis://localhost:6379/1",
)

# Redis result backend - used for storing task results
CELERY_RESULT_BACKEND = os.getenv(
    "CELERY_RESULT_BACKEND",
    "redis://localhost:6379/2",
)

# Use JSON for task serialization
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_ACCEPT_CONTENT = ["json"]

# Use UTC for task timestamps
CELERY_ENABLE_UTC = True
CELERY_TIMEZONE = "UTC"


# =============================================================================
# Fault Tolerance Configuration
# =============================================================================

# Acknowledge tasks after completion (not before)
# This ensures tasks are re-delivered if worker crashes
CELERY_TASK_ACKS_LATE = True

# Reject and requeue tasks if worker process is lost
CELERY_TASK_REJECT_ON_WORKER_LOST = True

# Hard time limit for tasks (30 minutes)
# Task will be terminated if exceeds this
CELERY_TASK_TIME_LIMIT = 1800

# Soft time limit (25 minutes)
# Raises SoftTimeLimitExceeded to allow graceful shutdown
CELERY_TASK_SOFT_TIME_LIMIT = 1500

# Default retry parameters
CELERY_TASK_MAX_RETRIES = 3
CELERY_TASK_DEFAULT_RETRY_DELAY = 60  # 1 minute base delay

# Exponential backoff for retries
CELERY_RETRY_BACKOFF = True
CELERY_RETRY_BACKOFF_MAX = 600  # Max 10 minutes between retries

# Prevent duplicate task execution
CELERY_TASK_TRACK_STARTED = True

# Prefetch multiplier - how many tasks to prefetch per worker
# Lower value = better for long-running tasks
CELERY_WORKER_PREFETCH_MULTIPLIER = 1

# Disable rate limits by default (batch jobs are already rate-limited)
CELERY_WORKER_DISABLE_RATE_LIMITS = True


# =============================================================================
# Queue Configuration
# =============================================================================

# Default queue for inquiry tasks
CELERY_TASK_DEFAULT_QUEUE = "titan.batch"

# Task routing
CELERY_TASK_ROUTES = {
    "titan.batch.worker.run_inquiry_session_task": {
        "queue": "titan.batch.inquiry",
    },
    "titan.batch.worker.synthesize_batch_task": {
        "queue": "titan.batch.synthesis",
    },
    "titan.batch.worker.store_artifact_task": {
        "queue": "titan.batch.artifacts",
    },
}

# Queue configuration
CELERY_TASK_QUEUES = {
    "titan.batch": {
        "exchange": "titan.batch",
        "routing_key": "titan.batch",
    },
    "titan.batch.inquiry": {
        "exchange": "titan.batch",
        "routing_key": "titan.batch.inquiry",
    },
    "titan.batch.synthesis": {
        "exchange": "titan.batch",
        "routing_key": "titan.batch.synthesis",
    },
    "titan.batch.artifacts": {
        "exchange": "titan.batch",
        "routing_key": "titan.batch.artifacts",
    },
}


# =============================================================================
# Result Configuration
# =============================================================================

# Keep task results for 24 hours
CELERY_RESULT_EXPIRES = 86400

# Compress results larger than 10KB
CELERY_RESULT_COMPRESSION = "gzip"

# Store full task state in result backend
CELERY_TASK_IGNORE_RESULT = False
CELERY_TASK_STORE_ERRORS_EVEN_IF_IGNORED = True


# =============================================================================
# Worker Configuration
# =============================================================================

# Default concurrency (will be overridden by --concurrency flag)
CELERY_WORKER_CONCURRENCY = int(os.getenv("CELERY_WORKER_CONCURRENCY", "2"))

# Worker log level
CELERY_WORKER_LOG_LEVEL = os.getenv("CELERY_WORKER_LOG_LEVEL", "INFO")

# Send task events for monitoring (Flower)
CELERY_WORKER_SEND_TASK_EVENTS = True
CELERY_TASK_SEND_SENT_EVENT = True


# =============================================================================
# Beat Schedule (Periodic Tasks)
# =============================================================================

CELERY_BEAT_SCHEDULE = {
    # Clean up old results every hour
    "cleanup-old-results": {
        "task": "titan.batch.worker.cleanup_old_results_task",
        "schedule": 3600.0,  # Every hour
    },
    # Check for stalled batches every 5 minutes
    "check-stalled-batches": {
        "task": "titan.batch.worker.check_stalled_batches_task",
        "schedule": 300.0,  # Every 5 minutes
    },
}


# =============================================================================
# Connection Pool Configuration
# =============================================================================

# Redis connection pool settings
CELERY_REDIS_SOCKET_TIMEOUT = 30
CELERY_REDIS_SOCKET_CONNECT_TIMEOUT = 30
CELERY_REDIS_SOCKET_KEEPALIVE = True

# Connection retry settings
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True
CELERY_BROKER_CONNECTION_MAX_RETRIES = 10


# =============================================================================
# Helper Functions
# =============================================================================

def get_celery_config() -> dict:
    """
    Get Celery configuration as a dictionary.

    Returns configuration suitable for celery_app.config_from_object().
    """
    config = {}
    for key, value in globals().items():
        if key.startswith("CELERY_"):
            # Convert CELERY_* to celery lowercase config keys
            celery_key = key.lower().replace("celery_", "")
            config[celery_key] = value
    return config


def get_broker_url() -> str:
    """Get the configured broker URL."""
    return CELERY_BROKER_URL


def get_result_backend() -> str:
    """Get the configured result backend URL."""
    return CELERY_RESULT_BACKEND
