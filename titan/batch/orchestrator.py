"""
Titan Batch - Orchestrator

Core orchestration engine for batch research pipeline processing.
Manages batch job lifecycle, session queuing, progress tracking,
and coordination with distributed Celery workers.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable
from uuid import UUID, uuid4

from titan.batch.models import (
    BatchJob,
    BatchProgress,
    BatchStatus,
    BatchSubmitRequest,
    QueuedSession,
    SessionQueueStatus,
)
from titan.metrics import get_metrics
from titan.workflows.inquiry_config import get_workflow, list_workflows

if TYPE_CHECKING:
    from hive.memory import HiveMind
    from titan.costs.budget import BudgetTracker
    from titan.persistence.postgres import PostgresClient

logger = logging.getLogger("titan.batch.orchestrator")


class BatchOrchestrator:
    """
    Orchestrates batch research pipeline execution.

    Manages the full lifecycle of batch jobs:
    - Batch submission and validation
    - Session queuing to Celery workers
    - Progress monitoring and status updates
    - Cancellation and cleanup
    - Cross-session synthesis triggering

    Features:
    - Configurable concurrency limits
    - Budget enforcement
    - Automatic retry handling
    - Real-time progress streaming
    - Integration with HiveMind for state persistence
    """

    def __init__(
        self,
        hive_mind: HiveMind | None = None,
        budget_tracker: BudgetTracker | None = None,
        postgres_client: PostgresClient | None = None,
        default_max_concurrent: int = 3,
        max_retries: int = 3,
        enable_celery: bool = True,
    ) -> None:
        """
        Initialize the batch orchestrator.

        Args:
            hive_mind: Shared memory for state persistence
            budget_tracker: Budget tracking integration
            postgres_client: PostgreSQL client for persistence
            default_max_concurrent: Default max concurrent sessions
            max_retries: Maximum retries per session
            enable_celery: Whether to use Celery for distributed execution
        """
        self._hive_mind = hive_mind
        self._budget_tracker = budget_tracker
        self._postgres: PostgresClient | None = postgres_client
        self._default_max_concurrent = default_max_concurrent
        self._max_retries = max_retries
        self._enable_celery = enable_celery

        # In-memory batch storage
        self._batches: dict[UUID, BatchJob] = {}

        # Event handlers
        self._on_batch_started: list[Callable[[BatchJob], None]] = []
        self._on_batch_completed: list[Callable[[BatchJob], None]] = []
        self._on_session_started: list[Callable[[BatchJob, QueuedSession], None]] = []
        self._on_session_completed: list[Callable[[BatchJob, QueuedSession], None]] = []

        # Background tasks
        self._monitoring_tasks: dict[UUID, asyncio.Task] = {}
        self._initialized = False

        logger.info("Batch orchestrator initialized")

    async def initialize(self) -> None:
        """
        Initialize the orchestrator, loading existing batches from PostgreSQL.

        Should be called on startup to restore state.
        """
        if self._initialized:
            return

        # Try to connect to PostgreSQL if not provided
        if self._postgres is None:
            try:
                from titan.persistence.postgres import get_postgres_client
                self._postgres = get_postgres_client()
                await self._postgres.connect()
            except Exception as e:
                logger.warning(f"PostgreSQL not available, using in-memory only: {e}")
                self._postgres = None

        # Load existing batches from PostgreSQL
        if self._postgres and self._postgres.is_connected:
            await self._load_batches_from_postgres()

        self._initialized = True
        logger.info("Batch orchestrator initialized from persistence")

    async def _load_batches_from_postgres(self) -> int:
        """
        Load existing batches from PostgreSQL.

        Returns:
            Number of batches loaded.
        """
        if not self._postgres or not self._postgres.is_connected:
            return 0

        try:
            # Load active batches (not in terminal state)
            batch_rows = await self._postgres.list_batch_jobs(limit=1000)
            loaded = 0

            for row in batch_rows:
                batch_id = row["id"]
                if isinstance(batch_id, str):
                    batch_id = UUID(batch_id)

                # Skip if already in memory
                if batch_id in self._batches:
                    continue

                # Load sessions for this batch
                session_rows = await self._postgres.get_sessions_for_batch(batch_id)

                # Reconstruct BatchJob
                sessions = []
                for sess_row in session_rows:
                    sessions.append(QueuedSession(
                        id=UUID(sess_row["id"]) if isinstance(sess_row["id"], str) else sess_row["id"],
                        batch_id=batch_id,
                        topic=sess_row["topic"],
                        status=SessionQueueStatus(sess_row["status"]),
                        worker_id=sess_row.get("worker_id"),
                        celery_task_id=sess_row.get("celery_task_id"),
                        artifact_uri=sess_row.get("artifact_uri"),
                        inquiry_session_id=sess_row.get("inquiry_session_id"),
                        tokens_used=sess_row.get("tokens_used", 0),
                        cost_usd=sess_row.get("cost_usd", 0.0),
                        retry_count=sess_row.get("retry_count", 0),
                        error=sess_row.get("error"),
                        created_at=sess_row["created_at"],
                        started_at=sess_row.get("started_at"),
                        completed_at=sess_row.get("completed_at"),
                        metadata=sess_row.get("metadata", {}),
                    ))

                batch = BatchJob(
                    id=batch_id,
                    topics=row.get("topics", []),
                    workflow_name=row.get("workflow_name", "expansive"),
                    max_concurrent=row.get("max_concurrent", 3),
                    budget_limit_usd=row.get("budget_limit_usd"),
                    status=BatchStatus(row["status"]),
                    sessions=sessions,
                    synthesis_uri=row.get("synthesis_uri"),
                    created_at=row["created_at"],
                    started_at=row.get("started_at"),
                    completed_at=row.get("completed_at"),
                    error=row.get("error"),
                    user_id=row.get("user_id"),
                    metadata=row.get("metadata", {}),
                )

                self._batches[batch_id] = batch
                loaded += 1

                # Resume monitoring for active batches
                if batch.status == BatchStatus.PROCESSING:
                    self._start_monitoring(batch)

            logger.info(f"Loaded {loaded} batches from PostgreSQL")
            return loaded

        except Exception as e:
            logger.error(f"Failed to load batches from PostgreSQL: {e}")
            return 0

    # =========================================================================
    # Batch Lifecycle Management
    # =========================================================================

    async def submit_batch(
        self,
        request: BatchSubmitRequest,
    ) -> BatchJob:
        """
        Submit a new batch job.

        Creates a BatchJob from the request, validates it,
        and optionally queues sessions to Celery workers.

        Args:
            request: Batch submission request

        Returns:
            Created BatchJob

        Raises:
            ValueError: If workflow is invalid or topics are empty
        """
        # Validate workflow
        workflow = get_workflow(request.workflow)
        if not workflow:
            raise ValueError(
                f"Unknown workflow: {request.workflow}. "
                f"Available: {list_workflows()}"
            )

        # Create batch job
        batch = BatchJob(
            id=uuid4(),
            topics=request.topics,
            workflow_name=request.workflow,
            max_concurrent=request.max_concurrent or self._default_max_concurrent,
            budget_limit_usd=request.budget_limit_usd,
            user_id=request.user_id,
            metadata=request.metadata,
        )

        # Estimate cost if budget tracker available
        if self._budget_tracker and request.budget_limit_usd:
            estimated = await self._estimate_batch_cost(batch)
            batch.metadata["estimated_cost_usd"] = estimated

        # Store batch
        self._batches[batch.id] = batch

        # Persist to HiveMind
        await self._persist_batch(batch)

        # Record metrics
        metrics = get_metrics()
        metrics.batch_submitted(request.workflow)

        logger.info(
            f"Submitted batch {batch.id} with {len(batch.topics)} topics, "
            f"workflow={request.workflow}, max_concurrent={batch.max_concurrent}"
        )

        return batch

    async def start_batch(self, batch_id: UUID | str) -> BatchJob:
        """
        Start processing a batch job.

        Queues sessions to Celery workers respecting concurrency limits.

        Args:
            batch_id: Batch job ID

        Returns:
            Updated BatchJob

        Raises:
            ValueError: If batch not found or already started
        """
        batch = self.get_batch(batch_id)
        if not batch:
            raise ValueError(f"Batch not found: {batch_id}")

        if batch.status not in (BatchStatus.PENDING, BatchStatus.PAUSED):
            raise ValueError(
                f"Cannot start batch in status {batch.status.value}"
            )

        # Mark as processing
        batch.mark_started()

        # Record metrics
        metrics = get_metrics()
        metrics.batch_started(batch.workflow_name)

        # Notify handlers
        for handler in self._on_batch_started:
            try:
                handler(batch)
            except Exception as e:
                logger.warning(f"Batch started handler error: {e}")

        # Queue initial sessions
        await self._queue_pending_sessions(batch)

        # Start monitoring task
        self._start_monitoring(batch)

        # Persist state
        await self._persist_batch(batch)

        logger.info(f"Started batch {batch.id}")
        return batch

    async def cancel_batch(self, batch_id: UUID | str) -> BatchJob:
        """
        Cancel a batch job.

        Cancels pending sessions and revokes queued Celery tasks.

        Args:
            batch_id: Batch job ID

        Returns:
            Updated BatchJob

        Raises:
            ValueError: If batch not found
        """
        batch = self.get_batch(batch_id)
        if not batch:
            raise ValueError(f"Batch not found: {batch_id}")

        if batch.status in (
            BatchStatus.COMPLETED,
            BatchStatus.CANCELLED,
            BatchStatus.FAILED,
        ):
            raise ValueError(
                f"Cannot cancel batch in status {batch.status.value}"
            )

        # Stop monitoring
        self._stop_monitoring(batch.id)

        # Revoke Celery tasks
        if self._enable_celery:
            await self._revoke_batch_tasks(batch)

        # Mark as cancelled
        batch.mark_cancelled()

        # Persist state
        await self._persist_batch(batch)

        logger.info(f"Cancelled batch {batch.id}")
        return batch

    async def pause_batch(self, batch_id: UUID | str) -> BatchJob:
        """
        Pause a batch job.

        Stops queuing new sessions but allows running sessions to complete.

        Args:
            batch_id: Batch job ID

        Returns:
            Updated BatchJob
        """
        batch = self.get_batch(batch_id)
        if not batch:
            raise ValueError(f"Batch not found: {batch_id}")

        if batch.status != BatchStatus.PROCESSING:
            raise ValueError(
                f"Cannot pause batch in status {batch.status.value}"
            )

        batch.status = BatchStatus.PAUSED
        self._stop_monitoring(batch.id)

        await self._persist_batch(batch)
        logger.info(f"Paused batch {batch.id}")
        return batch

    # =========================================================================
    # Batch Queries
    # =========================================================================

    def get_batch(self, batch_id: UUID | str) -> BatchJob | None:
        """Get a batch by ID."""
        target = UUID(batch_id) if isinstance(batch_id, str) else batch_id
        return self._batches.get(target)

    def list_batches(
        self,
        status: BatchStatus | None = None,
        user_id: str | None = None,
    ) -> list[BatchJob]:
        """
        List batches with optional filtering.

        Args:
            status: Filter by status
            user_id: Filter by user ID

        Returns:
            List of matching batches
        """
        batches = list(self._batches.values())

        if status:
            batches = [b for b in batches if b.status == status]

        if user_id:
            batches = [b for b in batches if b.user_id == user_id]

        # Sort by creation time (newest first)
        batches.sort(key=lambda b: b.created_at, reverse=True)
        return batches

    def get_batch_progress(self, batch_id: UUID | str) -> BatchProgress | None:
        """Get progress for a batch."""
        batch = self.get_batch(batch_id)
        return batch.progress if batch else None

    # =========================================================================
    # Session Management
    # =========================================================================

    async def handle_session_started(
        self,
        batch_id: UUID | str,
        session_id: UUID | str,
        worker_id: str,
    ) -> None:
        """
        Handle session start notification from worker.

        Args:
            batch_id: Batch job ID
            session_id: Session ID
            worker_id: Worker processing the session
        """
        batch = self.get_batch(batch_id)
        if not batch:
            logger.warning(f"Batch not found for session start: {batch_id}")
            return

        session = batch.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return

        session.mark_running(worker_id)

        # Record metrics
        metrics = get_metrics()
        metrics.batch_session_started()

        # Notify handlers
        for handler in self._on_session_started:
            try:
                handler(batch, session)
            except Exception as e:
                logger.warning(f"Session started handler error: {e}")

        await self._persist_batch(batch)
        logger.debug(f"Session {session_id} started on worker {worker_id}")

    async def handle_session_completed(
        self,
        batch_id: UUID | str,
        session_id: UUID | str,
        artifact_uri: str,
        tokens_used: int,
        cost_usd: float,
        inquiry_session_id: str | None = None,
    ) -> None:
        """
        Handle session completion notification from worker.

        Args:
            batch_id: Batch job ID
            session_id: Session ID
            artifact_uri: URI of stored artifact
            tokens_used: Total tokens consumed
            cost_usd: Estimated cost
            inquiry_session_id: ID of the underlying InquirySession
        """
        batch = self.get_batch(batch_id)
        if not batch:
            logger.warning(f"Batch not found for session completion: {batch_id}")
            return

        session = batch.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return

        session.mark_completed(artifact_uri, tokens_used, cost_usd)
        if inquiry_session_id:
            session.inquiry_session_id = inquiry_session_id

        # Record metrics
        metrics = get_metrics()
        metrics.batch_session_completed("completed", tokens_used, cost_usd)

        # Notify handlers
        for handler in self._on_session_completed:
            try:
                handler(batch, session)
            except Exception as e:
                logger.warning(f"Session completed handler error: {e}")

        # Check if batch should complete
        await self._check_batch_completion(batch)

        # Queue more sessions if under limit
        if batch.status == BatchStatus.PROCESSING:
            await self._queue_pending_sessions(batch)

        await self._persist_batch(batch)
        logger.info(
            f"Session {session_id} completed: {tokens_used} tokens, ${cost_usd:.4f}"
        )

    async def handle_session_failed(
        self,
        batch_id: UUID | str,
        session_id: UUID | str,
        error: str,
    ) -> None:
        """
        Handle session failure notification from worker.

        Implements retry logic with exponential backoff.

        Args:
            batch_id: Batch job ID
            session_id: Session ID
            error: Error message
        """
        batch = self.get_batch(batch_id)
        if not batch:
            logger.warning(f"Batch not found for session failure: {batch_id}")
            return

        session = batch.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return

        logger.warning(
            f"Session {session_id} failed (attempt {session.retry_count + 1}): {error}"
        )

        # Check if should retry
        if session.retry_count < self._max_retries:
            session.mark_retrying()
            # Re-queue with backoff
            await self._requeue_session(batch, session)
        else:
            session.mark_failed(error)
            # Record failed session metrics
            metrics = get_metrics()
            metrics.batch_session_completed("failed", session.tokens_used, session.cost_usd)
            await self._check_batch_completion(batch)

        await self._persist_batch(batch)

    # =========================================================================
    # Progress Streaming
    # =========================================================================

    async def stream_progress(
        self,
        batch_id: UUID | str,
        poll_interval: float = 1.0,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream batch progress updates.

        Yields progress events until batch completes.

        Args:
            batch_id: Batch job ID
            poll_interval: Seconds between updates

        Yields:
            Progress event dictionaries
        """
        batch = self.get_batch(batch_id)
        if not batch:
            yield {"type": "error", "error": f"Batch not found: {batch_id}"}
            return

        yield {
            "type": "batch_info",
            "batch_id": str(batch.id),
            "topics": batch.topics,
            "total_sessions": len(batch.sessions),
            "status": batch.status.value,
        }

        last_progress = None

        while True:
            batch = self.get_batch(batch_id)
            if not batch:
                yield {"type": "error", "error": "Batch no longer exists"}
                return

            progress = batch.progress

            # Only yield if progress changed
            progress_dict = progress.to_dict()
            if progress_dict != last_progress:
                yield {
                    "type": "progress",
                    "batch_id": str(batch.id),
                    "status": batch.status.value,
                    **progress_dict,
                    "total_tokens": batch.total_tokens,
                    "total_cost_usd": batch.total_cost_usd,
                }
                last_progress = progress_dict

            # Check for completion
            if batch.status in (
                BatchStatus.COMPLETED,
                BatchStatus.FAILED,
                BatchStatus.CANCELLED,
                BatchStatus.PARTIALLY_COMPLETED,
            ):
                yield {
                    "type": "batch_completed",
                    "batch_id": str(batch.id),
                    "status": batch.status.value,
                    "synthesis_uri": batch.synthesis_uri,
                    "total_tokens": batch.total_tokens,
                    "total_cost_usd": batch.total_cost_usd,
                }
                return

            await asyncio.sleep(poll_interval)

    # =========================================================================
    # Event Handler Registration
    # =========================================================================

    def on_batch_started(
        self,
        handler: Callable[[BatchJob], None],
    ) -> None:
        """Register handler for batch start events."""
        self._on_batch_started.append(handler)

    def on_batch_completed(
        self,
        handler: Callable[[BatchJob], None],
    ) -> None:
        """Register handler for batch completion events."""
        self._on_batch_completed.append(handler)

    def on_session_started(
        self,
        handler: Callable[[BatchJob, QueuedSession], None],
    ) -> None:
        """Register handler for session start events."""
        self._on_session_started.append(handler)

    def on_session_completed(
        self,
        handler: Callable[[BatchJob, QueuedSession], None],
    ) -> None:
        """Register handler for session completion events."""
        self._on_session_completed.append(handler)

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _queue_pending_sessions(self, batch: BatchJob) -> int:
        """
        Queue pending sessions respecting concurrency limit.

        Returns number of sessions queued.
        """
        # Check budget
        if batch.is_over_budget:
            logger.warning(f"Batch {batch.id} over budget, not queuing more sessions")
            return 0

        # Calculate how many to queue
        active = len(batch.get_active_sessions())
        slots = batch.max_concurrent - active

        if slots <= 0:
            return 0

        pending = batch.get_pending_sessions(limit=slots)
        queued = 0

        for session in pending:
            try:
                await self._queue_session(batch, session)
                queued += 1
            except Exception as e:
                logger.error(f"Failed to queue session {session.id}: {e}")
                session.mark_failed(str(e))

        if queued > 0:
            logger.debug(f"Queued {queued} sessions for batch {batch.id}")

        return queued

    async def _queue_session(
        self,
        batch: BatchJob,
        session: QueuedSession,
    ) -> None:
        """Queue a single session to Celery."""
        if self._enable_celery:
            from titan.batch.worker import run_inquiry_session_task

            # Submit to Celery
            task = run_inquiry_session_task.delay(
                session_data={
                    "session_id": str(session.id),
                    "batch_id": str(batch.id),
                    "topic": session.topic,
                    "workflow_name": batch.workflow_name,
                    "budget_remaining": batch.remaining_budget_usd,
                    "metadata": session.metadata,
                },
            )
            session.mark_queued(task.id)
        else:
            # Local execution fallback
            session.mark_queued(f"local-{uuid4().hex[:8]}")

    async def _requeue_session(
        self,
        batch: BatchJob,
        session: QueuedSession,
    ) -> None:
        """Requeue a session for retry with exponential backoff."""
        # Calculate backoff delay
        backoff = min(30, 2 ** session.retry_count)

        logger.info(
            f"Requeuing session {session.id} after {backoff}s "
            f"(attempt {session.retry_count})"
        )

        await asyncio.sleep(backoff)
        await self._queue_session(batch, session)

    async def _revoke_batch_tasks(self, batch: BatchJob) -> None:
        """Revoke all Celery tasks for a batch."""
        if not self._enable_celery:
            return

        try:
            from titan.batch.celery_app import celery_app

            for session in batch.sessions:
                if session.celery_task_id and not session.is_terminal:
                    celery_app.control.revoke(
                        session.celery_task_id,
                        terminate=True,
                    )
                    logger.debug(f"Revoked task {session.celery_task_id}")
        except Exception as e:
            logger.error(f"Error revoking tasks: {e}")

    async def _check_batch_completion(self, batch: BatchJob) -> None:
        """Check if batch should be marked complete."""
        progress = batch.progress

        if progress.is_complete:
            batch.mark_completed()

            # Calculate duration
            duration_seconds = 0.0
            if batch.started_at and batch.completed_at:
                duration_seconds = (batch.completed_at - batch.started_at).total_seconds()

            # Record metrics
            metrics = get_metrics()
            metrics.batch_completed(
                workflow=batch.workflow_name,
                status=batch.status.value,
                duration_seconds=duration_seconds,
                tokens=batch.total_tokens,
                cost_usd=batch.total_cost_usd,
            )

            # Notify handlers
            for handler in self._on_batch_completed:
                try:
                    handler(batch)
                except Exception as e:
                    logger.warning(f"Batch completed handler error: {e}")

            # Stop monitoring
            self._stop_monitoring(batch.id)

            logger.info(
                f"Batch {batch.id} completed: "
                f"{progress.completed} succeeded, {progress.failed} failed"
            )

    async def _estimate_batch_cost(self, batch: BatchJob) -> float:
        """Estimate total cost for a batch."""
        if not self._budget_tracker:
            return 0.0

        # Rough estimate: 4000 tokens per stage * 6 stages per session
        workflow = get_workflow(batch.workflow_name)
        stages = len(workflow.stages) if workflow else 6
        tokens_per_session = 4000 * stages

        total_tokens = tokens_per_session * len(batch.topics)
        return await self._budget_tracker.estimate_cost(
            total_tokens // 2,  # input tokens
            total_tokens // 2,  # output tokens
            "claude-3-5-sonnet-20241022",
        )

    async def _persist_batch(self, batch: BatchJob) -> None:
        """Persist batch state to PostgreSQL and HiveMind."""
        # Persist to PostgreSQL
        if self._postgres and self._postgres.is_connected:
            await self._persist_batch_to_postgres(batch)

        # Also persist to HiveMind for quick access
        if self._hive_mind:
            try:
                await self._hive_mind.set(
                    f"batch:{batch.id}",
                    batch.to_dict(),
                    ttl=3600 * 24 * 7,  # 7 days
                )
            except Exception as e:
                logger.warning(f"Failed to persist batch state to HiveMind: {e}")

    async def _persist_batch_to_postgres(self, batch: BatchJob) -> None:
        """Persist batch and sessions to PostgreSQL."""
        if not self._postgres or not self._postgres.is_connected:
            return

        try:
            # Check if batch exists
            existing = await self._postgres.get_batch_job(batch.id)

            if existing:
                # Update batch
                await self._postgres.update_batch_job(
                    batch.id,
                    {
                        "status": batch.status.value,
                        "synthesis_uri": batch.synthesis_uri,
                        "started_at": batch.started_at,
                        "completed_at": batch.completed_at,
                        "error": batch.error,
                        "metadata": batch.metadata,
                    },
                )
            else:
                # Insert batch
                await self._postgres.insert_batch_job(
                    batch_id=batch.id,
                    topics=batch.topics,
                    workflow_name=batch.workflow_name,
                    max_concurrent=batch.max_concurrent,
                    status=batch.status.value,
                    budget_limit_usd=batch.budget_limit_usd,
                    user_id=batch.user_id,
                    metadata=batch.metadata,
                )

                # Insert sessions
                for session in batch.sessions:
                    await self._postgres.insert_queued_session(
                        session_id=session.id,
                        batch_id=batch.id,
                        topic=session.topic,
                        status=session.status.value,
                        metadata=session.metadata,
                    )

            # Update session states
            for session in batch.sessions:
                await self._postgres.update_queued_session(
                    session.id,
                    {
                        "status": session.status.value,
                        "worker_id": session.worker_id,
                        "celery_task_id": session.celery_task_id,
                        "artifact_uri": session.artifact_uri,
                        "inquiry_session_id": session.inquiry_session_id,
                        "tokens_used": session.tokens_used,
                        "cost_usd": session.cost_usd,
                        "retry_count": session.retry_count,
                        "error": session.error,
                        "started_at": session.started_at,
                        "completed_at": session.completed_at,
                    },
                )

        except Exception as e:
            logger.warning(f"Failed to persist batch to PostgreSQL: {e}")

    def _start_monitoring(self, batch: BatchJob) -> None:
        """Start background monitoring task for a batch."""
        if batch.id in self._monitoring_tasks:
            return

        task = asyncio.create_task(self._monitor_batch(batch.id))
        self._monitoring_tasks[batch.id] = task
        logger.debug(f"Started monitoring for batch {batch.id}")

    def _stop_monitoring(self, batch_id: UUID) -> None:
        """Stop monitoring task for a batch."""
        task = self._monitoring_tasks.pop(batch_id, None)
        if task and not task.done():
            task.cancel()
            logger.debug(f"Stopped monitoring for batch {batch_id}")

    async def _monitor_batch(self, batch_id: UUID) -> None:
        """
        Background task to monitor batch progress.

        Handles timeouts and stalled sessions.
        """
        try:
            while True:
                batch = self.get_batch(batch_id)
                if not batch or batch.status not in (
                    BatchStatus.PROCESSING,
                    BatchStatus.QUEUED,
                ):
                    break

                # Check for stalled sessions (running > 30 min)
                now = datetime.now()
                for session in batch.get_active_sessions():
                    if session.started_at:
                        elapsed = (now - session.started_at).total_seconds()
                        if elapsed > 1800:  # 30 minutes
                            logger.warning(
                                f"Session {session.id} stalled after {elapsed}s"
                            )
                            await self.handle_session_failed(
                                batch_id,
                                session.id,
                                "Session timed out after 30 minutes",
                            )

                # Queue more sessions if needed
                await self._queue_pending_sessions(batch)

                await asyncio.sleep(10)  # Check every 10 seconds

        except asyncio.CancelledError:
            logger.debug(f"Monitoring cancelled for batch {batch_id}")
        except Exception as e:
            logger.error(f"Error monitoring batch {batch_id}: {e}")

    # =========================================================================
    # Stalled Batch Detection and Recovery
    # =========================================================================

    async def get_stalled_batches(
        self,
        threshold_minutes: int = 30,
    ) -> list[UUID]:
        """
        Find batches with no progress for longer than threshold.

        A batch is considered stalled if:
        - It's in RUNNING or PAUSED status
        - No session activity for threshold_minutes
        - Has sessions that should be progressing

        Args:
            threshold_minutes: Minutes without activity to consider stalled

        Returns:
            List of stalled batch IDs
        """
        from datetime import timedelta

        stalled = []
        cutoff = datetime.now() - timedelta(minutes=threshold_minutes)

        for batch_id, batch in self._batches.items():
            if batch.status not in (BatchStatus.PROCESSING, BatchStatus.PAUSED):
                continue

            # Check last activity time
            last_activity = batch.started_at or batch.created_at

            # Find most recent session activity
            for session in batch.sessions:
                if session.completed_at and session.completed_at > last_activity:
                    last_activity = session.completed_at
                if session.started_at and session.started_at > last_activity:
                    last_activity = session.started_at

            # Also check updated_at if available
            if hasattr(batch, "updated_at") and batch.updated_at:
                if batch.updated_at > last_activity:
                    last_activity = batch.updated_at

            if last_activity < cutoff:
                # Verify there are sessions that should be progressing
                active_sessions = batch.get_active_sessions()
                pending_sessions = batch.get_pending_sessions()

                if active_sessions or pending_sessions:
                    # Has work to do but no progress
                    stalled.append(batch_id)
                    logger.debug(
                        f"Batch {batch_id} stalled: last activity {last_activity}, "
                        f"{len(active_sessions)} active, {len(pending_sessions)} pending"
                    )

        return stalled

    async def recover_stalled_batch(
        self,
        batch_id: UUID | str,
        strategy: str = "retry",
    ) -> bool:
        """
        Attempt to recover a stalled batch.

        Recovery strategies:
        - retry: Reset stalled sessions and re-queue them
        - skip: Mark stalled sessions as failed and continue
        - fail: Mark entire batch as failed

        Args:
            batch_id: Batch to recover
            strategy: Recovery strategy ("retry", "skip", "fail")

        Returns:
            True if recovery was successful
        """
        target = UUID(batch_id) if isinstance(batch_id, str) else batch_id
        batch = self._batches.get(target)

        if not batch:
            logger.warning(f"Batch not found for recovery: {target}")
            return False

        logger.info(f"Attempting recovery of batch {target} with strategy: {strategy}")

        if strategy == "retry":
            return await self._recover_retry(batch)
        elif strategy == "skip":
            return await self._recover_skip(batch)
        elif strategy == "fail":
            return await self._recover_fail(batch)
        else:
            logger.warning(f"Unknown recovery strategy: {strategy}")
            return False

    async def _recover_retry(self, batch: BatchJob) -> bool:
        """Reset stalled sessions and re-queue them."""
        recovered = 0

        for session in batch.sessions:
            if session.status == SessionQueueStatus.RUNNING:
                # Session stuck running - mark for retry
                if session.retry_count < self._max_retries:
                    session.status = SessionQueueStatus.PENDING
                    session.retry_count += 1
                    session.error = "Recovered from stalled state"
                    session.worker_id = None
                    session.celery_task_id = None
                    recovered += 1
                    logger.info(
                        f"Reset stalled session {session.id} for retry "
                        f"(attempt {session.retry_count})"
                    )
                else:
                    # Exceeded max retries
                    session.status = SessionQueueStatus.FAILED
                    session.error = f"Exceeded max retries ({self._max_retries})"
                    session.completed_at = datetime.now()
                    logger.warning(f"Session {session.id} exceeded max retries")

        if recovered > 0:
            # Re-queue pending sessions
            await self._queue_pending_sessions(batch)
            await self._persist_batch(batch)
            logger.info(f"Recovered {recovered} sessions for batch {batch.id}")

        return recovered > 0

    async def _recover_skip(self, batch: BatchJob) -> bool:
        """Mark stalled sessions as failed and continue."""
        skipped = 0

        for session in batch.sessions:
            if session.status == SessionQueueStatus.RUNNING:
                session.status = SessionQueueStatus.FAILED
                session.error = "Skipped - stalled without progress"
                session.completed_at = datetime.now()
                skipped += 1
                logger.info(f"Skipped stalled session {session.id}")

        # Check if batch should complete
        await self._check_batch_completion(batch)
        await self._persist_batch(batch)

        return skipped > 0

    async def _recover_fail(self, batch: BatchJob) -> bool:
        """Mark entire batch as failed."""
        # Cancel all non-terminal sessions
        for session in batch.sessions:
            if not session.is_terminal:
                session.status = SessionQueueStatus.CANCELLED
                session.error = "Batch recovery failed"
                session.completed_at = datetime.now()

        batch.status = BatchStatus.FAILED
        batch.error = "Recovery failed - batch marked as failed"
        batch.completed_at = datetime.now()

        self._stop_monitoring(batch.id)
        await self._persist_batch(batch)

        logger.info(f"Batch {batch.id} marked as failed during recovery")
        return True


# =============================================================================
# Factory Functions
# =============================================================================

_default_orchestrator: BatchOrchestrator | None = None


def get_batch_orchestrator() -> BatchOrchestrator:
    """Get the default batch orchestrator instance."""
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = BatchOrchestrator()
    return _default_orchestrator


async def get_initialized_batch_orchestrator() -> BatchOrchestrator:
    """Get the default batch orchestrator, initialized with PostgreSQL."""
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = BatchOrchestrator()
    if not _default_orchestrator._initialized:
        await _default_orchestrator.initialize()
    return _default_orchestrator


def set_batch_orchestrator(orchestrator: BatchOrchestrator) -> None:
    """Set the default batch orchestrator instance."""
    global _default_orchestrator
    _default_orchestrator = orchestrator
