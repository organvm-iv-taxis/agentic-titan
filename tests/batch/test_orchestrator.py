"""
Tests for BatchOrchestrator.

Covers batch submission, monitoring, cancellation, and progress tracking.
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from titan.batch.models import (
    BatchJob,
    BatchProgress,
    BatchStatus,
    BatchSubmitRequest,
    QueuedSession,
    SessionQueueStatus,
)
from titan.batch.orchestrator import (
    BatchOrchestrator,
    get_batch_orchestrator,
    set_batch_orchestrator,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def orchestrator():
    """Create a test orchestrator with Celery disabled."""
    return BatchOrchestrator(enable_celery=False)


@pytest.fixture
def sample_request():
    """Create a sample batch submit request."""
    return BatchSubmitRequest(
        topics=["AI safety", "Prompt engineering", "Agent architectures"],
        workflow="quick",
        max_concurrent=2,
        budget_limit_usd=5.0,
        metadata={"test": True},
    )


# =============================================================================
# Model Tests
# =============================================================================

class TestBatchJob:
    """Tests for BatchJob dataclass."""

    def test_create_batch_job(self):
        """Test creating a batch job."""
        topics = ["topic1", "topic2", "topic3"]
        batch = BatchJob(topics=topics, workflow_name="quick")

        assert len(batch.sessions) == 3
        assert batch.status == BatchStatus.PENDING
        assert batch.workflow_name == "quick"
        assert batch.max_concurrent == 3

    def test_progress_calculation(self):
        """Test progress calculation."""
        batch = BatchJob(topics=["t1", "t2", "t3", "t4"])

        # Initially all pending
        progress = batch.progress
        assert progress.total == 4
        assert progress.pending == 4
        assert progress.percent_complete == 0.0

        # Mark some as completed
        batch.sessions[0].status = SessionQueueStatus.COMPLETED
        batch.sessions[1].status = SessionQueueStatus.RUNNING

        progress = batch.progress
        assert progress.completed == 1
        assert progress.running == 1
        assert progress.pending == 2
        assert progress.percent_complete == 25.0

    def test_get_pending_sessions(self):
        """Test getting pending sessions."""
        batch = BatchJob(topics=["t1", "t2", "t3"])
        batch.sessions[0].status = SessionQueueStatus.RUNNING

        pending = batch.get_pending_sessions()
        assert len(pending) == 2

        # With limit
        pending_limited = batch.get_pending_sessions(limit=1)
        assert len(pending_limited) == 1

    def test_budget_tracking(self):
        """Test budget tracking."""
        batch = BatchJob(
            topics=["t1", "t2"],
            budget_limit_usd=1.0,
        )

        assert batch.remaining_budget_usd == 1.0
        assert not batch.is_over_budget

        batch.sessions[0].cost_usd = 0.6
        batch.sessions[1].cost_usd = 0.5

        assert batch.total_cost_usd == 1.1
        assert batch.is_over_budget
        assert batch.remaining_budget_usd == 0.0

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        batch = BatchJob(
            topics=["t1", "t2"],
            workflow_name="expansive",
            budget_limit_usd=10.0,
        )
        batch.sessions[0].status = SessionQueueStatus.COMPLETED
        batch.sessions[0].tokens_used = 1000

        data = batch.to_dict()
        restored = BatchJob.from_dict(data)

        assert str(restored.id) == str(batch.id)
        assert restored.topics == batch.topics
        assert restored.sessions[0].status == SessionQueueStatus.COMPLETED
        assert restored.sessions[0].tokens_used == 1000


class TestQueuedSession:
    """Tests for QueuedSession dataclass."""

    def test_mark_state_transitions(self):
        """Test state transition methods."""
        session = QueuedSession(topic="test topic")

        # Initial state
        assert session.status == SessionQueueStatus.PENDING

        # Queue
        session.mark_queued("task-123")
        assert session.status == SessionQueueStatus.QUEUED
        assert session.celery_task_id == "task-123"

        # Start running
        session.mark_running("worker-1")
        assert session.status == SessionQueueStatus.RUNNING
        assert session.worker_id == "worker-1"
        assert session.started_at is not None

        # Complete
        session.mark_completed("s3://bucket/artifact.md", 1000, 0.05)
        assert session.status == SessionQueueStatus.COMPLETED
        assert session.artifact_uri == "s3://bucket/artifact.md"
        assert session.tokens_used == 1000
        assert session.cost_usd == 0.05
        assert session.completed_at is not None

    def test_retry_tracking(self):
        """Test retry counting."""
        session = QueuedSession(topic="test")

        session.mark_failed("First error")
        assert session.error == "First error"
        assert session.retry_count == 0

        session.mark_retrying()
        assert session.status == SessionQueueStatus.RETRYING
        assert session.retry_count == 1
        assert session.error is None

    def test_duration_calculation(self):
        """Test duration calculation."""
        session = QueuedSession(topic="test")
        assert session.duration_ms is None

        session.started_at = datetime.now()
        session.completed_at = datetime.now()
        assert session.duration_ms is not None
        assert session.duration_ms >= 0


class TestBatchProgress:
    """Tests for BatchProgress dataclass."""

    def test_percent_complete(self):
        """Test percent complete calculation."""
        progress = BatchProgress(total=10, completed=3, failed=2)
        assert progress.percent_complete == 50.0

    def test_success_rate(self):
        """Test success rate calculation."""
        progress = BatchProgress(total=10, completed=8, failed=2)
        assert progress.success_rate == 80.0

    def test_is_complete(self):
        """Test completion detection."""
        progress = BatchProgress(total=5, completed=3, failed=2)
        assert progress.is_complete

        progress2 = BatchProgress(total=5, completed=3, running=2)
        assert not progress2.is_complete


# =============================================================================
# Orchestrator Tests
# =============================================================================

class TestBatchOrchestrator:
    """Tests for BatchOrchestrator."""

    @pytest.mark.asyncio
    async def test_submit_batch(self, orchestrator, sample_request):
        """Test submitting a batch."""
        batch = await orchestrator.submit_batch(sample_request)

        assert batch.id is not None
        assert len(batch.topics) == 3
        assert len(batch.sessions) == 3
        assert batch.status == BatchStatus.PENDING
        assert batch.workflow_name == "quick"
        assert batch.budget_limit_usd == 5.0

    @pytest.mark.asyncio
    async def test_submit_invalid_workflow(self, orchestrator):
        """Test submitting with invalid workflow."""
        request = BatchSubmitRequest(
            topics=["test"],
            workflow="nonexistent",
        )

        with pytest.raises(ValueError, match="Unknown workflow"):
            await orchestrator.submit_batch(request)

    @pytest.mark.asyncio
    async def test_get_batch(self, orchestrator, sample_request):
        """Test getting a batch by ID."""
        batch = await orchestrator.submit_batch(sample_request)

        retrieved = orchestrator.get_batch(batch.id)
        assert retrieved is not None
        assert retrieved.id == batch.id

        # Test with string ID
        retrieved_str = orchestrator.get_batch(str(batch.id))
        assert retrieved_str is not None

        # Test nonexistent
        missing = orchestrator.get_batch(uuid4())
        assert missing is None

    @pytest.mark.asyncio
    async def test_list_batches(self, orchestrator):
        """Test listing batches."""
        # Submit multiple batches
        for i in range(3):
            request = BatchSubmitRequest(topics=[f"topic-{i}"])
            await orchestrator.submit_batch(request)

        batches = orchestrator.list_batches()
        assert len(batches) == 3

        # Test with status filter
        batches_pending = orchestrator.list_batches(status=BatchStatus.PENDING)
        assert len(batches_pending) == 3

        batches_completed = orchestrator.list_batches(status=BatchStatus.COMPLETED)
        assert len(batches_completed) == 0

    @pytest.mark.asyncio
    async def test_start_batch(self, orchestrator, sample_request):
        """Test starting a batch."""
        batch = await orchestrator.submit_batch(sample_request)

        # Start the batch
        started = await orchestrator.start_batch(batch.id)

        assert started.status == BatchStatus.PROCESSING
        assert started.started_at is not None

        # Sessions should be queued
        queued = [s for s in started.sessions if s.status == SessionQueueStatus.QUEUED]
        assert len(queued) == 2  # max_concurrent = 2

    @pytest.mark.asyncio
    async def test_start_already_started(self, orchestrator, sample_request):
        """Test starting an already started batch."""
        batch = await orchestrator.submit_batch(sample_request)
        await orchestrator.start_batch(batch.id)

        with pytest.raises(ValueError, match="Cannot start batch"):
            await orchestrator.start_batch(batch.id)

    @pytest.mark.asyncio
    async def test_cancel_batch(self, orchestrator, sample_request):
        """Test cancelling a batch."""
        batch = await orchestrator.submit_batch(sample_request)
        await orchestrator.start_batch(batch.id)

        cancelled = await orchestrator.cancel_batch(batch.id)

        assert cancelled.status == BatchStatus.CANCELLED
        assert cancelled.completed_at is not None

        # All non-terminal sessions should be cancelled
        for session in cancelled.sessions:
            assert session.status == SessionQueueStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_pause_batch(self, orchestrator, sample_request):
        """Test pausing a batch."""
        batch = await orchestrator.submit_batch(sample_request)
        await orchestrator.start_batch(batch.id)

        paused = await orchestrator.pause_batch(batch.id)

        assert paused.status == BatchStatus.PAUSED

    @pytest.mark.asyncio
    async def test_handle_session_started(self, orchestrator, sample_request):
        """Test handling session started notification."""
        batch = await orchestrator.submit_batch(sample_request)
        await orchestrator.start_batch(batch.id)

        session = batch.sessions[0]
        await orchestrator.handle_session_started(
            batch.id,
            session.id,
            "worker-1",
        )

        updated_session = batch.get_session(session.id)
        assert updated_session.status == SessionQueueStatus.RUNNING
        assert updated_session.worker_id == "worker-1"

    @pytest.mark.asyncio
    async def test_handle_session_completed(self, orchestrator, sample_request):
        """Test handling session completed notification."""
        batch = await orchestrator.submit_batch(sample_request)
        await orchestrator.start_batch(batch.id)

        session = batch.sessions[0]
        session.mark_running("worker-1")

        await orchestrator.handle_session_completed(
            batch.id,
            session.id,
            artifact_uri="file:///artifacts/result.md",
            tokens_used=5000,
            cost_usd=0.05,
            inquiry_session_id="inq-12345",
        )

        updated_session = batch.get_session(session.id)
        assert updated_session.status == SessionQueueStatus.COMPLETED
        assert updated_session.tokens_used == 5000
        assert updated_session.cost_usd == 0.05
        assert updated_session.inquiry_session_id == "inq-12345"

    @pytest.mark.asyncio
    async def test_handle_session_failed_with_retry(self, orchestrator, sample_request):
        """Test handling session failure with retry."""
        orchestrator._max_retries = 2
        batch = await orchestrator.submit_batch(sample_request)
        await orchestrator.start_batch(batch.id)

        session = batch.sessions[0]
        session.mark_running("worker-1")

        # First failure - should retry (mark as retrying then requeue)
        await orchestrator.handle_session_failed(
            batch.id,
            session.id,
            "Network error",
        )

        updated_session = batch.get_session(session.id)
        # Session is requeued after retry, so it could be RETRYING or QUEUED
        # The key assertion is that retry_count increased
        assert updated_session.retry_count == 1
        assert updated_session.status in (
            SessionQueueStatus.RETRYING,
            SessionQueueStatus.QUEUED,
        )

    @pytest.mark.asyncio
    async def test_handle_session_failed_max_retries(self, orchestrator, sample_request):
        """Test handling session failure with max retries exceeded."""
        orchestrator._max_retries = 1
        batch = await orchestrator.submit_batch(sample_request)
        await orchestrator.start_batch(batch.id)

        session = batch.sessions[0]
        session.retry_count = 1  # Already retried
        session.mark_running("worker-1")

        await orchestrator.handle_session_failed(
            batch.id,
            session.id,
            "Persistent error",
        )

        updated_session = batch.get_session(session.id)
        assert updated_session.status == SessionQueueStatus.FAILED
        assert updated_session.error == "Persistent error"

    @pytest.mark.asyncio
    async def test_batch_completion(self, orchestrator):
        """Test batch completion detection."""
        request = BatchSubmitRequest(topics=["t1", "t2"])
        batch = await orchestrator.submit_batch(request)
        await orchestrator.start_batch(batch.id)

        # Complete all sessions
        for session in batch.sessions:
            session.mark_running("worker-1")
            await orchestrator.handle_session_completed(
                batch.id,
                session.id,
                artifact_uri="file:///result.md",
                tokens_used=1000,
                cost_usd=0.01,
            )

        assert batch.status == BatchStatus.COMPLETED
        assert batch.completed_at is not None

    @pytest.mark.asyncio
    async def test_partial_completion(self, orchestrator):
        """Test partial completion when some sessions fail."""
        request = BatchSubmitRequest(topics=["t1", "t2"])
        batch = await orchestrator.submit_batch(request)
        await orchestrator.start_batch(batch.id)

        # Complete one, fail one
        batch.sessions[0].mark_running("worker-1")
        await orchestrator.handle_session_completed(
            batch.id,
            batch.sessions[0].id,
            artifact_uri="file:///result.md",
            tokens_used=1000,
            cost_usd=0.01,
        )

        batch.sessions[1].mark_running("worker-1")
        batch.sessions[1].retry_count = 3  # Exceed max retries
        await orchestrator.handle_session_failed(
            batch.id,
            batch.sessions[1].id,
            "Fatal error",
        )

        assert batch.status == BatchStatus.PARTIALLY_COMPLETED

    @pytest.mark.asyncio
    async def test_event_handlers(self, orchestrator, sample_request):
        """Test event handler registration and invocation."""
        started_events = []
        completed_events = []

        orchestrator.on_batch_started(lambda b: started_events.append(b))
        orchestrator.on_batch_completed(lambda b: completed_events.append(b))

        batch = await orchestrator.submit_batch(sample_request)
        await orchestrator.start_batch(batch.id)

        assert len(started_events) == 1
        assert started_events[0].id == batch.id

    @pytest.mark.asyncio
    async def test_progress_streaming(self, orchestrator):
        """Test progress streaming."""
        request = BatchSubmitRequest(topics=["t1"])
        batch = await orchestrator.submit_batch(request)
        await orchestrator.start_batch(batch.id)

        # Complete the session immediately
        batch.sessions[0].mark_running("worker-1")
        batch.sessions[0].mark_completed("file:///result.md", 1000, 0.01)
        batch.mark_completed()

        events = []
        async for event in orchestrator.stream_progress(batch.id, poll_interval=0.1):
            events.append(event)
            if event.get("type") == "batch_completed":
                break

        assert len(events) >= 2  # batch_info + batch_completed
        assert events[0]["type"] == "batch_info"
        assert events[-1]["type"] == "batch_completed"


class TestBatchSubmitRequest:
    """Tests for BatchSubmitRequest validation."""

    def test_valid_request(self):
        """Test valid request creation."""
        request = BatchSubmitRequest(
            topics=["topic1", "topic2"],
            workflow="quick",
            max_concurrent=3,
        )
        assert len(request.topics) == 2

    def test_empty_topics(self):
        """Test that empty topics raises error."""
        with pytest.raises(ValueError, match="At least one topic"):
            BatchSubmitRequest(topics=[])

    def test_invalid_max_concurrent(self):
        """Test that invalid max_concurrent raises error."""
        with pytest.raises(ValueError, match="max_concurrent"):
            BatchSubmitRequest(topics=["test"], max_concurrent=0)

    def test_invalid_budget(self):
        """Test that invalid budget raises error."""
        with pytest.raises(ValueError, match="budget_limit_usd"):
            BatchSubmitRequest(topics=["test"], budget_limit_usd=-1.0)


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for module factory functions."""

    def test_get_batch_orchestrator(self):
        """Test getting default orchestrator."""
        orchestrator = get_batch_orchestrator()
        assert orchestrator is not None

        # Should return same instance
        orchestrator2 = get_batch_orchestrator()
        assert orchestrator is orchestrator2

    def test_set_batch_orchestrator(self):
        """Test setting custom orchestrator."""
        custom = BatchOrchestrator(default_max_concurrent=5)
        set_batch_orchestrator(custom)

        retrieved = get_batch_orchestrator()
        assert retrieved._default_max_concurrent == 5

        # Reset
        set_batch_orchestrator(None)
