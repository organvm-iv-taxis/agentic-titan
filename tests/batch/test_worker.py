"""
Tests for Celery worker tasks.

Covers task execution, retry logic, and fault tolerance.
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from uuid import uuid4

from titan.batch.models import (
    BatchJob,
    BatchStatus,
    QueuedSession,
    SessionQueueStatus,
)

# Check if celery is available
try:
    import celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_session_data():
    """Create sample session data for task."""
    return {
        "session_id": str(uuid4()),
        "batch_id": str(uuid4()),
        "topic": "Test topic for research",
        "workflow_name": "quick",
        "budget_remaining": 10.0,
        "metadata": {"test": True},
    }


# =============================================================================
# Model Tests (No Celery Required)
# =============================================================================

class TestBatchModelsIntegration:
    """Integration tests for batch models."""

    def test_session_lifecycle(self):
        """Test full session lifecycle."""
        session = QueuedSession(
            topic="Test topic",
            batch_id=uuid4(),
        )

        # Queue
        session.mark_queued("task-123")
        assert session.status == SessionQueueStatus.QUEUED

        # Run
        session.mark_running("worker-1")
        assert session.status == SessionQueueStatus.RUNNING

        # Complete
        session.mark_completed("file:///result.md", 1000, 0.05)
        assert session.status == SessionQueueStatus.COMPLETED
        assert session.is_terminal

    def test_batch_job_lifecycle(self):
        """Test full batch lifecycle."""
        batch = BatchJob(
            topics=["t1", "t2", "t3"],
            workflow_name="quick",
            max_concurrent=2,
        )

        assert batch.status == BatchStatus.PENDING
        assert len(batch.sessions) == 3

        # Start
        batch.mark_started()
        assert batch.status == BatchStatus.PROCESSING
        assert batch.started_at is not None

        # Complete sessions
        for session in batch.sessions:
            session.mark_running("worker-1")
            session.mark_completed("file:///result.md", 1000, 0.05)

        # Complete batch
        batch.mark_completed()
        assert batch.status == BatchStatus.COMPLETED


# =============================================================================
# Celery Configuration Tests
# =============================================================================

class TestCeleryConfig:
    """Tests for Celery configuration."""

    def test_config_values(self):
        """Test configuration values are set correctly."""
        from titan.batch.celery_config import (
            CELERY_TASK_ACKS_LATE,
            CELERY_TASK_REJECT_ON_WORKER_LOST,
            CELERY_TASK_TIME_LIMIT,
            CELERY_TASK_MAX_RETRIES,
            get_celery_config,
        )

        assert CELERY_TASK_ACKS_LATE is True
        assert CELERY_TASK_REJECT_ON_WORKER_LOST is True
        assert CELERY_TASK_TIME_LIMIT == 1800  # 30 minutes
        assert CELERY_TASK_MAX_RETRIES == 3

    def test_get_config_dict(self):
        """Test getting config as dictionary."""
        from titan.batch.celery_config import get_celery_config

        config = get_celery_config()

        assert "task_acks_late" in config
        assert "broker_url" in config

    def test_task_routes(self):
        """Test task routing configuration."""
        from titan.batch.celery_config import CELERY_TASK_ROUTES

        assert "titan.batch.worker.run_inquiry_session_task" in CELERY_TASK_ROUTES
        assert CELERY_TASK_ROUTES["titan.batch.worker.run_inquiry_session_task"]["queue"] == "titan.batch.inquiry"

    def test_broker_urls(self):
        """Test broker URL getters."""
        from titan.batch.celery_config import get_broker_url, get_result_backend

        broker = get_broker_url()
        backend = get_result_backend()

        assert "redis" in broker
        assert "redis" in backend


# =============================================================================
# Worker Tests (Celery Required)
# =============================================================================

@pytest.mark.skipif(not CELERY_AVAILABLE, reason="Celery not installed")
class TestCeleryApp:
    """Tests for Celery app configuration."""

    def test_configure_for_testing(self):
        """Test testing configuration."""
        from titan.batch.celery_app import celery_app, configure_for_testing

        configure_for_testing()

        assert celery_app.conf.task_always_eager is True
        assert celery_app.conf.task_eager_propagates is True

    def test_get_active_workers_empty(self):
        """Test getting active workers when none connected."""
        from titan.batch.celery_app import get_active_workers

        # Should return empty list when no workers
        workers = get_active_workers()
        assert isinstance(workers, list)


@pytest.mark.skipif(not CELERY_AVAILABLE, reason="Celery not installed")
class TestCostEstimation:
    """Tests for cost estimation."""

    def test_estimate_cost_sonnet(self):
        """Test cost estimation for Sonnet model."""
        from titan.batch.worker import _estimate_cost

        results = [
            MagicMock(model_used="claude-3-5-sonnet-20241022", tokens_used=1000),
            MagicMock(model_used="claude-3-5-sonnet-20241022", tokens_used=2000),
        ]

        cost = _estimate_cost(3000, results)

        # Sonnet: $7.50 per 1M tokens
        # 3000 tokens = 0.003M tokens
        # Expected: 0.003 * 7.50 = 0.0225
        assert 0.02 < cost < 0.03

    def test_estimate_cost_haiku(self):
        """Test cost estimation for Haiku model."""
        from titan.batch.worker import _estimate_cost

        results = [
            MagicMock(model_used="claude-3-haiku-20240307", tokens_used=10000),
        ]

        cost = _estimate_cost(10000, results)

        # Haiku: $0.625 per 1M tokens
        # 10000 tokens = 0.01M tokens
        # Expected: 0.01 * 0.625 = 0.00625
        assert 0.005 < cost < 0.008

    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown model (uses default)."""
        from titan.batch.worker import _estimate_cost

        results = [
            MagicMock(model_used="unknown-model", tokens_used=1000),
        ]

        cost = _estimate_cost(1000, results)

        # Should use default Sonnet pricing
        assert cost > 0


@pytest.mark.skipif(not CELERY_AVAILABLE, reason="Celery not installed")
class TestMaintenanceTasks:
    """Tests for maintenance tasks."""

    def test_cleanup_task(self):
        """Test cleanup task."""
        from titan.batch.worker import cleanup_old_results_task

        result = cleanup_old_results_task()

        assert result["status"] == "completed"
        assert "timestamp" in result

    def test_check_stalled_task(self):
        """Test check stalled batches task."""
        from titan.batch.worker import check_stalled_batches_task

        result = check_stalled_batches_task()

        assert result["status"] == "completed"
        assert "timestamp" in result


# =============================================================================
# Scheduler Tests (No Celery Required)
# =============================================================================

class TestBatchScheduler:
    """Tests for batch scheduler."""

    def test_system_load_detection(self):
        """Test system load detection."""
        from titan.batch.scheduler import get_system_load, SystemLoad, LoadLevel

        load = get_system_load()

        assert isinstance(load, SystemLoad)
        assert 0 <= load.cpu_percent <= 100
        assert 0 <= load.memory_percent <= 100
        assert load.level in LoadLevel

    def test_load_level_classification(self):
        """Test load level classification."""
        from titan.batch.scheduler import SystemLoad, LoadLevel

        # Low load
        low = SystemLoad(cpu_percent=30, memory_percent=40)
        assert low.level == LoadLevel.LOW
        assert not low.should_offload

        # Moderate load
        moderate = SystemLoad(cpu_percent=60, memory_percent=70)
        assert moderate.level == LoadLevel.MODERATE
        assert not moderate.should_offload

        # High load
        high = SystemLoad(cpu_percent=85, memory_percent=75)
        assert high.level == LoadLevel.HIGH
        assert high.should_offload

        # Critical load
        critical = SystemLoad(cpu_percent=95, memory_percent=95)
        assert critical.level == LoadLevel.CRITICAL
        assert critical.should_offload

    def test_worker_status(self):
        """Test worker status tracking."""
        from titan.batch.scheduler import WorkerStatus

        worker = WorkerStatus(
            worker_id="worker-1",
            hostname="host1",
            runtime_type="docker",
            concurrency=4,
            active_tasks=2,
            available_slots=2,
        )

        assert worker.utilization == 50.0
        assert worker.is_available

        # Fully loaded worker
        full_worker = WorkerStatus(
            worker_id="worker-2",
            hostname="host2",
            concurrency=4,
            active_tasks=4,
            available_slots=0,
        )

        assert full_worker.utilization == 100.0
        assert not full_worker.is_available

    def test_scheduling_strategies(self):
        """Test different scheduling strategies."""
        from titan.batch.scheduler import (
            BatchScheduler,
            SchedulingStrategy,
            WorkerStatus,
        )
        from titan.batch.models import QueuedSession, BatchJob

        scheduler = BatchScheduler(strategy=SchedulingStrategy.PREFER_LOCAL)

        # Register workers
        scheduler.register_worker(WorkerStatus(
            worker_id="local-1",
            hostname="localhost",
            runtime_type="local",
            concurrency=2,
            available_slots=2,
        ))
        scheduler.register_worker(WorkerStatus(
            worker_id="docker-1",
            hostname="docker-host",
            runtime_type="docker",
            concurrency=4,
            available_slots=4,
        ))

        # Create test session and batch
        batch = BatchJob(topics=["test"])
        session = batch.sessions[0]

        decision = scheduler.schedule(session, batch)

        assert decision.runtime_type == "local"


# =============================================================================
# Artifact Store Tests (No Celery Required)
# =============================================================================

class TestArtifactStore:
    """Tests for artifact storage."""

    @pytest.mark.asyncio
    async def test_filesystem_store(self, tmp_path):
        """Test filesystem artifact store."""
        from titan.batch.artifact_store import FilesystemArtifactStore

        store = FilesystemArtifactStore(base_path=tmp_path)

        batch_id = str(uuid4())
        session_id = str(uuid4())
        content = b"# Test Markdown\n\nThis is test content."

        # Save artifact
        uri = await store.save_artifact(
            batch_id=batch_id,
            session_id=session_id,
            content=content,
            format="markdown",
            metadata={"topic": "Test topic"},
        )

        assert uri.startswith("file://")
        assert session_id in uri

        # Retrieve artifact
        retrieved = await store.get_artifact(uri)
        assert retrieved == content

        # List artifacts
        artifacts = await store.list_artifacts(batch_id)
        assert len(artifacts) == 1
        assert artifacts[0].format == "markdown"

        # Export archive
        archive = await store.export_batch_archive(batch_id)
        assert len(archive) > 0  # Non-empty ZIP

        # Delete artifact
        deleted = await store.delete_artifact(uri)
        assert deleted

        exists = await store.artifact_exists(uri)
        assert not exists

    @pytest.mark.asyncio
    async def test_artifact_store_factory(self):
        """Test artifact store factory function."""
        from titan.batch.artifact_store import get_artifact_store, FilesystemArtifactStore

        store = get_artifact_store()

        # Default should be filesystem
        assert isinstance(store, FilesystemArtifactStore)


# =============================================================================
# Synthesizer Tests (No Celery Required)
# =============================================================================

class TestBatchSynthesizer:
    """Tests for batch synthesizer."""

    def test_synthesis_result(self):
        """Test synthesis result structure."""
        from titan.batch.synthesizer import SynthesisResult

        result = SynthesisResult(
            batch_id="batch-123",
            summary="Test summary",
            themes=["theme1", "theme2"],
            key_insights=["insight1", "insight2"],
            cross_references=[{"connection": "A relates to B"}],
        )

        data = result.to_dict()

        assert data["batch_id"] == "batch-123"
        assert len(data["themes"]) == 2
        assert len(data["key_insights"]) == 2

    def test_parse_session_content(self):
        """Test parsing session markdown content."""
        from titan.batch.synthesizer import BatchSynthesizer

        synthesizer = BatchSynthesizer()

        content = """# Scope Clarification: Test Topic

**Workflow:** Quick Inquiry

## Overview

This is the overview.

### Key Points

- First key point about the topic
- Second key point with more detail
- Third point exploring another angle

## Results

Some results here.
"""

        summary = synthesizer._parse_session_content("session-1", content)

        assert summary["session_id"] == "session-1"
        assert len(summary["key_points"]) >= 3

    def test_extract_synthesis_without_llm(self):
        """Test synthesis extraction without LLM."""
        from titan.batch.synthesizer import BatchSynthesizer

        synthesizer = BatchSynthesizer(llm_caller=None)

        summaries = [
            {
                "topic": "AI Safety",
                "key_points": [
                    "Alignment is crucial",
                    "Interpretability helps understanding",
                    "Robustness against attacks",
                ],
                "stages": [],
            },
            {
                "topic": "Machine Learning",
                "key_points": [
                    "Deep learning advances",
                    "Interpretability in models",
                    "Training efficiency",
                ],
                "stages": [],
            },
        ]

        result = synthesizer._extract_synthesis("batch-123", summaries)

        assert result.batch_id == "batch-123"
        assert len(result.summary) > 0
        assert "2 topics" in result.summary or len(result.metadata.get("topics", [])) == 2


# =============================================================================
# Runtime Selector Load-Awareness Tests
# =============================================================================

class TestRuntimeSelectorLoadAwareness:
    """Tests for load-aware runtime selection."""

    def test_get_system_load(self):
        """Test getting system load."""
        from runtime.selector import get_system_load, SystemLoad

        load = get_system_load()

        assert isinstance(load, SystemLoad)
        assert load.cpu_percent >= 0
        assert load.memory_percent >= 0

    def test_load_caching(self):
        """Test load caching in selector."""
        from runtime.selector import RuntimeSelector

        selector = RuntimeSelector()

        load1 = selector.get_system_load()
        load2 = selector.get_system_load()

        # Should return cached value
        assert load1.timestamp == load2.timestamp

        # Force refresh
        load3 = selector.get_system_load(refresh=True)
        # Timestamp might be same if executed quickly, but should still work
        assert load3 is not None

    def test_load_aware_selection(self):
        """Test load-aware runtime selection."""
        from runtime.selector import (
            RuntimeSelector,
            SelectionStrategy,
            RuntimeType,
        )

        selector = RuntimeSelector(strategy=SelectionStrategy.LOAD_AWARE)

        # Selection should work without errors
        runtime = selector.select()
        assert runtime in RuntimeType

    def test_suggest_with_load_info(self):
        """Test suggestion includes load info."""
        from runtime.selector import RuntimeSelector

        selector = RuntimeSelector()
        suggestion = selector.suggest()

        assert "system_load" in suggestion
        assert "cpu_percent" in suggestion["system_load"]
        assert "level" in suggestion["system_load"]
