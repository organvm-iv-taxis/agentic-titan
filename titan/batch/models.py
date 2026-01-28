"""
Titan Batch - Data Models

Core dataclasses for batch research pipeline processing.
Defines BatchJob, QueuedSession, and BatchProgress for tracking
multi-topic inquiry batches with distributed worker execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class BatchStatus(str, Enum):
    """Status of a batch job."""

    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIALLY_COMPLETED = "partially_completed"


class SessionQueueStatus(str, Enum):
    """Status of a queued session within a batch."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class QueuedSession:
    """
    A single session queued within a batch job.

    Represents one topic from the batch that will be processed
    through the inquiry workflow.

    Attributes:
        id: Unique session identifier
        batch_id: Parent batch job ID
        topic: The topic to explore
        status: Current processing status
        worker_id: ID of worker processing this session
        celery_task_id: Celery task ID for tracking
        artifact_uri: URI where the artifact is stored
        inquiry_session_id: ID of the underlying InquirySession
        tokens_used: Total tokens consumed
        cost_usd: Estimated cost in USD
        retry_count: Number of retry attempts
        error: Error message if failed
        created_at: When the session was created
        started_at: When processing started
        completed_at: When processing completed
        metadata: Additional metadata
    """

    id: UUID = field(default_factory=uuid4)
    batch_id: UUID = field(default_factory=uuid4)
    topic: str = ""
    status: SessionQueueStatus = SessionQueueStatus.PENDING
    worker_id: str | None = None
    celery_task_id: str | None = None
    artifact_uri: str | None = None
    inquiry_session_id: str | None = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    retry_count: int = 0
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_queued(self, celery_task_id: str) -> None:
        """Mark session as queued in Celery."""
        self.status = SessionQueueStatus.QUEUED
        self.celery_task_id = celery_task_id

    def mark_running(self, worker_id: str) -> None:
        """Mark session as running on a worker."""
        self.status = SessionQueueStatus.RUNNING
        self.worker_id = worker_id
        self.started_at = datetime.now()

    def mark_completed(self, artifact_uri: str, tokens: int, cost: float) -> None:
        """Mark session as completed."""
        self.status = SessionQueueStatus.COMPLETED
        self.artifact_uri = artifact_uri
        self.tokens_used = tokens
        self.cost_usd = cost
        self.completed_at = datetime.now()

    def mark_failed(self, error: str) -> None:
        """Mark session as failed."""
        self.status = SessionQueueStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()

    def mark_retrying(self) -> None:
        """Mark session for retry."""
        self.status = SessionQueueStatus.RETRYING
        self.retry_count += 1
        self.error = None

    @property
    def is_terminal(self) -> bool:
        """Whether the session is in a terminal state."""
        return self.status in (
            SessionQueueStatus.COMPLETED,
            SessionQueueStatus.FAILED,
            SessionQueueStatus.CANCELLED,
        )

    @property
    def duration_ms(self) -> int | None:
        """Duration in milliseconds if completed."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "batch_id": str(self.batch_id),
            "topic": self.topic,
            "status": self.status.value,
            "worker_id": self.worker_id,
            "celery_task_id": self.celery_task_id,
            "artifact_uri": self.artifact_uri,
            "inquiry_session_id": self.inquiry_session_id,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "retry_count": self.retry_count,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueuedSession:
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            batch_id=UUID(data["batch_id"]) if isinstance(data["batch_id"], str) else data["batch_id"],
            topic=data["topic"],
            status=SessionQueueStatus(data["status"]),
            worker_id=data.get("worker_id"),
            celery_task_id=data.get("celery_task_id"),
            artifact_uri=data.get("artifact_uri"),
            inquiry_session_id=data.get("inquiry_session_id"),
            tokens_used=data.get("tokens_used", 0),
            cost_usd=data.get("cost_usd", 0.0),
            retry_count=data.get("retry_count", 0),
            error=data.get("error"),
            created_at=datetime.fromisoformat(data["created_at"])
            if isinstance(data["created_at"], str)
            else data["created_at"],
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at") and isinstance(data["started_at"], str)
            else data.get("started_at"),
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at") and isinstance(data["completed_at"], str)
            else data.get("completed_at"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BatchProgress:
    """
    Progress tracking for a batch job.

    Provides real-time status counts and computed metrics.
    """

    total: int = 0
    pending: int = 0
    queued: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    retrying: int = 0

    @property
    def percent_complete(self) -> float:
        """Percentage of sessions completed (including failed)."""
        if self.total == 0:
            return 100.0
        terminal = self.completed + self.failed + self.cancelled
        return (terminal / self.total) * 100

    @property
    def success_rate(self) -> float:
        """Percentage of completed sessions that succeeded."""
        terminal = self.completed + self.failed + self.cancelled
        if terminal == 0:
            return 100.0
        return (self.completed / terminal) * 100

    @property
    def is_complete(self) -> bool:
        """Whether all sessions are in terminal state."""
        terminal = self.completed + self.failed + self.cancelled
        return terminal >= self.total

    @property
    def active(self) -> int:
        """Number of active (non-terminal) sessions."""
        return self.pending + self.queued + self.running + self.retrying

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "pending": self.pending,
            "queued": self.queued,
            "running": self.running,
            "completed": self.completed,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "retrying": self.retrying,
            "percent_complete": round(self.percent_complete, 1),
            "success_rate": round(self.success_rate, 1),
            "is_complete": self.is_complete,
            "active": self.active,
        }


@dataclass
class BatchJob:
    """
    A batch job containing multiple research topics.

    Orchestrates the execution of multiple inquiry sessions,
    tracking overall progress and aggregated metrics.

    Attributes:
        id: Unique batch identifier
        topics: List of topics to explore
        workflow_name: Name of workflow to use for each topic
        max_concurrent: Maximum concurrent sessions
        budget_limit_usd: Optional budget cap for the batch
        status: Current batch status
        sessions: List of queued sessions
        synthesis_uri: URI of the cross-session synthesis artifact
        created_at: When the batch was created
        started_at: When processing started
        completed_at: When processing completed
        error: Error message if batch failed
        user_id: Optional user identifier
        metadata: Additional metadata
    """

    id: UUID = field(default_factory=uuid4)
    topics: list[str] = field(default_factory=list)
    workflow_name: str = "expansive"
    max_concurrent: int = 3
    budget_limit_usd: float | None = None
    status: BatchStatus = BatchStatus.PENDING
    sessions: list[QueuedSession] = field(default_factory=list)
    synthesis_uri: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and initialize sessions."""
        if not self.sessions and self.topics:
            # Auto-create sessions for topics
            self.sessions = [
                QueuedSession(
                    batch_id=self.id,
                    topic=topic,
                    status=SessionQueueStatus.PENDING,
                )
                for topic in self.topics
            ]

    @property
    def progress(self) -> BatchProgress:
        """Calculate current progress."""
        progress = BatchProgress(total=len(self.sessions))

        for session in self.sessions:
            if session.status == SessionQueueStatus.PENDING:
                progress.pending += 1
            elif session.status == SessionQueueStatus.QUEUED:
                progress.queued += 1
            elif session.status == SessionQueueStatus.RUNNING:
                progress.running += 1
            elif session.status == SessionQueueStatus.COMPLETED:
                progress.completed += 1
            elif session.status == SessionQueueStatus.FAILED:
                progress.failed += 1
            elif session.status == SessionQueueStatus.CANCELLED:
                progress.cancelled += 1
            elif session.status == SessionQueueStatus.RETRYING:
                progress.retrying += 1

        return progress

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all sessions."""
        return sum(s.tokens_used for s in self.sessions)

    @property
    def total_cost_usd(self) -> float:
        """Total cost in USD across all sessions."""
        return sum(s.cost_usd for s in self.sessions)

    @property
    def is_over_budget(self) -> bool:
        """Whether the batch has exceeded its budget limit."""
        if self.budget_limit_usd is None:
            return False
        return self.total_cost_usd > self.budget_limit_usd

    @property
    def remaining_budget_usd(self) -> float | None:
        """Remaining budget if set."""
        if self.budget_limit_usd is None:
            return None
        return max(0, self.budget_limit_usd - self.total_cost_usd)

    def get_session(self, session_id: UUID | str) -> QueuedSession | None:
        """Get a session by ID."""
        target = UUID(session_id) if isinstance(session_id, str) else session_id
        for session in self.sessions:
            if session.id == target:
                return session
        return None

    def get_pending_sessions(self, limit: int | None = None) -> list[QueuedSession]:
        """Get sessions waiting to be queued."""
        pending = [s for s in self.sessions if s.status == SessionQueueStatus.PENDING]
        if limit:
            return pending[:limit]
        return pending

    def get_active_sessions(self) -> list[QueuedSession]:
        """Get sessions currently being processed."""
        return [
            s for s in self.sessions
            if s.status in (SessionQueueStatus.QUEUED, SessionQueueStatus.RUNNING, SessionQueueStatus.RETRYING)
        ]

    def get_completed_sessions(self) -> list[QueuedSession]:
        """Get successfully completed sessions."""
        return [s for s in self.sessions if s.status == SessionQueueStatus.COMPLETED]

    def get_failed_sessions(self) -> list[QueuedSession]:
        """Get failed sessions."""
        return [s for s in self.sessions if s.status == SessionQueueStatus.FAILED]

    def mark_started(self) -> None:
        """Mark batch as started."""
        self.status = BatchStatus.PROCESSING
        self.started_at = datetime.now()

    def mark_completed(self) -> None:
        """Mark batch as completed."""
        progress = self.progress
        if progress.failed > 0 and progress.completed > 0:
            self.status = BatchStatus.PARTIALLY_COMPLETED
        elif progress.failed > 0:
            self.status = BatchStatus.FAILED
        else:
            self.status = BatchStatus.COMPLETED
        self.completed_at = datetime.now()

    def mark_cancelled(self) -> None:
        """Mark batch as cancelled."""
        self.status = BatchStatus.CANCELLED
        self.completed_at = datetime.now()
        # Cancel pending sessions
        for session in self.sessions:
            if not session.is_terminal:
                session.status = SessionQueueStatus.CANCELLED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "topics": self.topics,
            "workflow_name": self.workflow_name,
            "max_concurrent": self.max_concurrent,
            "budget_limit_usd": self.budget_limit_usd,
            "status": self.status.value,
            "sessions": [s.to_dict() for s in self.sessions],
            "synthesis_uri": self.synthesis_uri,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "user_id": self.user_id,
            "progress": self.progress.to_dict(),
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "remaining_budget_usd": self.remaining_budget_usd,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchJob:
        """Create from dictionary."""
        sessions = [
            QueuedSession.from_dict(s) for s in data.get("sessions", [])
        ]
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            topics=data.get("topics", []),
            workflow_name=data.get("workflow_name", "expansive"),
            max_concurrent=data.get("max_concurrent", 3),
            budget_limit_usd=data.get("budget_limit_usd"),
            status=BatchStatus(data["status"]),
            sessions=sessions,
            synthesis_uri=data.get("synthesis_uri"),
            created_at=datetime.fromisoformat(data["created_at"])
            if isinstance(data["created_at"], str)
            else data["created_at"],
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at") and isinstance(data["started_at"], str)
            else data.get("started_at"),
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at") and isinstance(data["completed_at"], str)
            else data.get("completed_at"),
            error=data.get("error"),
            user_id=data.get("user_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BatchSubmitRequest:
    """Request to submit a new batch job."""

    topics: list[str]
    workflow: str = "expansive"
    max_concurrent: int = 3
    budget_limit_usd: float | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate request."""
        if not self.topics:
            raise ValueError("At least one topic is required")
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if self.budget_limit_usd is not None and self.budget_limit_usd <= 0:
            raise ValueError("budget_limit_usd must be positive")


@dataclass
class SessionArtifact:
    """Metadata for a stored artifact."""

    session_id: UUID
    batch_id: UUID
    topic: str
    artifact_uri: str
    format: str = "markdown"
    size_bytes: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    checksum: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": str(self.session_id),
            "batch_id": str(self.batch_id),
            "topic": self.topic,
            "artifact_uri": self.artifact_uri,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
            "checksum": self.checksum,
            "metadata": self.metadata,
        }


# SQL table definitions for batch persistence
BATCH_JOBS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS batch_jobs (
    id UUID PRIMARY KEY,
    topics JSONB NOT NULL,
    workflow_name VARCHAR(50) NOT NULL DEFAULT 'expansive',
    max_concurrent INTEGER NOT NULL DEFAULT 3,
    budget_limit_usd FLOAT,
    status VARCHAR(30) NOT NULL DEFAULT 'pending',
    synthesis_uri VARCHAR(500),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error TEXT,
    user_id VARCHAR(100),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs(status);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_user_id ON batch_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_created_at ON batch_jobs(created_at);
"""

QUEUED_SESSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS queued_sessions (
    id UUID PRIMARY KEY,
    batch_id UUID REFERENCES batch_jobs(id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'pending',
    worker_id VARCHAR(100),
    celery_task_id VARCHAR(100),
    artifact_uri VARCHAR(500),
    inquiry_session_id VARCHAR(50),
    tokens_used INTEGER DEFAULT 0,
    cost_usd FLOAT DEFAULT 0,
    retry_count INTEGER DEFAULT 0,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_queued_sessions_batch_id ON queued_sessions(batch_id);
CREATE INDEX IF NOT EXISTS idx_queued_sessions_status ON queued_sessions(status);
CREATE INDEX IF NOT EXISTS idx_queued_sessions_celery_task_id ON queued_sessions(celery_task_id);
"""

SESSION_ARTIFACTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS session_artifacts (
    session_id UUID PRIMARY KEY REFERENCES queued_sessions(id) ON DELETE CASCADE,
    batch_id UUID REFERENCES batch_jobs(id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    artifact_uri VARCHAR(500) NOT NULL,
    format VARCHAR(20) DEFAULT 'markdown',
    size_bytes INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    checksum VARCHAR(64),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_session_artifacts_batch_id ON session_artifacts(batch_id);
"""
