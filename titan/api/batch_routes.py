"""
Titan API - Batch Routes

REST API endpoints for batch research pipeline.

Endpoints:
    POST /api/batch/submit        - Submit new batch job
    GET  /api/batch/{batch_id}    - Get batch status and progress
    POST /api/batch/{batch_id}/start - Start processing a batch
    POST /api/batch/{batch_id}/cancel - Cancel a batch
    POST /api/batch/{batch_id}/pause  - Pause a batch
    GET  /api/batch/{batch_id}/artifacts - List artifacts
    GET  /api/batch/{batch_id}/export - Download as archive
    GET  /api/batch/list          - List all batches
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from titan.batch.models import (
    BatchJob,
    BatchProgress,
    BatchStatus,
    BatchSubmitRequest,
)
from titan.batch.orchestrator import (
    BatchOrchestrator,
    get_batch_orchestrator,
)
from titan.batch.artifact_store import get_artifact_store
from titan.workflows.inquiry_config import list_workflows

logger = logging.getLogger("titan.api.batch")

batch_router = APIRouter(prefix="/batch", tags=["batch"])


# =============================================================================
# Request/Response Models
# =============================================================================

class SubmitBatchRequest(BaseModel):
    """Request to submit a new batch job."""

    topics: list[str] = Field(
        ...,
        description="List of topics to explore",
        min_length=1,
        max_length=100,
    )
    workflow: str = Field(
        default="expansive",
        description="Workflow to use for each topic",
    )
    max_concurrent: int = Field(
        default=3,
        description="Maximum concurrent sessions",
        ge=1,
        le=10,
    )
    budget_limit_usd: float | None = Field(
        default=None,
        description="Optional budget limit in USD",
        gt=0,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata",
    )


class SubmitBatchResponse(BaseModel):
    """Response after submitting a batch."""

    batch_id: str
    session_count: int
    status: str
    estimated_cost_usd: float | None
    workflow: str


class BatchStatusResponse(BaseModel):
    """Batch status and progress."""

    id: str
    status: str
    topics: list[str]
    workflow_name: str
    max_concurrent: int
    budget_limit_usd: float | None
    progress: dict[str, Any]
    total_tokens: int
    total_cost_usd: float
    remaining_budget_usd: float | None
    synthesis_uri: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None
    error: str | None


class SessionStatusResponse(BaseModel):
    """Status of a single session in a batch."""

    id: str
    topic: str
    status: str
    worker_id: str | None
    artifact_uri: str | None
    tokens_used: int
    cost_usd: float
    retry_count: int
    error: str | None
    duration_ms: int | None


class ArtifactResponse(BaseModel):
    """Artifact metadata response."""

    session_id: str
    batch_id: str
    topic: str
    artifact_uri: str
    format: str
    size_bytes: int
    created_at: str


class BatchListResponse(BaseModel):
    """Summary for batch list."""

    id: str
    status: str
    topic_count: int
    workflow: str
    progress_percent: float
    created_at: str


# =============================================================================
# Endpoints
# =============================================================================

@batch_router.post("/submit", response_model=SubmitBatchResponse)
async def submit_batch(request: SubmitBatchRequest) -> SubmitBatchResponse:
    """
    Submit a new batch job.

    Creates a batch with the specified topics and configuration.
    The batch is created in PENDING status and must be started
    by calling /batch/{id}/start.
    """
    orchestrator = get_batch_orchestrator()

    # Validate workflow
    if request.workflow not in list_workflows():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown workflow: {request.workflow}. "
            f"Available: {list_workflows()}",
        )

    try:
        submit_request = BatchSubmitRequest(
            topics=request.topics,
            workflow=request.workflow,
            max_concurrent=request.max_concurrent,
            budget_limit_usd=request.budget_limit_usd,
            metadata=request.metadata,
        )
        batch = await orchestrator.submit_batch(submit_request)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        f"Submitted batch {batch.id} with {len(batch.topics)} topics"
    )

    return SubmitBatchResponse(
        batch_id=str(batch.id),
        session_count=len(batch.sessions),
        status=batch.status.value,
        estimated_cost_usd=batch.metadata.get("estimated_cost_usd"),
        workflow=batch.workflow_name,
    )


@batch_router.get("/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(batch_id: str) -> BatchStatusResponse:
    """Get the current status and progress of a batch."""
    orchestrator = get_batch_orchestrator()
    batch = orchestrator.get_batch(batch_id)

    if not batch:
        raise HTTPException(
            status_code=404,
            detail=f"Batch not found: {batch_id}",
        )

    return BatchStatusResponse(
        id=str(batch.id),
        status=batch.status.value,
        topics=batch.topics,
        workflow_name=batch.workflow_name,
        max_concurrent=batch.max_concurrent,
        budget_limit_usd=batch.budget_limit_usd,
        progress=batch.progress.to_dict(),
        total_tokens=batch.total_tokens,
        total_cost_usd=batch.total_cost_usd,
        remaining_budget_usd=batch.remaining_budget_usd,
        synthesis_uri=batch.synthesis_uri,
        created_at=batch.created_at.isoformat(),
        started_at=batch.started_at.isoformat() if batch.started_at else None,
        completed_at=batch.completed_at.isoformat() if batch.completed_at else None,
        error=batch.error,
    )


@batch_router.get("/{batch_id}/sessions", response_model=list[SessionStatusResponse])
async def get_batch_sessions(
    batch_id: str,
    status: str | None = Query(None, description="Filter by status"),
) -> list[SessionStatusResponse]:
    """Get status of all sessions in a batch."""
    orchestrator = get_batch_orchestrator()
    batch = orchestrator.get_batch(batch_id)

    if not batch:
        raise HTTPException(
            status_code=404,
            detail=f"Batch not found: {batch_id}",
        )

    sessions = batch.sessions
    if status:
        sessions = [s for s in sessions if s.status.value == status]

    return [
        SessionStatusResponse(
            id=str(s.id),
            topic=s.topic,
            status=s.status.value,
            worker_id=s.worker_id,
            artifact_uri=s.artifact_uri,
            tokens_used=s.tokens_used,
            cost_usd=s.cost_usd,
            retry_count=s.retry_count,
            error=s.error,
            duration_ms=s.duration_ms,
        )
        for s in sessions
    ]


@batch_router.post("/{batch_id}/start", response_model=BatchStatusResponse)
async def start_batch(batch_id: str) -> BatchStatusResponse:
    """
    Start processing a batch.

    Queues sessions to workers respecting concurrency limits.
    """
    orchestrator = get_batch_orchestrator()

    try:
        batch = await orchestrator.start_batch(batch_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"Started batch {batch_id}")

    return BatchStatusResponse(
        id=str(batch.id),
        status=batch.status.value,
        topics=batch.topics,
        workflow_name=batch.workflow_name,
        max_concurrent=batch.max_concurrent,
        budget_limit_usd=batch.budget_limit_usd,
        progress=batch.progress.to_dict(),
        total_tokens=batch.total_tokens,
        total_cost_usd=batch.total_cost_usd,
        remaining_budget_usd=batch.remaining_budget_usd,
        synthesis_uri=batch.synthesis_uri,
        created_at=batch.created_at.isoformat(),
        started_at=batch.started_at.isoformat() if batch.started_at else None,
        completed_at=batch.completed_at.isoformat() if batch.completed_at else None,
        error=batch.error,
    )


@batch_router.post("/{batch_id}/cancel")
async def cancel_batch(batch_id: str) -> dict[str, Any]:
    """Cancel a running batch."""
    orchestrator = get_batch_orchestrator()

    try:
        batch = await orchestrator.cancel_batch(batch_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"Cancelled batch {batch_id}")

    return {
        "status": "cancelled",
        "batch_id": str(batch.id),
        "sessions_cancelled": batch.progress.cancelled,
    }


@batch_router.post("/{batch_id}/pause")
async def pause_batch(batch_id: str) -> dict[str, Any]:
    """Pause a running batch."""
    orchestrator = get_batch_orchestrator()

    try:
        batch = await orchestrator.pause_batch(batch_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"Paused batch {batch_id}")

    return {
        "status": "paused",
        "batch_id": str(batch.id),
        "progress": batch.progress.to_dict(),
    }


@batch_router.get("/{batch_id}/artifacts", response_model=list[ArtifactResponse])
async def list_artifacts(batch_id: str) -> list[ArtifactResponse]:
    """List all artifacts for a batch."""
    orchestrator = get_batch_orchestrator()
    batch = orchestrator.get_batch(batch_id)

    if not batch:
        raise HTTPException(
            status_code=404,
            detail=f"Batch not found: {batch_id}",
        )

    artifact_store = get_artifact_store()
    artifacts = await artifact_store.list_artifacts(batch_id)

    return [
        ArtifactResponse(
            session_id=str(a.session_id),
            batch_id=str(a.batch_id),
            topic=a.topic,
            artifact_uri=a.artifact_uri,
            format=a.format,
            size_bytes=a.size_bytes,
            created_at=a.created_at.isoformat(),
        )
        for a in artifacts
    ]


@batch_router.get("/{batch_id}/export")
async def export_batch(
    batch_id: str,
    format: str = Query("zip", description="Archive format"),
) -> StreamingResponse:
    """
    Export all batch artifacts as archive.

    Returns a ZIP file containing all session artifacts.
    """
    orchestrator = get_batch_orchestrator()
    batch = orchestrator.get_batch(batch_id)

    if not batch:
        raise HTTPException(
            status_code=404,
            detail=f"Batch not found: {batch_id}",
        )

    if format != "zip":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {format}. Only 'zip' is supported.",
        )

    artifact_store = get_artifact_store()

    try:
        archive_bytes = await artifact_store.export_batch_archive(batch_id)
    except Exception as e:
        logger.error(f"Failed to export batch {batch_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export batch: {e}",
        )

    # Generate filename from first topic
    topic_slug = batch.topics[0][:30].lower() if batch.topics else "batch"
    topic_slug = "".join(c for c in topic_slug if c.isalnum() or c == "-")
    filename = f"{topic_slug}-batch-{batch_id[:8]}.zip"

    return StreamingResponse(
        iter([archive_bytes]),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(archive_bytes)),
        },
    )


@batch_router.get("/list", response_model=list[BatchListResponse])
async def list_batches(
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, description="Maximum batches to return", le=100),
) -> list[BatchListResponse]:
    """List all batches, optionally filtered by status."""
    orchestrator = get_batch_orchestrator()

    # Parse status filter
    status_filter = None
    if status:
        try:
            status_filter = BatchStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. "
                f"Valid values: {[s.value for s in BatchStatus]}",
            )

    batches = orchestrator.list_batches(status=status_filter)[:limit]

    return [
        BatchListResponse(
            id=str(b.id),
            status=b.status.value,
            topic_count=len(b.topics),
            workflow=b.workflow_name,
            progress_percent=b.progress.percent_complete,
            created_at=b.created_at.isoformat(),
        )
        for b in batches
    ]


@batch_router.post("/{batch_id}/synthesize")
async def synthesize_batch(batch_id: str) -> dict[str, Any]:
    """
    Trigger cross-session synthesis for a batch.

    Generates a unified summary across all completed sessions.
    """
    orchestrator = get_batch_orchestrator()
    batch = orchestrator.get_batch(batch_id)

    if not batch:
        raise HTTPException(
            status_code=404,
            detail=f"Batch not found: {batch_id}",
        )

    if batch.progress.completed == 0:
        raise HTTPException(
            status_code=400,
            detail="No completed sessions to synthesize",
        )

    from titan.batch.synthesizer import get_batch_synthesizer

    synthesizer = get_batch_synthesizer()

    try:
        result = await synthesizer.synthesize_batch(batch_id, batch)
    except Exception as e:
        logger.error(f"Synthesis failed for batch {batch_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis failed: {e}",
        )

    # Update batch with synthesis URI
    if result.get("artifact_uri"):
        batch.synthesis_uri = result["artifact_uri"]

    logger.info(f"Synthesized batch {batch_id}")

    return {
        "status": "completed",
        "batch_id": batch_id,
        "synthesis_uri": result.get("artifact_uri"),
        "themes": result.get("themes", []),
        "insight_count": len(result.get("key_insights", [])),
    }
