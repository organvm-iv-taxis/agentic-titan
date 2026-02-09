"""
Titan API - Inquiry Routes

REST API endpoints for the multi-perspective collaborative inquiry system.

Endpoints:
    POST /api/inquiry/start         - Start new inquiry session
    GET  /api/inquiry/{session_id}  - Get session status and results
    POST /api/inquiry/{session_id}/run-stage - Run next stage
    POST /api/inquiry/{session_id}/run-all   - Run all remaining stages
    POST /api/inquiry/{session_id}/cancel    - Cancel running session
    GET  /api/inquiry/workflows     - List available workflows
    GET  /api/inquiry/sessions      - List all sessions
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

from titan.api.typing_helpers import BaseModel, Field, typed_get, typed_post
from titan.workflows.inquiry_config import (
    DEFAULT_WORKFLOWS,
    get_workflow,
    list_workflows,
)
from titan.workflows.inquiry_engine import (
    InquiryStatus,
    get_inquiry_engine,
)
from titan.workflows.inquiry_export import (
    export_session_to_markdown,
    export_stage_to_markdown,
)

logger = logging.getLogger("titan.api.inquiry")

inquiry_router = APIRouter(prefix="/inquiry", tags=["inquiry"])


# =============================================================================
# Request/Response Models
# =============================================================================


class StartInquiryRequest(BaseModel):
    """Request to start a new inquiry session."""

    topic: str = Field(..., description="The topic to explore", min_length=1)
    workflow: str = Field(
        default="expansive",
        description="Name of workflow to use (expansive, quick, creative)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata to attach to the session",
    )


class StartInquiryResponse(BaseModel):
    """Response after starting an inquiry."""

    session_id: str
    topic: str
    workflow_name: str
    total_stages: int
    status: str


class SessionResponse(BaseModel):
    """Full session details response."""

    id: str
    topic: str
    workflow_name: str
    status: str
    current_stage: int
    total_stages: int
    progress: float
    results: list[dict[str, Any]]
    created_at: str
    started_at: str | None
    completed_at: str | None
    error: str | None


class StageResultResponse(BaseModel):
    """Response for a single stage result."""

    stage_name: str
    role: str
    content: str
    model_used: str
    timestamp: str
    tokens_used: int
    duration_ms: int
    stage_index: int
    error: str | None


class WorkflowResponse(BaseModel):
    """Response describing a workflow."""

    name: str
    description: str
    stages: list[dict[str, Any]]
    stage_count: int


class ExportResponse(BaseModel):
    """Response with exported markdown content."""

    markdown: str
    filename: str


# =============================================================================
# Endpoints
# =============================================================================


@typed_post(inquiry_router, "/start", response_model=StartInquiryResponse)
async def start_inquiry(
    request: StartInquiryRequest,
    background_tasks: BackgroundTasks,
) -> StartInquiryResponse:
    """
    Start a new inquiry session.

    Creates a new session with the specified topic and workflow.
    The session is created in PENDING status and must be advanced
    by calling run-stage or run-all.
    """
    engine = get_inquiry_engine()

    # Get the workflow
    workflow = get_workflow(request.workflow)
    if not workflow:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown workflow: {request.workflow}. Available: {list_workflows()}",
        )

    # Start the session
    session = await engine.start_inquiry(
        topic=request.topic,
        workflow=workflow,
        metadata=request.metadata,
    )

    logger.info(f"Started inquiry session {session.id} for topic: {request.topic[:50]}")

    return StartInquiryResponse(
        session_id=session.id,
        topic=session.topic,
        workflow_name=session.workflow.name,
        total_stages=session.total_stages,
        status=session.status.value,
    )


@typed_get(inquiry_router, "/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str) -> SessionResponse:
    """
    Get the current status and results of an inquiry session.
    """
    engine = get_inquiry_engine()
    session = engine.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}",
        )

    return SessionResponse(
        id=session.id,
        topic=session.topic,
        workflow_name=session.workflow.name,
        status=session.status.value,
        current_stage=session.current_stage,
        total_stages=session.total_stages,
        progress=session.progress,
        results=[r.to_dict() for r in session.results],
        created_at=session.created_at.isoformat(),
        started_at=session.started_at.isoformat() if session.started_at else None,
        completed_at=session.completed_at.isoformat() if session.completed_at else None,
        error=session.error,
    )


@typed_post(inquiry_router, "/{session_id}/run-stage", response_model=StageResultResponse)
async def run_stage(
    session_id: str,
    stage_index: int | None = None,
) -> StageResultResponse:
    """
    Run the next stage (or a specific stage) of the inquiry.

    If stage_index is not provided, runs the next unexecuted stage.
    """
    engine = get_inquiry_engine()
    session = engine.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}",
        )

    if session.status == InquiryStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail="Session is already completed",
        )

    if session.status == InquiryStatus.CANCELLED:
        raise HTTPException(
            status_code=400,
            detail="Session was cancelled",
        )

    if session.status == InquiryStatus.FAILED:
        raise HTTPException(
            status_code=400,
            detail=f"Session failed: {session.error}",
        )

    try:
        result = await engine.run_stage(session, stage_index)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error running stage for session {session_id}")
        raise HTTPException(status_code=500, detail=str(e))

    return StageResultResponse(
        stage_name=result.stage_name,
        role=result.role,
        content=result.content,
        model_used=result.model_used,
        timestamp=result.timestamp.isoformat(),
        tokens_used=result.tokens_used,
        duration_ms=result.duration_ms,
        stage_index=result.stage_index,
        error=result.error,
    )


@typed_post(inquiry_router, "/{session_id}/run-all", response_model=SessionResponse)
async def run_all_stages(session_id: str) -> SessionResponse:
    """
    Run all remaining stages of the inquiry.

    Blocks until all stages complete or an error occurs.
    """
    engine = get_inquiry_engine()
    session = engine.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}",
        )

    if session.status in (InquiryStatus.COMPLETED, InquiryStatus.CANCELLED, InquiryStatus.FAILED):
        raise HTTPException(
            status_code=400,
            detail=f"Session cannot be run: status is {session.status.value}",
        )

    try:
        session = await engine.run_full_workflow(session)
    except Exception as e:
        logger.exception(f"Error running workflow for session {session_id}")
        raise HTTPException(status_code=500, detail=str(e))

    return SessionResponse(
        id=session.id,
        topic=session.topic,
        workflow_name=session.workflow.name,
        status=session.status.value,
        current_stage=session.current_stage,
        total_stages=session.total_stages,
        progress=session.progress,
        results=[r.to_dict() for r in session.results],
        created_at=session.created_at.isoformat(),
        started_at=session.started_at.isoformat() if session.started_at else None,
        completed_at=session.completed_at.isoformat() if session.completed_at else None,
        error=session.error,
    )


@typed_post(inquiry_router, "/{session_id}/cancel")
async def cancel_session(session_id: str) -> dict[str, Any]:
    """
    Cancel a running inquiry session.
    """
    engine = get_inquiry_engine()

    if not engine.get_session(session_id):
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}",
        )

    success = engine.cancel_session(session_id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail="Session cannot be cancelled (may not be running)",
        )

    return {"status": "cancelled", "session_id": session_id}


@typed_get(inquiry_router, "/{session_id}/export", response_model=ExportResponse)
async def export_session(
    session_id: str,
    stage_index: int | None = None,
) -> ExportResponse:
    """
    Export session or stage to markdown.

    If stage_index is provided, exports only that stage.
    Otherwise, exports the full session.
    """
    engine = get_inquiry_engine()
    session = engine.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}",
        )

    if not session.results:
        raise HTTPException(
            status_code=400,
            detail="Session has no results to export",
        )

    try:
        if stage_index is not None:
            markdown = export_stage_to_markdown(session, stage_index)
            stage = session.workflow.stages[stage_index]
            filename = f"{session_id}-{stage_index:02d}-{stage.name.lower().replace(' ', '-')}.md"
        else:
            markdown = export_session_to_markdown(session)
            filename = f"{session_id}-full-inquiry.md"

    except (ValueError, IndexError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ExportResponse(markdown=markdown, filename=filename)


@typed_get(inquiry_router, "/workflows", response_model=list[WorkflowResponse])
async def list_available_workflows() -> list[WorkflowResponse]:
    """
    List all available inquiry workflows.
    """
    return [
        WorkflowResponse(
            name=w.name,
            description=w.description,
            stages=[s.to_dict() for s in w.stages],
            stage_count=len(w.stages),
        )
        for w in DEFAULT_WORKFLOWS.values()
    ]


@typed_get(inquiry_router, "/sessions")
async def list_sessions(
    status: str | None = None,
) -> list[dict[str, Any]]:
    """
    List all inquiry sessions, optionally filtered by status.
    """
    engine = get_inquiry_engine()

    # Parse status filter
    status_filter = None
    if status:
        try:
            status_filter = InquiryStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. "
                f"Valid values: {[s.value for s in InquiryStatus]}",
            )

    sessions = engine.list_sessions(status=status_filter)

    return [
        {
            "id": s.id,
            "topic": s.topic[:100],  # Truncate for list view
            "workflow_name": s.workflow.name,
            "status": s.status.value,
            "progress": s.progress,
            "stages_completed": len(s.results),
            "total_stages": s.total_stages,
            "created_at": s.created_at.isoformat(),
        }
        for s in sessions
    ]
