"""
Titan API - Inquiry WebSocket Handler

WebSocket endpoint for streaming inquiry progress in real-time.
Clients can connect to receive stage-by-stage updates as the inquiry runs.

Protocol:
    Client sends: {"action": "start"|"cancel"}
    Server sends: {"type": "stage_started"|"stage_completed"|"session_completed"|...}
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from titan.api.typing_helpers import typed_get, typed_websocket
from titan.workflows.inquiry_engine import (
    InquiryEngine,
    InquiryStatus,
    get_inquiry_engine,
)

logger = logging.getLogger("titan.api.inquiry_ws")

ws_router = APIRouter()


class InquiryWebSocketManager:
    """
    Manages WebSocket connections for inquiry sessions.

    Handles:
    - Connection lifecycle
    - Message routing
    - Progress streaming
    """

    def __init__(self) -> None:
        # Active connections: session_id -> list of websockets
        self._connections: dict[str, list[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        session_id: str,
    ) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()

        async with self._lock:
            if session_id not in self._connections:
                self._connections[session_id] = []
            self._connections[session_id].append(websocket)

        logger.info(f"WebSocket connected for session {session_id}")

    async def disconnect(
        self,
        websocket: WebSocket,
        session_id: str,
    ) -> None:
        """Handle WebSocket disconnection."""
        async with self._lock:
            if session_id in self._connections:
                if websocket in self._connections[session_id]:
                    self._connections[session_id].remove(websocket)
                if not self._connections[session_id]:
                    del self._connections[session_id]

        logger.info(f"WebSocket disconnected from session {session_id}")

    async def broadcast(
        self,
        session_id: str,
        message: dict[str, Any],
    ) -> None:
        """Broadcast a message to all connections for a session."""
        async with self._lock:
            connections = self._connections.get(session_id, []).copy()

        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Error sending to WebSocket: {e}")
                # Don't remove here - let the disconnect handler do it

    async def send_to(
        self,
        websocket: WebSocket,
        message: dict[str, Any],
    ) -> None:
        """Send a message to a specific WebSocket."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Error sending to WebSocket: {e}")


# Global WebSocket manager
ws_manager = InquiryWebSocketManager()


@typed_websocket(ws_router, "/api/inquiry/{session_id}/stream")
async def inquiry_websocket(
    websocket: WebSocket,
    session_id: str,
) -> None:
    """
    WebSocket endpoint for streaming inquiry progress.

    Connect to receive real-time updates as stages execute.
    Send {"action": "start"} to begin running remaining stages.
    Send {"action": "cancel"} to cancel the inquiry.

    Events sent:
    - connected: Initial connection acknowledgment
    - session_status: Current session state
    - stage_started: Stage is beginning execution
    - stage_progress: Intermediate progress (if available)
    - stage_completed: Stage finished with results
    - session_completed: All stages finished
    - session_failed: Session encountered an error
    - session_cancelled: Session was cancelled
    - error: Error message
    """
    engine = get_inquiry_engine()

    # Validate session exists
    session = engine.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    # Connect
    await ws_manager.connect(websocket, session_id)

    try:
        # Send initial status
        await ws_manager.send_to(
            websocket,
            {
                "type": "connected",
                "session_id": session_id,
                "topic": session.topic,
                "status": session.status.value,
                "progress": session.progress,
                "stages_completed": len(session.results),
                "total_stages": session.total_stages,
            },
        )

        # Send existing results if any
        if session.results:
            await ws_manager.send_to(
                websocket,
                {
                    "type": "session_status",
                    "session_id": session_id,
                    "results": [r.to_dict() for r in session.results],
                },
            )

        # Main message loop
        while True:
            try:
                # Wait for client message
                data = await websocket.receive_json()
                action = data.get("action")

                if action == "start":
                    # Start running remaining stages
                    await handle_start_action(websocket, session_id, engine)

                elif action == "cancel":
                    # Cancel the session
                    success = engine.cancel_session(session_id)
                    await ws_manager.send_to(
                        websocket,
                        {
                            "type": "session_cancelled" if success else "error",
                            "session_id": session_id,
                            "message": "Session cancelled" if success else "Cannot cancel session",
                        },
                    )

                elif action == "status":
                    # Get current status
                    session = engine.get_session(session_id)
                    if session:
                        await ws_manager.send_to(
                            websocket,
                            {
                                "type": "session_status",
                                "session_id": session_id,
                                "status": session.status.value,
                                "progress": session.progress,
                                "stages_completed": len(session.results),
                            },
                        )

                else:
                    await ws_manager.send_to(
                        websocket,
                        {
                            "type": "error",
                            "message": f"Unknown action: {action}",
                        },
                    )

            except json.JSONDecodeError:
                await ws_manager.send_to(
                    websocket,
                    {"type": "error", "message": "Invalid JSON"},
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected from session {session_id}")
    except Exception as e:
        logger.exception(f"WebSocket error for session {session_id}: {e}")
        try:
            await ws_manager.send_to(
                websocket,
                {"type": "error", "message": str(e)},
            )
        except Exception:
            pass
    finally:
        await ws_manager.disconnect(websocket, session_id)


async def handle_start_action(
    websocket: WebSocket,
    session_id: str,
    engine: InquiryEngine,
) -> None:
    """
    Handle the 'start' action - run remaining stages with streaming updates.
    """
    session = engine.get_session(session_id)
    if not session:
        await ws_manager.send_to(
            websocket,
            {"type": "error", "message": "Session not found"},
        )
        return

    if session.status in (InquiryStatus.COMPLETED, InquiryStatus.CANCELLED, InquiryStatus.FAILED):
        await ws_manager.send_to(
            websocket,
            {
                "type": "error",
                "message": f"Session cannot be started: {session.status.value}",
            },
        )
        return

    # Stream the workflow
    try:
        async for event in engine.stream_workflow(session):
            # Broadcast to all connected clients
            await ws_manager.broadcast(session_id, event)

    except Exception as e:
        logger.exception(f"Error streaming workflow for session {session_id}")
        await ws_manager.broadcast(
            session_id,
            {
                "type": "session_failed",
                "session_id": session_id,
                "error": str(e),
            },
        )


# Alternative endpoint using Server-Sent Events (SSE) for simpler clients
@typed_get(ws_router, "/api/inquiry/{session_id}/events")
async def inquiry_sse(
    session_id: str,
) -> StreamingResponse | dict[str, str]:
    """
    Server-Sent Events endpoint for inquiry progress.

    Alternative to WebSocket for clients that prefer SSE.
    """
    engine = get_inquiry_engine()
    session = engine.get_session(session_id)

    if not session:
        return {"error": "Session not found"}

    async def event_generator() -> AsyncIterator[str]:
        """Generate SSE events."""
        # Send initial status
        initial_event = {"type": "connected", "session_id": session_id}
        yield f"data: {json.dumps(initial_event)}\n\n"

        if session.status in (
            InquiryStatus.COMPLETED,
            InquiryStatus.CANCELLED,
            InquiryStatus.FAILED,
        ):
            status_event = {"type": "session_status", "status": session.status.value}
            yield f"data: {json.dumps(status_event)}\n\n"
            return

        # Stream workflow events
        try:
            async for event in engine.stream_workflow(session):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
