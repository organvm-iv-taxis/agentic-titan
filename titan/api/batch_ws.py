"""
Titan API - Batch WebSocket Handler

WebSocket endpoint for real-time batch progress streaming.
Provides live updates as sessions complete within a batch.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from titan.batch.orchestrator import get_batch_orchestrator

logger = logging.getLogger("titan.api.batch_ws")

batch_ws_router = APIRouter()


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class BatchConnectionManager:
    """
    Manages WebSocket connections for batch progress streaming.

    Tracks active connections per batch and handles broadcasting.
    """

    def __init__(self) -> None:
        # Map batch_id -> list of active websockets
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, batch_id: str) -> None:
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        if batch_id not in self._connections:
            self._connections[batch_id] = []
        self._connections[batch_id].append(websocket)
        logger.debug(f"WebSocket connected for batch {batch_id}")

    def disconnect(self, websocket: WebSocket, batch_id: str) -> None:
        """Remove a WebSocket connection."""
        if batch_id in self._connections:
            if websocket in self._connections[batch_id]:
                self._connections[batch_id].remove(websocket)
            if not self._connections[batch_id]:
                del self._connections[batch_id]
        logger.debug(f"WebSocket disconnected for batch {batch_id}")

    async def broadcast(self, batch_id: str, message: dict[str, Any]) -> None:
        """Broadcast a message to all connections for a batch."""
        if batch_id not in self._connections:
            return

        disconnected = []
        for websocket in self._connections[batch_id]:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)

        # Clean up disconnected clients
        for ws in disconnected:
            self.disconnect(ws, batch_id)

    def get_connection_count(self, batch_id: str) -> int:
        """Get number of active connections for a batch."""
        return len(self._connections.get(batch_id, []))


# Global connection manager
manager = BatchConnectionManager()


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@batch_ws_router.websocket("/batch/{batch_id}/ws")
async def batch_progress_websocket(
    websocket: WebSocket,
    batch_id: str,
) -> None:
    """
    WebSocket endpoint for real-time batch progress.

    Streams progress events until the batch completes or connection closes.

    Events:
    - batch_info: Initial batch information
    - progress: Progress update with counts
    - session_started: A session started processing
    - session_completed: A session completed
    - session_failed: A session failed
    - batch_completed: Batch finished processing
    - error: Error occurred

    Client can send:
    - {"type": "ping"}: Keep-alive ping
    - {"type": "close"}: Request graceful close
    """
    orchestrator = get_batch_orchestrator()

    # Verify batch exists
    batch = orchestrator.get_batch(batch_id)
    if not batch:
        await websocket.close(code=4004, reason="Batch not found")
        return

    await manager.connect(websocket, batch_id)

    try:
        # Send initial batch info
        await websocket.send_json({
            "type": "batch_info",
            "batch_id": batch_id,
            "topics": batch.topics,
            "total_sessions": len(batch.sessions),
            "status": batch.status.value,
            "workflow": batch.workflow_name,
        })

        # Create tasks for streaming and receiving
        stream_task = asyncio.create_task(
            _stream_progress(websocket, batch_id)
        )
        receive_task = asyncio.create_task(
            _receive_messages(websocket, batch_id)
        )

        # Wait for either task to complete
        done, pending = await asyncio.wait(
            [stream_task, receive_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        logger.debug(f"WebSocket disconnected for batch {batch_id}")
    except Exception as e:
        logger.error(f"WebSocket error for batch {batch_id}: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e),
            })
        except Exception:
            pass
    finally:
        manager.disconnect(websocket, batch_id)


async def _stream_progress(
    websocket: WebSocket,
    batch_id: str,
    poll_interval: float = 1.0,
) -> None:
    """Stream progress updates to WebSocket client."""
    orchestrator = get_batch_orchestrator()
    last_progress = None
    last_session_states: dict[str, str] = {}

    while True:
        batch = orchestrator.get_batch(batch_id)
        if not batch:
            await websocket.send_json({
                "type": "error",
                "error": "Batch no longer exists",
            })
            break

        # Check for session state changes
        for session in batch.sessions:
            session_id = str(session.id)
            current_state = session.status.value

            if session_id not in last_session_states:
                last_session_states[session_id] = current_state
            elif last_session_states[session_id] != current_state:
                # State changed
                last_session_states[session_id] = current_state

                if current_state == "running":
                    await websocket.send_json({
                        "type": "session_started",
                        "session_id": session_id,
                        "topic": session.topic,
                        "worker_id": session.worker_id,
                    })
                elif current_state == "completed":
                    await websocket.send_json({
                        "type": "session_completed",
                        "session_id": session_id,
                        "topic": session.topic,
                        "artifact_uri": session.artifact_uri,
                        "tokens_used": session.tokens_used,
                        "cost_usd": session.cost_usd,
                        "duration_ms": session.duration_ms,
                    })
                elif current_state == "failed":
                    await websocket.send_json({
                        "type": "session_failed",
                        "session_id": session_id,
                        "topic": session.topic,
                        "error": session.error,
                        "retry_count": session.retry_count,
                    })

        # Send progress update if changed
        progress = batch.progress.to_dict()
        if progress != last_progress:
            await websocket.send_json({
                "type": "progress",
                "batch_id": batch_id,
                "status": batch.status.value,
                **progress,
                "total_tokens": batch.total_tokens,
                "total_cost_usd": batch.total_cost_usd,
            })
            last_progress = progress

        # Check for completion
        if batch.status.value in ("completed", "failed", "cancelled", "partially_completed"):
            await websocket.send_json({
                "type": "batch_completed",
                "batch_id": batch_id,
                "status": batch.status.value,
                "synthesis_uri": batch.synthesis_uri,
                "total_tokens": batch.total_tokens,
                "total_cost_usd": batch.total_cost_usd,
                "progress": progress,
            })
            break

        await asyncio.sleep(poll_interval)


async def _receive_messages(
    websocket: WebSocket,
    batch_id: str,
) -> None:
    """Handle incoming messages from WebSocket client."""
    while True:
        try:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "close":
                await websocket.send_json({"type": "closing"})
                await websocket.close()
                break

            elif msg_type == "get_session":
                # Get status of specific session
                session_id = data.get("session_id")
                orchestrator = get_batch_orchestrator()
                batch = orchestrator.get_batch(batch_id)

                if batch:
                    session = batch.get_session(session_id)
                    if session:
                        await websocket.send_json({
                            "type": "session_status",
                            "session_id": str(session.id),
                            "topic": session.topic,
                            "status": session.status.value,
                            "tokens_used": session.tokens_used,
                            "error": session.error,
                        })

        except WebSocketDisconnect:
            break
        except json.JSONDecodeError:
            await websocket.send_json({
                "type": "error",
                "error": "Invalid JSON",
            })
        except Exception as e:
            logger.warning(f"Error processing WebSocket message: {e}")


# =============================================================================
# Server-Sent Events Alternative
# =============================================================================

@batch_ws_router.get("/batch/{batch_id}/stream")
async def batch_progress_sse(batch_id: str):
    """
    Server-Sent Events endpoint for batch progress.

    Alternative to WebSocket for simpler clients.
    """
    from fastapi.responses import StreamingResponse

    orchestrator = get_batch_orchestrator()
    batch = orchestrator.get_batch(batch_id)

    if not batch:
        return StreamingResponse(
            iter([f"event: error\ndata: Batch not found\n\n"]),
            media_type="text/event-stream",
            status_code=404,
        )

    async def event_generator():
        """Generate SSE events."""
        last_progress = None

        # Initial info
        yield f"event: batch_info\ndata: {json.dumps({'batch_id': batch_id, 'total_sessions': len(batch.sessions)})}\n\n"

        while True:
            current_batch = orchestrator.get_batch(batch_id)
            if not current_batch:
                yield f"event: error\ndata: Batch no longer exists\n\n"
                break

            progress = current_batch.progress.to_dict()
            if progress != last_progress:
                yield f"event: progress\ndata: {json.dumps(progress)}\n\n"
                last_progress = progress

            if current_batch.status.value in ("completed", "failed", "cancelled", "partially_completed"):
                yield f"event: batch_completed\ndata: {json.dumps({'status': current_batch.status.value})}\n\n"
                break

            await asyncio.sleep(1.0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# Broadcast Helpers (for use by orchestrator)
# =============================================================================

async def broadcast_session_started(
    batch_id: str,
    session_id: str,
    topic: str,
    worker_id: str,
) -> None:
    """Broadcast session started event."""
    await manager.broadcast(batch_id, {
        "type": "session_started",
        "session_id": session_id,
        "topic": topic,
        "worker_id": worker_id,
    })


async def broadcast_session_completed(
    batch_id: str,
    session_id: str,
    topic: str,
    artifact_uri: str,
    tokens_used: int,
    cost_usd: float,
) -> None:
    """Broadcast session completed event."""
    await manager.broadcast(batch_id, {
        "type": "session_completed",
        "session_id": session_id,
        "topic": topic,
        "artifact_uri": artifact_uri,
        "tokens_used": tokens_used,
        "cost_usd": cost_usd,
    })


async def broadcast_session_failed(
    batch_id: str,
    session_id: str,
    topic: str,
    error: str,
) -> None:
    """Broadcast session failed event."""
    await manager.broadcast(batch_id, {
        "type": "session_failed",
        "session_id": session_id,
        "topic": topic,
        "error": error,
    })


async def broadcast_batch_completed(
    batch_id: str,
    status: str,
    synthesis_uri: str | None,
) -> None:
    """Broadcast batch completed event."""
    await manager.broadcast(batch_id, {
        "type": "batch_completed",
        "batch_id": batch_id,
        "status": status,
        "synthesis_uri": synthesis_uri,
    })
