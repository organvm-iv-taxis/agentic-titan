"""
Titan Dashboard - FastAPI Application

Provides a web interface for managing and monitoring the agent swarm.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

logger = logging.getLogger("titan.dashboard")

# Try to import FastAPI, provide helpful error if missing
try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed. Install with: pip install fastapi uvicorn jinja2")


# Get package directory for templates
PACKAGE_DIR = Path(__file__).parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)


class TitanDashboard:
    """
    Titan Dashboard application.

    Manages the FastAPI application and integrates with Titan components.
    """

    def __init__(
        self,
        hive_mind: Any | None = None,
        topology_engine: Any | None = None,
        event_bus: Any | None = None,
    ) -> None:
        """
        Initialize dashboard.

        Args:
            hive_mind: HiveMind instance for memory access
            topology_engine: TopologyEngine instance
            event_bus: EventBus instance for real-time updates
        """
        self.hive_mind = hive_mind
        self.topology_engine = topology_engine
        self.event_bus = event_bus

        self.app: FastAPI | None = None
        self.manager = ConnectionManager()

        # State tracking
        self._active_agents: dict[str, dict[str, Any]] = {}
        self._task_history: list[dict[str, Any]] = []
        self._topology_history: list[dict[str, Any]] = []

    def create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI not installed. Install with: pip install fastapi uvicorn jinja2"
            )

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            """Application lifespan handler."""
            logger.info("Dashboard starting...")
            # Subscribe to events if event bus is available
            if self.event_bus:
                await self._setup_event_handlers()
            yield
            logger.info("Dashboard shutting down...")

        self.app = FastAPI(
            title="Titan Dashboard",
            description="Web dashboard for Agentic Titan swarm management",
            version="1.0.0",
            lifespan=lifespan,
        )

        # Mount static files
        if STATIC_DIR.exists():
            self.app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        # Setup templates
        templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

        # Register routes
        self._register_routes(templates)

        return self.app

    def _register_routes(self, templates: Any) -> None:
        """Register all routes."""
        if not self.app:
            return

        # ====================================================================
        # HTML Routes
        # ====================================================================

        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request) -> HTMLResponse:
            """Dashboard home page."""
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "title": "Titan Dashboard",
                    "active_agents": len(self._active_agents),
                    "current_topology": self._get_current_topology(),
                },
            )

        @self.app.get("/agents", response_class=HTMLResponse)
        async def agents_page(request: Request) -> HTMLResponse:
            """Agents management page."""
            return templates.TemplateResponse(
                "agents.html",
                {
                    "request": request,
                    "title": "Agents",
                    "agents": list(self._active_agents.values()),
                },
            )

        @self.app.get("/topology", response_class=HTMLResponse)
        async def topology_page(request: Request) -> HTMLResponse:
            """Topology visualization page."""
            return templates.TemplateResponse(
                "topology.html",
                {
                    "request": request,
                    "title": "Topology",
                    "current": self._get_current_topology(),
                    "history": self._topology_history[-20:],
                },
            )

        # ====================================================================
        # API Routes
        # ====================================================================

        @self.app.get("/api/status")
        async def get_status() -> dict[str, Any]:
            """Get system status."""
            return {
                "status": "healthy",
                "active_agents": len(self._active_agents),
                "current_topology": self._get_current_topology(),
                "timestamp": datetime.now().isoformat(),
            }

        @self.app.get("/api/agents")
        async def get_agents() -> list[dict[str, Any]]:
            """Get all active agents."""
            return list(self._active_agents.values())

        @self.app.get("/api/agents/{agent_id}")
        async def get_agent(agent_id: str) -> dict[str, Any]:
            """Get specific agent details."""
            if agent_id not in self._active_agents:
                raise HTTPException(status_code=404, detail="Agent not found")
            return self._active_agents[agent_id]

        @self.app.post("/api/agents/{agent_id}/cancel")
        async def cancel_agent(agent_id: str) -> dict[str, str]:
            """Cancel an agent."""
            if agent_id not in self._active_agents:
                raise HTTPException(status_code=404, detail="Agent not found")
            # In a real implementation, this would cancel the agent
            return {"status": "cancelled", "agent_id": agent_id}

        @self.app.get("/api/topology")
        async def get_topology() -> dict[str, Any]:
            """Get current topology."""
            topology = self._get_current_topology()
            return {
                "current": topology,
                "agents": [
                    {"id": a["id"], "role": a.get("role", "worker")}
                    for a in self._active_agents.values()
                ],
                "history": self._topology_history[-10:],
            }

        @self.app.post("/api/topology/switch/{topology_type}")
        async def switch_topology(topology_type: str) -> dict[str, Any]:
            """Switch to a different topology."""
            valid_types = ["swarm", "hierarchy", "pipeline", "mesh", "ring", "star"]
            if topology_type not in valid_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid topology type. Valid types: {valid_types}",
                )

            if self.topology_engine:
                try:
                    await self.topology_engine.switch_topology(topology_type)
                    return {"status": "success", "new_topology": topology_type}
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            else:
                # Mock response
                self._topology_history.append({
                    "from": self._get_current_topology(),
                    "to": topology_type,
                    "timestamp": datetime.now().isoformat(),
                })
                return {"status": "success", "new_topology": topology_type}

        @self.app.get("/api/events")
        async def get_events(limit: int = 50) -> list[dict[str, Any]]:
            """Get recent events."""
            if self.event_bus:
                events = self.event_bus.get_history(limit=limit)
                return [e.to_dict() for e in events]
            return []

        @self.app.get("/api/learning/stats")
        async def get_learning_stats() -> dict[str, Any]:
            """Get learning statistics."""
            # This would integrate with the EpisodicLearner
            return {
                "total_episodes": 0,
                "topologies": {},
                "message": "Integrate with EpisodicLearner for real stats",
            }

        @self.app.get("/api/metrics")
        async def get_metrics() -> str:
            """Get Prometheus metrics."""
            try:
                from titan.metrics import get_metrics_text
                return get_metrics_text()
            except Exception as e:
                return f"# Error getting metrics: {e}"

        # ====================================================================
        # WebSocket Route
        # ====================================================================

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            """WebSocket for real-time updates."""
            await self.manager.connect(websocket)
            try:
                while True:
                    # Keep connection alive and handle incoming messages
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    # Handle different message types
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif message.get("type") == "subscribe":
                        # Handle subscription requests
                        pass

            except WebSocketDisconnect:
                self.manager.disconnect(websocket)

    async def _setup_event_handlers(self) -> None:
        """Setup event handlers for real-time updates."""
        if not self.event_bus:
            return

        from hive.events import EventType

        async def on_agent_joined(event: Any) -> None:
            agent_data = event.payload
            self._active_agents[agent_data["agent_id"]] = {
                "id": agent_data["agent_id"],
                "name": agent_data.get("name", "unknown"),
                "state": "running",
                "joined_at": event.timestamp.isoformat(),
            }
            await self.manager.broadcast({
                "type": "agent_joined",
                "data": agent_data,
            })

        async def on_agent_left(event: Any) -> None:
            agent_id = event.payload.get("agent_id")
            if agent_id in self._active_agents:
                del self._active_agents[agent_id]
            await self.manager.broadcast({
                "type": "agent_left",
                "data": event.payload,
            })

        async def on_topology_changed(event: Any) -> None:
            self._topology_history.append({
                "from": event.payload.get("old_type"),
                "to": event.payload.get("new_type"),
                "timestamp": event.timestamp.isoformat(),
            })
            await self.manager.broadcast({
                "type": "topology_changed",
                "data": event.payload,
            })

        self.event_bus.subscribe(EventType.AGENT_JOINED, on_agent_joined)
        self.event_bus.subscribe(EventType.AGENT_LEFT, on_agent_left)
        self.event_bus.subscribe(EventType.TOPOLOGY_CHANGED, on_topology_changed)

    def _get_current_topology(self) -> str:
        """Get current topology type."""
        if self.topology_engine and self.topology_engine.current_topology:
            return self.topology_engine.current_topology.topology_type.value
        if self._topology_history:
            return self._topology_history[-1]["to"]
        return "swarm"

    def register_agent(self, agent_id: str, name: str, role: str = "worker") -> None:
        """Register an agent with the dashboard."""
        self._active_agents[agent_id] = {
            "id": agent_id,
            "name": name,
            "role": role,
            "state": "running",
            "joined_at": datetime.now().isoformat(),
        }

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the dashboard."""
        if agent_id in self._active_agents:
            del self._active_agents[agent_id]


def create_app(
    hive_mind: Any | None = None,
    topology_engine: Any | None = None,
    event_bus: Any | None = None,
) -> FastAPI:
    """
    Create and return the FastAPI application.

    Args:
        hive_mind: HiveMind instance
        topology_engine: TopologyEngine instance
        event_bus: EventBus instance

    Returns:
        Configured FastAPI application
    """
    dashboard = TitanDashboard(
        hive_mind=hive_mind,
        topology_engine=topology_engine,
        event_bus=event_bus,
    )
    return dashboard.create_app()


def run_dashboard(
    host: str = "0.0.0.0",
    port: int = 8080,
    reload: bool = False,
    **kwargs: Any,
) -> None:
    """
    Run the dashboard server.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
        **kwargs: Additional uvicorn options
    """
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn")
        return

    app = create_app()
    uvicorn.run(app, host=host, port=port, reload=reload, **kwargs)
