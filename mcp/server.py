"""
Titan MCP Server - JSON-RPC server implementing MCP protocol.

Provides:
- Tool: spawn_agent - Create and run agents
- Tool: agent_status - Check agent state
- Tool: list_agents - List active agents
- Tool: agent_result - Get agent results
- Resource: agent_types - Available agent archetypes
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger("titan.mcp")


# ============================================================================
# MCP Protocol Types
# ============================================================================

class MCPMethod(str, Enum):
    """MCP JSON-RPC methods."""

    # Lifecycle
    INITIALIZE = "initialize"
    INITIALIZED = "notifications/initialized"
    SHUTDOWN = "shutdown"

    # Tools
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"

    # Resources
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"

    # Prompts
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"


@dataclass
class MCPRequest:
    """Incoming MCP request."""

    jsonrpc: str
    id: int | str | None
    method: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPResponse:
    """Outgoing MCP response."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: Any = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d


@dataclass
class MCPTool:
    """MCP tool definition."""

    name: str
    description: str
    inputSchema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema,
        }


@dataclass
class MCPResource:
    """MCP resource definition."""

    uri: str
    name: str
    description: str
    mimeType: str = "application/json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mimeType,
        }


# ============================================================================
# Agent Session Management
# ============================================================================

@dataclass
class AgentSession:
    """Active agent session."""

    id: str
    agent_type: str
    task: str
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    result: Any = None
    error: str | None = None


class AgentManager:
    """Manages agent sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    async def spawn_agent(
        self,
        agent_type: str,
        task: str,
        **kwargs: Any,
    ) -> str:
        """Spawn a new agent and return session ID."""
        session_id = f"sess_{uuid.uuid4().hex[:8]}"

        session = AgentSession(
            id=session_id,
            agent_type=agent_type,
            task=task,
            status="running",
        )
        self._sessions[session_id] = session

        # Start agent in background
        agent_task = asyncio.create_task(
            self._run_agent(session, **kwargs)
        )
        self._tasks[session_id] = agent_task

        logger.info(f"Spawned agent {agent_type} with session {session_id}")
        return session_id

    async def _run_agent(self, session: AgentSession, **kwargs: Any) -> None:
        """Run an agent to completion."""
        try:
            # Import agent types
            from agents.archetypes.researcher import ResearcherAgent
            from agents.archetypes.coder import CoderAgent
            from agents.archetypes.reviewer import ReviewerAgent
            from agents.archetypes.orchestrator import OrchestratorAgent
            from agents.framework.tool_agent import SimpleToolAgent

            agent_classes = {
                "researcher": ResearcherAgent,
                "coder": CoderAgent,
                "reviewer": ReviewerAgent,
                "orchestrator": OrchestratorAgent,
                "simple": SimpleToolAgent,
            }

            agent_class = agent_classes.get(session.agent_type.lower())
            if not agent_class:
                session.status = "failed"
                session.error = f"Unknown agent type: {session.agent_type}"
                return

            # Create agent based on type
            if session.agent_type == "researcher":
                agent = agent_class(topic=session.task, **kwargs)
            elif session.agent_type == "coder":
                agent = agent_class(task_description=session.task, **kwargs)
            elif session.agent_type == "reviewer":
                agent = agent_class(content=session.task, **kwargs)
            elif session.agent_type == "orchestrator":
                agent = agent_class(task=session.task, **kwargs)
            else:
                agent = agent_class(task=session.task, **kwargs)

            # Run agent
            result = await agent.run()

            session.status = "completed" if result.success else "failed"
            session.result = result.result
            session.error = result.error

            logger.info(f"Agent {session.id} completed with status {session.status}")

        except Exception as e:
            logger.exception(f"Agent {session.id} failed: {e}")
            session.status = "failed"
            session.error = str(e)

    def get_session(self, session_id: str) -> AgentSession | None:
        """Get agent session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[AgentSession]:
        """List all sessions."""
        return list(self._sessions.values())

    async def cancel_session(self, session_id: str) -> bool:
        """Cancel a running session."""
        task = self._tasks.get(session_id)
        if task and not task.done():
            task.cancel()
            session = self._sessions.get(session_id)
            if session:
                session.status = "cancelled"
            return True
        return False


# ============================================================================
# MCP Server
# ============================================================================

class TitanMCPServer:
    """
    MCP Server for Titan multi-agent system.

    Exposes agents via JSON-RPC over stdin/stdout.
    """

    def __init__(self) -> None:
        self._agent_manager = AgentManager()
        self._initialized = False
        self._handlers: dict[str, Callable[..., Awaitable[Any]]] = {
            MCPMethod.INITIALIZE.value: self._handle_initialize,
            MCPMethod.SHUTDOWN.value: self._handle_shutdown,
            MCPMethod.TOOLS_LIST.value: self._handle_tools_list,
            MCPMethod.TOOLS_CALL.value: self._handle_tools_call,
            MCPMethod.RESOURCES_LIST.value: self._handle_resources_list,
            MCPMethod.RESOURCES_READ.value: self._handle_resources_read,
        }

    def get_tools(self) -> list[MCPTool]:
        """Get available MCP tools."""
        return [
            MCPTool(
                name="spawn_agent",
                description="Spawn a new Titan agent to perform a task. Returns a session ID to track progress.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_type": {
                            "type": "string",
                            "description": "Type of agent: researcher, coder, reviewer, orchestrator, or simple",
                            "enum": ["researcher", "coder", "reviewer", "orchestrator", "simple"],
                        },
                        "task": {
                            "type": "string",
                            "description": "Task description for the agent",
                        },
                    },
                    "required": ["agent_type", "task"],
                },
            ),
            MCPTool(
                name="agent_status",
                description="Check the status of an agent session.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID returned by spawn_agent",
                        },
                    },
                    "required": ["session_id"],
                },
            ),
            MCPTool(
                name="agent_result",
                description="Get the result of a completed agent session.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID returned by spawn_agent",
                        },
                    },
                    "required": ["session_id"],
                },
            ),
            MCPTool(
                name="list_agents",
                description="List all active agent sessions.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            MCPTool(
                name="cancel_agent",
                description="Cancel a running agent session.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID to cancel",
                        },
                    },
                    "required": ["session_id"],
                },
            ),
        ]

    def get_resources(self) -> list[MCPResource]:
        """Get available MCP resources."""
        return [
            MCPResource(
                uri="titan://agents/types",
                name="Agent Types",
                description="List of available agent archetypes and their capabilities",
            ),
            MCPResource(
                uri="titan://agents/tools",
                name="Agent Tools",
                description="List of tools available to agents",
            ),
        ]

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle an incoming MCP request."""
        handler = self._handlers.get(request.method)

        if not handler:
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {request.method}",
                },
            )

        try:
            result = await handler(request.params)
            return MCPResponse(id=request.id, result=result)
        except Exception as e:
            logger.exception(f"Error handling {request.method}: {e}")
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32603,
                    "message": str(e),
                },
            )

    # ========================================================================
    # Handlers
    # ========================================================================

    async def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle initialize request."""
        self._initialized = True
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "titan-mcp",
                "version": "0.1.0",
            },
            "capabilities": {
                "tools": {},
                "resources": {},
            },
        }

    async def _handle_shutdown(self, params: dict[str, Any]) -> None:
        """Handle shutdown request."""
        self._initialized = False
        return None

    async def _handle_tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/list request."""
        return {"tools": [t.to_dict() for t in self.get_tools()]}

    async def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/call request."""
        name = params.get("name")
        arguments = params.get("arguments", {})

        if name == "spawn_agent":
            session_id = await self._agent_manager.spawn_agent(
                agent_type=arguments["agent_type"],
                task=arguments["task"],
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "session_id": session_id,
                            "status": "running",
                            "message": f"Agent spawned with session {session_id}",
                        }),
                    }
                ],
            }

        elif name == "agent_status":
            session = self._agent_manager.get_session(arguments["session_id"])
            if not session:
                return {
                    "content": [{"type": "text", "text": "Session not found"}],
                    "isError": True,
                }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "session_id": session.id,
                            "agent_type": session.agent_type,
                            "status": session.status,
                            "created_at": session.created_at.isoformat(),
                            "error": session.error,
                        }),
                    }
                ],
            }

        elif name == "agent_result":
            session = self._agent_manager.get_session(arguments["session_id"])
            if not session:
                return {
                    "content": [{"type": "text", "text": "Session not found"}],
                    "isError": True,
                }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "session_id": session.id,
                            "status": session.status,
                            "result": session.result,
                            "error": session.error,
                        }, default=str),
                    }
                ],
            }

        elif name == "list_agents":
            sessions = self._agent_manager.list_sessions()
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps([
                            {
                                "session_id": s.id,
                                "agent_type": s.agent_type,
                                "status": s.status,
                                "task": s.task[:100],
                            }
                            for s in sessions
                        ]),
                    }
                ],
            }

        elif name == "cancel_agent":
            success = await self._agent_manager.cancel_session(arguments["session_id"])
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "cancelled": success,
                            "session_id": arguments["session_id"],
                        }),
                    }
                ],
            }

        else:
            return {
                "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                "isError": True,
            }

    async def _handle_resources_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle resources/list request."""
        return {"resources": [r.to_dict() for r in self.get_resources()]}

    async def _handle_resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri")

        if uri == "titan://agents/types":
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps({
                            "agent_types": [
                                {
                                    "name": "researcher",
                                    "description": "Research and analyze information on topics",
                                    "capabilities": ["web_search", "document_analysis", "summarization"],
                                },
                                {
                                    "name": "coder",
                                    "description": "Write, test, and review code",
                                    "capabilities": ["code_generation", "code_review", "testing"],
                                },
                                {
                                    "name": "reviewer",
                                    "description": "Review code and documents for quality",
                                    "capabilities": ["code_review", "document_analysis"],
                                },
                                {
                                    "name": "orchestrator",
                                    "description": "Coordinate multi-agent workflows",
                                    "capabilities": ["planning", "coordination", "aggregation"],
                                },
                                {
                                    "name": "simple",
                                    "description": "Simple tool-using agent",
                                    "capabilities": ["tool_use"],
                                },
                            ],
                        }, indent=2),
                    }
                ],
            }

        elif uri == "titan://agents/tools":
            from tools.base import get_registry
            registry = get_registry()
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps({
                            "tools": [
                                {
                                    "name": t.name,
                                    "description": t.description,
                                }
                                for t in registry.list()
                            ],
                        }, indent=2),
                    }
                ],
            }

        return {
            "contents": [{"uri": uri, "mimeType": "text/plain", "text": "Resource not found"}],
        }

    async def run_stdio(self) -> None:
        """Run the server on stdin/stdout."""
        logger.info("Titan MCP Server starting on stdio...")

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(
            lambda: protocol, sys.stdin
        )

        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, asyncio.get_event_loop())

        try:
            while True:
                # Read line (JSON-RPC message)
                line = await reader.readline()
                if not line:
                    break

                try:
                    data = json.loads(line.decode())
                    request = MCPRequest(
                        jsonrpc=data.get("jsonrpc", "2.0"),
                        id=data.get("id"),
                        method=data.get("method", ""),
                        params=data.get("params", {}),
                    )

                    response = await self.handle_request(request)

                    # Write response
                    response_json = json.dumps(response.to_dict()) + "\n"
                    writer.write(response_json.encode())
                    await writer.drain()

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                except Exception as e:
                    logger.exception(f"Error processing message: {e}")

        except asyncio.CancelledError:
            pass
        finally:
            logger.info("Titan MCP Server shutting down")


def create_server() -> TitanMCPServer:
    """Create a new MCP server instance."""
    return TitanMCPServer()


def run_server() -> None:
    """Run the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    server = create_server()
    asyncio.run(server.run_stdio())


if __name__ == "__main__":
    run_server()
