"""
Titan Dashboard - Pydantic Models

Request and response models for the REST API.
Enables automatic OpenAPI documentation generation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================


class TopologyTypeEnum(str, Enum):
    """Supported topology types."""

    SWARM = "swarm"
    HIERARCHY = "hierarchy"
    PIPELINE = "pipeline"
    MESH = "mesh"
    RING = "ring"
    STAR = "star"


class AgentStateEnum(str, Enum):
    """Agent execution states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================================
# Common Models
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str = Field(..., description="Error message")
    code: str | None = Field(None, description="Error code for programmatic handling")

    model_config = {"json_schema_extra": {"example": {"detail": "Agent not found", "code": "AGENT_NOT_FOUND"}}}


# ============================================================================
# Status Models
# ============================================================================


class StatusResponse(BaseModel):
    """System status response."""

    status: str = Field(..., description="System health status")
    active_agents: int = Field(..., description="Number of currently active agents")
    current_topology: TopologyTypeEnum = Field(..., description="Current topology type")
    timestamp: datetime = Field(..., description="Response timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "active_agents": 3,
                "current_topology": "swarm",
                "timestamp": "2024-01-19T12:00:00Z",
            }
        }
    }


# ============================================================================
# Agent Models
# ============================================================================


class AgentCapability(BaseModel):
    """Agent capability description."""

    name: str = Field(..., description="Capability name")
    description: str | None = Field(None, description="Capability description")


class AgentResponse(BaseModel):
    """Agent information response."""

    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent display name")
    role: str = Field(..., description="Agent role in current topology")
    state: AgentStateEnum = Field(..., description="Current agent state")
    joined_at: datetime = Field(..., description="When agent joined the swarm")
    capabilities: list[str] = Field(default_factory=list, description="Agent capabilities")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "agent_001",
                "name": "researcher",
                "role": "peer",
                "state": "running",
                "joined_at": "2024-01-19T12:00:00Z",
                "capabilities": ["web_search", "summarization"],
            }
        }
    }


class AgentCancelResponse(BaseModel):
    """Agent cancellation response."""

    status: str = Field(..., description="Cancellation status")
    agent_id: str = Field(..., description="ID of cancelled agent")

    model_config = {"json_schema_extra": {"example": {"status": "cancelled", "agent_id": "agent_001"}}}


# ============================================================================
# Topology Models
# ============================================================================


class TopologyAgentInfo(BaseModel):
    """Agent information within topology context."""

    id: str = Field(..., description="Agent ID")
    role: str = Field(..., description="Agent role in topology")


class TopologyHistoryEntry(BaseModel):
    """Topology history entry."""

    from_type: TopologyTypeEnum | None = Field(None, alias="from", description="Previous topology type")
    to_type: TopologyTypeEnum = Field(..., alias="to", description="New topology type")
    timestamp: datetime = Field(..., description="When the switch occurred")


class TopologyResponse(BaseModel):
    """Current topology state response."""

    current: TopologyTypeEnum = Field(..., description="Current topology type")
    agents: list[TopologyAgentInfo] = Field(default_factory=list, description="Agents in topology")
    history: list[TopologyHistoryEntry] = Field(default_factory=list, description="Recent topology history")

    model_config = {
        "json_schema_extra": {
            "example": {
                "current": "swarm",
                "agents": [{"id": "agent_001", "role": "peer"}, {"id": "agent_002", "role": "peer"}],
                "history": [{"from": "pipeline", "to": "swarm", "timestamp": "2024-01-19T12:00:00Z"}],
            }
        }
    }


class TopologySwitchResponse(BaseModel):
    """Topology switch response."""

    status: str = Field(..., description="Switch status")
    new_topology: TopologyTypeEnum = Field(..., description="New topology type")
    agent_count: int = Field(0, description="Number of agents migrated")
    duration_ms: float = Field(0.0, description="Switch duration in milliseconds")

    model_config = {
        "json_schema_extra": {
            "example": {"status": "success", "new_topology": "pipeline", "agent_count": 3, "duration_ms": 42.5}
        }
    }


# ============================================================================
# Event Models
# ============================================================================


class EventResponse(BaseModel):
    """Event history entry."""

    type: str = Field(..., description="Event type")
    timestamp: datetime = Field(..., description="Event timestamp")
    payload: dict[str, Any] = Field(default_factory=dict, description="Event payload data")

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "agent_joined",
                "timestamp": "2024-01-19T12:00:00Z",
                "payload": {"agent_id": "agent_001", "name": "researcher"},
            }
        }
    }


# ============================================================================
# Learning Models
# ============================================================================


class TopologyStatsEntry(BaseModel):
    """Statistics for a single topology type."""

    episodes: int = Field(..., description="Number of episodes with this topology")
    avg_score: float = Field(..., description="Average episode score")
    success_rate: float = Field(..., description="Success rate (0-1)")


class LearningStatsResponse(BaseModel):
    """Learning statistics response."""

    total_episodes: int = Field(..., description="Total recorded episodes")
    topologies: dict[str, TopologyStatsEntry] = Field(
        default_factory=dict, description="Statistics per topology type"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "total_episodes": 42,
                "topologies": {
                    "swarm": {"episodes": 20, "avg_score": 0.75, "success_rate": 0.85},
                    "pipeline": {"episodes": 22, "avg_score": 0.82, "success_rate": 0.91},
                },
            }
        }
    }


# ============================================================================
# WebSocket Models (for documentation)
# ============================================================================


class WebSocketMessage(BaseModel):
    """Base WebSocket message structure."""

    type: str = Field(..., description="Message type")


class WebSocketPing(WebSocketMessage):
    """Client ping message."""

    type: str = Field("ping", description="Message type")


class WebSocketPong(WebSocketMessage):
    """Server pong response."""

    type: str = Field("pong", description="Message type")


class WebSocketSubscribe(WebSocketMessage):
    """Client subscription request."""

    type: str = Field("subscribe", description="Message type")
    topics: list[str] = Field(..., description="Topics to subscribe to")


class WebSocketAgentJoined(WebSocketMessage):
    """Agent joined notification."""

    type: str = Field("agent_joined", description="Message type")
    agent_id: str = Field(..., description="ID of the joined agent")
    name: str = Field(..., description="Name of the joined agent")


class WebSocketAgentLeft(WebSocketMessage):
    """Agent left notification."""

    type: str = Field("agent_left", description="Message type")
    agent_id: str = Field(..., description="ID of the departed agent")
    reason: str | None = Field(None, description="Reason for departure")


class WebSocketTopologyChanged(WebSocketMessage):
    """Topology change notification."""

    type: str = Field("topology_changed", description="Message type")
    old_type: TopologyTypeEnum | None = Field(None, description="Previous topology")
    new_type: TopologyTypeEnum = Field(..., description="New topology")
