"""
Agentic Titan - Base Agent

Abstract base class for all agents in the Titan swarm. Provides:
- Lifecycle management (initialize -> work -> shutdown)
- Hive Mind integration for shared memory
- Topology-aware communication
- Built-in resilience patterns

Ported from: metasystem-core/agent_utils/base_agent.py
Extended with: Hive Mind integration, topology awareness, async support
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from agents.framework.errors import AgentError, TitanError

if TYPE_CHECKING:
    from hive.memory import HiveMind
    from hive.topology import TopologyEngine


logger = logging.getLogger("titan.agent")


class AgentState(Enum):
    """Agent lifecycle states."""

    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    WORKING = "working"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentContext:
    """
    Context passed to agent during work.

    Contains information from the Hive Mind about past decisions,
    similar work, and shared knowledge.
    """

    agent_id: str
    session_id: str
    topology_role: str | None = None
    parent_agent_id: str | None = None
    child_agent_ids: list[str] = field(default_factory=list)

    # Knowledge from Hive Mind
    past_decisions: list[dict[str, Any]] = field(default_factory=list)
    similar_work: list[dict[str, Any]] = field(default_factory=list)
    shared_memory: dict[str, Any] = field(default_factory=dict)

    # Runtime metadata
    turn_number: int = 0
    max_turns: int = 20
    start_time: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return (
            f"AgentContext(id={self.agent_id}, "
            f"decisions={len(self.past_decisions)}, "
            f"turn={self.turn_number}/{self.max_turns})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "topology_role": self.topology_role,
            "parent_agent_id": self.parent_agent_id,
            "child_agent_ids": self.child_agent_ids,
            "turn_number": self.turn_number,
            "max_turns": self.max_turns,
            "start_time": self.start_time.isoformat(),
        }


@dataclass
class AgentResult:
    """Result of agent execution."""

    agent_id: str
    session_id: str
    state: AgentState
    result: Any = None
    error: str | None = None
    turns_taken: int = 0
    execution_time_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.state == AgentState.COMPLETED and self.error is None


class BaseAgent(ABC):
    """
    Abstract base class for agents integrated with the Titan swarm.

    Subclasses must implement:
    - initialize() - Setup before work
    - work() - Main work loop
    - shutdown() - Cleanup after work

    Available methods:
    - remember() - Store in Hive Mind
    - recall() - Retrieve from Hive Mind
    - broadcast() - Send to all connected agents
    - send() - Send to specific agent
    - spawn_child() - Spawn a child agent
    """

    def __init__(
        self,
        agent_id: str | None = None,
        name: str = "unnamed",
        capabilities: list[str] | None = None,
        *,
        hive_mind: HiveMind | None = None,
        topology_engine: TopologyEngine | None = None,
        max_turns: int = 20,
        timeout_ms: int = 300_000,
    ) -> None:
        """
        Initialize agent.

        Args:
            agent_id: Unique identifier (auto-generated if not provided)
            name: Human-readable name
            capabilities: List of capabilities this agent has
            hive_mind: Shared Hive Mind instance
            topology_engine: Topology engine for routing
            max_turns: Maximum execution turns
            timeout_ms: Execution timeout in milliseconds
        """
        self.agent_id = agent_id or f"{name}-{uuid.uuid4().hex[:8]}"
        self.name = name
        self.capabilities = capabilities or []

        # External dependencies (injected)
        self._hive_mind = hive_mind
        self._topology_engine = topology_engine

        # Configuration
        self.max_turns = max_turns
        self.timeout_ms = timeout_ms

        # State
        self._state = AgentState.CREATED
        self._context: AgentContext | None = None
        self._session_id: str | None = None
        self._decisions_logged: list[dict[str, Any]] = []
        self._last_error: Exception | None = None

        # Event handlers
        self._on_state_change: list[Callable[[AgentState, AgentState], None]] = []

        logger.info(f"Agent '{self.name}' ({self.agent_id}) created")

    @property
    def state(self) -> AgentState:
        """Current agent state."""
        return self._state

    @state.setter
    def state(self, new_state: AgentState) -> None:
        """Set state with change notification."""
        old_state = self._state
        self._state = new_state
        for handler in self._on_state_change:
            try:
                handler(old_state, new_state)
            except Exception as e:
                logger.warning(f"State change handler error: {e}")

    @property
    def context(self) -> AgentContext | None:
        """Current execution context."""
        return self._context

    def on_state_change(self, handler: Callable[[AgentState, AgentState], None]) -> None:
        """Register a state change handler."""
        self._on_state_change.append(handler)

    # =========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize agent before work.

        Override this to:
        - Load configuration
        - Get initial context from Hive Mind
        - Setup internal state
        """
        pass

    @abstractmethod
    async def work(self) -> Any:
        """
        Main work loop.

        Override this with agent's actual work logic.
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Cleanup after work.

        Override this to:
        - Save state to Hive Mind
        - Close resources
        - Final logging
        """
        pass

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def run(self, prompt: str | None = None) -> AgentResult:
        """
        Execute agent lifecycle: initialize -> work -> shutdown.

        Args:
            prompt: Optional initial prompt/task

        Returns:
            AgentResult with execution details
        """
        start_time = datetime.now()
        self._session_id = uuid.uuid4().hex

        self._context = AgentContext(
            agent_id=self.agent_id,
            session_id=self._session_id,
            max_turns=self.max_turns,
        )

        try:
            # Initialize
            self.state = AgentState.INITIALIZING
            logger.info(f"Agent '{self.name}' initializing...")
            await asyncio.wait_for(
                self.initialize(),
                timeout=self.timeout_ms / 1000,
            )

            # Work
            self.state = AgentState.READY
            self.state = AgentState.WORKING
            logger.info(f"Agent '{self.name}' working...")
            result = await asyncio.wait_for(
                self.work(),
                timeout=self.timeout_ms / 1000,
            )

            # Complete
            self.state = AgentState.COMPLETED
            logger.info(
                f"Agent '{self.name}' completed. "
                f"Logged {len(self._decisions_logged)} decisions"
            )

            # Shutdown
            await self.shutdown()

            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            return AgentResult(
                agent_id=self.agent_id,
                session_id=self._session_id,
                state=AgentState.COMPLETED,
                result=result,
                turns_taken=self._context.turn_number,
                execution_time_ms=execution_time_ms,
                metadata={"decisions_logged": len(self._decisions_logged)},
            )

        except asyncio.TimeoutError:
            self.state = AgentState.FAILED
            self._last_error = AgentError(
                f"Agent timed out after {self.timeout_ms}ms",
                agent_id=self.agent_id,
            )
            await self._safe_shutdown()

            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            return AgentResult(
                agent_id=self.agent_id,
                session_id=self._session_id or "no-session",
                state=AgentState.FAILED,
                error=f"Timeout after {self.timeout_ms}ms",
                turns_taken=self._context.turn_number if self._context else 0,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            self.state = AgentState.FAILED
            self._last_error = e
            logger.exception(f"Agent '{self.name}' failed: {e}")
            await self._safe_shutdown()

            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            return AgentResult(
                agent_id=self.agent_id,
                session_id=self._session_id or "no-session",
                state=AgentState.FAILED,
                error=str(e),
                turns_taken=self._context.turn_number if self._context else 0,
                execution_time_ms=execution_time_ms,
            )

    async def _safe_shutdown(self) -> None:
        """Attempt shutdown even after errors."""
        try:
            await asyncio.wait_for(self.shutdown(), timeout=5.0)
        except Exception as e:
            logger.warning(f"Shutdown error (ignored): {e}")

    def pause(self) -> None:
        """Pause agent execution."""
        if self.state == AgentState.WORKING:
            self.state = AgentState.PAUSED
            logger.info(f"Agent '{self.name}' paused")

    def resume(self) -> None:
        """Resume paused agent."""
        if self.state == AgentState.PAUSED:
            self.state = AgentState.WORKING
            logger.info(f"Agent '{self.name}' resumed")

    def cancel(self) -> None:
        """Cancel agent execution."""
        if self.state in (AgentState.WORKING, AgentState.PAUSED, AgentState.INITIALIZING):
            self.state = AgentState.CANCELLED
            logger.info(f"Agent '{self.name}' cancelled")

    # =========================================================================
    # Hive Mind Integration
    # =========================================================================

    async def remember(
        self,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Store content in the Hive Mind's long-term memory.

        Args:
            content: Content to remember
            importance: Importance score (0.0 to 1.0)
            tags: Optional tags for categorization
            metadata: Optional metadata

        Returns:
            Memory ID if successful, None otherwise
        """
        if not self._hive_mind:
            logger.warning("Hive Mind not available, cannot remember")
            return None

        try:
            memory_id = await self._hive_mind.remember(
                agent_id=self.agent_id,
                content=content,
                importance=importance,
                tags=tags or [],
                metadata=metadata or {},
            )
            logger.debug(f"Remembered content: {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Failed to remember: {e}")
            return None

    async def recall(
        self,
        query: str,
        k: int = 10,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Recall memories from the Hive Mind.

        Args:
            query: Search query (semantic)
            k: Number of results to return
            tags: Optional tag filter

        Returns:
            List of matching memories
        """
        if not self._hive_mind:
            logger.warning("Hive Mind not available, cannot recall")
            return []

        try:
            memories = await self._hive_mind.recall(
                query=query,
                k=k,
                tags=tags,
            )
            logger.debug(f"Recalled {len(memories)} memories")
            return memories
        except Exception as e:
            logger.error(f"Failed to recall: {e}")
            return []

    async def log_decision(
        self,
        decision: str,
        category: str,
        rationale: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """
        Log a decision to the Hive Mind.

        Args:
            decision: Decision description
            category: Decision type (architecture, design, implementation, etc.)
            rationale: Why this decision was made
            tags: Optional tags
        """
        decision_record = {
            "agent_id": self.agent_id,
            "decision": decision,
            "category": category,
            "rationale": rationale,
            "tags": tags or [],
            "timestamp": datetime.now().isoformat(),
        }
        self._decisions_logged.append(decision_record)

        # Store in Hive Mind
        await self.remember(
            content=f"[{category}] {decision}\nRationale: {rationale}",
            importance=0.7,
            tags=["decision", category] + (tags or []),
            metadata=decision_record,
        )

        logger.info(f"Decision logged: {decision[:50]}...")

    # =========================================================================
    # Inter-Agent Communication
    # =========================================================================

    async def broadcast(
        self,
        message: dict[str, Any],
        topic: str = "general",
    ) -> None:
        """
        Broadcast message to all agents in the current topology.

        Args:
            message: Message content
            topic: Message topic for filtering
        """
        if not self._hive_mind:
            logger.warning("Hive Mind not available, cannot broadcast")
            return

        try:
            await self._hive_mind.broadcast(
                source_agent_id=self.agent_id,
                message=message,
                topic=topic,
            )
            logger.debug(f"Broadcast to topic '{topic}'")
        except Exception as e:
            logger.error(f"Failed to broadcast: {e}")

    async def send(
        self,
        target_agent_id: str,
        message: dict[str, Any],
    ) -> bool:
        """
        Send a direct message to another agent.

        Args:
            target_agent_id: Target agent ID
            message: Message content

        Returns:
            True if sent successfully
        """
        if not self._hive_mind:
            logger.warning("Hive Mind not available, cannot send")
            return False

        try:
            await self._hive_mind.send(
                source_agent_id=self.agent_id,
                target_agent_id=target_agent_id,
                message=message,
            )
            logger.debug(f"Sent message to {target_agent_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send: {e}")
            return False

    async def subscribe(
        self,
        topic: str,
        handler: Callable[[dict[str, Any]], None],
    ) -> None:
        """
        Subscribe to messages on a topic.

        Args:
            topic: Topic to subscribe to
            handler: Callback for received messages
        """
        if not self._hive_mind:
            logger.warning("Hive Mind not available, cannot subscribe")
            return

        try:
            await self._hive_mind.subscribe(
                agent_id=self.agent_id,
                topic=topic,
                handler=handler,
            )
            logger.debug(f"Subscribed to topic '{topic}'")
        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def increment_turn(self) -> int:
        """Increment turn counter and return new value."""
        if self._context:
            self._context.turn_number += 1
            return self._context.turn_number
        return 0

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} '{self.name}' id={self.agent_id} state={self.state.value}>"

    def __enter__(self) -> BaseAgent:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        pass
