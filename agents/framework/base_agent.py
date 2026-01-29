"""
Agentic Titan - Base Agent

Abstract base class for all agents in the Titan swarm. Provides:
- Lifecycle management (initialize -> work -> shutdown)
- Hive Mind integration for shared memory
- Topology-aware communication
- Built-in resilience patterns
- PostgreSQL audit logging
- Explicit stopping conditions (Anthropic best practice)
- Checkpointing for recovery

Ported from: metasystem-core/agent_utils/base_agent.py
Extended with: Hive Mind integration, topology awareness, async support, audit logging

Based on research:
- Anthropic: Explicit stopping conditions and human checkpoints
- "Ground truth" environmental feedback at each step
- Uncertainty tracking for agent decisions
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
from titan.metrics import get_metrics

if TYPE_CHECKING:
    from hive.memory import HiveMind
    from hive.topology import TopologyEngine
    from titan.persistence.audit import AuditLogger


logger = logging.getLogger("titan.agent")


# =============================================================================
# Stopping Conditions
# =============================================================================


class StoppingReason(str, Enum):
    """Reasons for stopping agent execution."""

    SUCCESS = "success"  # Task completed successfully
    FAILURE = "failure"  # Task failed (unrecoverable)
    MAX_TURNS = "max_turns"  # Reached maximum turns
    TIMEOUT = "timeout"  # Execution timeout
    BUDGET_EXHAUSTED = "budget_exhausted"  # No budget remaining
    USER_CANCELLED = "user_cancelled"  # User requested stop
    CHECKPOINT_REQUIRED = "checkpoint_required"  # Human review needed
    STUCK_DETECTED = "stuck_detected"  # Agent appears stuck
    ERROR_THRESHOLD = "error_threshold"  # Too many errors


@dataclass
class StoppingCondition:
    """
    A condition that determines when an agent should stop.

    Based on Anthropic's guidance: "Implement explicit stopping conditions."
    """

    reason: StoppingReason
    check: Callable[["BaseAgent"], bool]
    message: str = ""
    priority: int = 0  # Higher priority checked first

    def evaluate(self, agent: "BaseAgent") -> bool:
        """Evaluate if this stopping condition is met."""
        try:
            return self.check(agent)
        except Exception as e:
            logger.warning(f"Stopping condition check failed: {e}")
            return False


@dataclass
class AgentCheckpoint:
    """
    A checkpoint of agent state for recovery or review.

    Checkpoints enable:
    - Recovery from failures
    - Human-in-the-loop review
    - Audit trail of agent progress
    """

    agent_id: str
    session_id: str
    turn_number: int
    timestamp: datetime = field(default_factory=datetime.now)
    state: dict[str, Any] = field(default_factory=dict)
    decisions_made: list[dict[str, Any]] = field(default_factory=list)
    memory_snapshot: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "timestamp": self.timestamp.isoformat(),
            "state": self.state,
            "decisions_made": self.decisions_made,
            "memory_snapshot": self.memory_snapshot,
            "metadata": self.metadata,
        }


@dataclass
class AgentDecision:
    """
    A tracked decision made by an agent.

    Based on Anthropic's guidance for uncertainty tracking.
    """

    choice: str
    confidence: float  # 0-1
    uncertainty_bounds: tuple[float, float] = (0.0, 1.0)  # Low, high
    alternatives: list[str] = field(default_factory=list)
    rationale: str = ""
    category: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_high_confidence(self) -> bool:
        """Whether this is a high-confidence decision."""
        return self.confidence >= 0.8

    @property
    def uncertainty_range(self) -> float:
        """The range of uncertainty."""
        return self.uncertainty_bounds[1] - self.uncertainty_bounds[0]


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
        audit_logger: AuditLogger | None = None,
        max_turns: int = 20,
        timeout_ms: int = 300_000,
        checkpoint_interval: int = 5,
        error_threshold: int = 3,
        stopping_conditions: list[StoppingCondition] | None = None,
    ) -> None:
        """
        Initialize agent.

        Args:
            agent_id: Unique identifier (auto-generated if not provided)
            name: Human-readable name
            capabilities: List of capabilities this agent has
            hive_mind: Shared Hive Mind instance
            topology_engine: Topology engine for routing
            audit_logger: Audit logger for persistent logging
            max_turns: Maximum execution turns
            timeout_ms: Execution timeout in milliseconds
            checkpoint_interval: Turns between automatic checkpoints
            error_threshold: Number of consecutive errors before stopping
            stopping_conditions: Custom stopping conditions
        """
        self.agent_id = agent_id or f"{name}-{uuid.uuid4().hex[:8]}"
        self.name = name
        self.capabilities = capabilities or []

        # External dependencies (injected)
        self._hive_mind = hive_mind
        self._topology_engine = topology_engine
        self._audit_logger = audit_logger

        # Configuration
        self.max_turns = max_turns
        self.timeout_ms = timeout_ms
        self.checkpoint_interval = checkpoint_interval
        self.error_threshold = error_threshold

        # Stopping conditions (with defaults)
        self._stopping_conditions = stopping_conditions or []
        self._add_default_stopping_conditions()

        # State
        self._state = AgentState.CREATED
        self._context: AgentContext | None = None
        self._session_id: str | None = None
        self._decisions_logged: list[dict[str, Any]] = []
        self._tracked_decisions: list[AgentDecision] = []
        self._last_error: Exception | None = None
        self._consecutive_errors: int = 0
        self._checkpoints: list[AgentCheckpoint] = []
        self._last_checkpoint_turn: int = 0

        # Event handlers
        self._on_state_change: list[Callable[[AgentState, AgentState], None]] = []
        self._on_checkpoint: list[Callable[[AgentCheckpoint], None]] = []

        logger.info(f"Agent '{self.name}' ({self.agent_id}) created")

    def _add_default_stopping_conditions(self) -> None:
        """Add default stopping conditions."""
        # Max turns condition
        self._stopping_conditions.append(StoppingCondition(
            reason=StoppingReason.MAX_TURNS,
            check=lambda agent: (
                agent._context is not None and
                agent._context.turn_number >= agent.max_turns
            ),
            message="Maximum turns reached",
            priority=10,
        ))

        # Error threshold condition
        self._stopping_conditions.append(StoppingCondition(
            reason=StoppingReason.ERROR_THRESHOLD,
            check=lambda agent: agent._consecutive_errors >= agent.error_threshold,
            message="Error threshold exceeded",
            priority=20,
        ))

        # Sort by priority
        self._stopping_conditions.sort(key=lambda c: c.priority, reverse=True)

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
        metrics = get_metrics()
        archetype = self._get_archetype_name()

        # Record agent spawn
        metrics.agent_spawned(archetype)

        self._context = AgentContext(
            agent_id=self.agent_id,
            session_id=self._session_id,
            max_turns=self.max_turns,
        )

        try:
            # Initialize
            self.state = AgentState.INITIALIZING
            logger.info(f"Agent '{self.name}' initializing...")

            # Audit: Log agent start
            await self._audit_agent_started()

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
            execution_time_seconds = execution_time_ms / 1000

            # Record agent completion metrics
            metrics.agent_completed(
                archetype=archetype,
                status="completed",
                duration_seconds=execution_time_seconds,
                turns=self._context.turn_number,
            )

            # Audit: Log agent completion
            await self._audit_agent_completed(result, execution_time_ms)

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
            execution_time_seconds = execution_time_ms / 1000

            # Record agent timeout metrics
            metrics.agent_completed(
                archetype=archetype,
                status="timeout",
                duration_seconds=execution_time_seconds,
                turns=self._context.turn_number if self._context else 0,
            )
            metrics.agent_error(archetype, "timeout")

            # Audit: Log agent failure
            await self._audit_agent_failed(
                f"Timeout after {self.timeout_ms}ms",
                execution_time_ms,
            )

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
            execution_time_seconds = execution_time_ms / 1000

            # Record agent failure metrics
            metrics.agent_completed(
                archetype=archetype,
                status="failed",
                duration_seconds=execution_time_seconds,
                turns=self._context.turn_number if self._context else 0,
            )
            metrics.agent_error(archetype, type(e).__name__)

            # Audit: Log agent failure
            await self._audit_agent_failed(str(e), execution_time_ms)

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

    def _get_archetype_name(self) -> str:
        """Get the archetype name for metrics.

        Returns the class name without 'Agent' suffix in lowercase,
        or falls back to the agent name.
        """
        class_name = self.__class__.__name__
        if class_name.endswith("Agent"):
            return class_name[:-5].lower()
        return self.name.lower()

    # =========================================================================
    # Stopping Conditions & Checkpoints
    # =========================================================================

    async def should_stop(self) -> tuple[bool, StoppingReason | None, str]:
        """
        Check if the agent should stop execution.

        Based on Anthropic's guidance: "Implement explicit stopping conditions."

        Returns:
            Tuple of (should_stop, reason, message)
        """
        for condition in self._stopping_conditions:
            if condition.evaluate(self):
                logger.info(
                    f"Agent '{self.name}' stopping: {condition.reason.value} - {condition.message}"
                )
                return True, condition.reason, condition.message

        return False, None, ""

    def add_stopping_condition(self, condition: StoppingCondition) -> None:
        """Add a custom stopping condition."""
        self._stopping_conditions.append(condition)
        self._stopping_conditions.sort(key=lambda c: c.priority, reverse=True)

    async def checkpoint(self, force: bool = False) -> AgentCheckpoint | None:
        """
        Create a checkpoint of current agent state.

        Checkpoints are created:
        - Automatically every checkpoint_interval turns
        - When force=True
        - Before critical operations

        Args:
            force: Force checkpoint creation regardless of interval

        Returns:
            AgentCheckpoint if created, None otherwise
        """
        if not self._context:
            return None

        # Check if checkpoint is due
        turns_since_last = self._context.turn_number - self._last_checkpoint_turn
        if not force and turns_since_last < self.checkpoint_interval:
            return None

        checkpoint = AgentCheckpoint(
            agent_id=self.agent_id,
            session_id=self._session_id or "unknown",
            turn_number=self._context.turn_number,
            state={
                "agent_state": self._state.value,
                "consecutive_errors": self._consecutive_errors,
            },
            decisions_made=[
                {
                    "choice": d.choice,
                    "confidence": d.confidence,
                    "category": d.category,
                }
                for d in self._tracked_decisions[-10:]  # Last 10 decisions
            ],
            memory_snapshot={
                "decisions_count": len(self._decisions_logged),
            },
        )

        self._checkpoints.append(checkpoint)
        self._last_checkpoint_turn = self._context.turn_number

        # Notify handlers
        for handler in self._on_checkpoint:
            try:
                handler(checkpoint)
            except Exception as e:
                logger.warning(f"Checkpoint handler error: {e}")

        # Store in Hive Mind if available
        if self._hive_mind:
            try:
                await self._hive_mind.set(
                    f"checkpoint:{self.agent_id}:{self._context.turn_number}",
                    checkpoint.to_dict(),
                    ttl=3600 * 24,  # 24 hours
                )
            except Exception as e:
                logger.warning(f"Failed to store checkpoint: {e}")

        logger.debug(f"Checkpoint created at turn {self._context.turn_number}")
        return checkpoint

    def get_latest_checkpoint(self) -> AgentCheckpoint | None:
        """Get the most recent checkpoint."""
        return self._checkpoints[-1] if self._checkpoints else None

    async def restore_from_checkpoint(
        self,
        checkpoint: AgentCheckpoint,
    ) -> bool:
        """
        Restore agent state from a checkpoint.

        Args:
            checkpoint: Checkpoint to restore from

        Returns:
            True if restoration successful
        """
        try:
            # Restore basic state
            if self._context:
                self._context.turn_number = checkpoint.turn_number

            state = checkpoint.state
            if "consecutive_errors" in state:
                self._consecutive_errors = state["consecutive_errors"]

            logger.info(
                f"Agent '{self.name}' restored from checkpoint "
                f"at turn {checkpoint.turn_number}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to restore from checkpoint: {e}")
            return False

    def on_checkpoint(
        self,
        handler: Callable[[AgentCheckpoint], None],
    ) -> None:
        """Register a checkpoint event handler."""
        self._on_checkpoint.append(handler)

    # =========================================================================
    # Decision Tracking
    # =========================================================================

    def track_decision(
        self,
        choice: str,
        confidence: float,
        alternatives: list[str] | None = None,
        rationale: str = "",
        category: str = "",
        uncertainty_bounds: tuple[float, float] | None = None,
    ) -> AgentDecision:
        """
        Track a decision with confidence and uncertainty.

        Based on Anthropic's guidance for uncertainty tracking in agents.

        Args:
            choice: The selected option
            confidence: Confidence score (0-1)
            alternatives: Other options considered
            rationale: Reasoning for the decision
            category: Decision category
            uncertainty_bounds: Low/high confidence bounds

        Returns:
            The tracked AgentDecision
        """
        # Calculate default uncertainty bounds from confidence
        if uncertainty_bounds is None:
            margin = (1 - confidence) / 2
            uncertainty_bounds = (
                max(0.0, confidence - margin),
                min(1.0, confidence + margin),
            )

        decision = AgentDecision(
            choice=choice,
            confidence=confidence,
            uncertainty_bounds=uncertainty_bounds,
            alternatives=alternatives or [],
            rationale=rationale,
            category=category,
        )

        self._tracked_decisions.append(decision)

        # Reset consecutive errors on successful decision
        if confidence > 0.5:
            self._consecutive_errors = 0

        logger.debug(
            f"Decision tracked: {choice} "
            f"(confidence={confidence:.2f}, category={category})"
        )

        return decision

    def get_decision_summary(self) -> dict[str, Any]:
        """Get summary of tracked decisions."""
        if not self._tracked_decisions:
            return {"count": 0}

        confidences = [d.confidence for d in self._tracked_decisions]
        categories = {}
        for d in self._tracked_decisions:
            categories[d.category] = categories.get(d.category, 0) + 1

        return {
            "count": len(self._tracked_decisions),
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "by_category": categories,
            "high_confidence_count": sum(1 for d in self._tracked_decisions if d.is_high_confidence),
        }

    def record_error(self, error: Exception | str) -> None:
        """Record an error occurrence for threshold tracking."""
        self._consecutive_errors += 1
        self._last_error = error if isinstance(error, Exception) else Exception(str(error))
        logger.warning(
            f"Agent '{self.name}' error #{self._consecutive_errors}: {error}"
        )

    # =========================================================================
    # Audit Logging
    # =========================================================================

    async def _audit_agent_started(self) -> None:
        """Log agent start to audit log."""
        if not self._audit_logger:
            return
        try:
            await self._audit_logger.log_agent_started(
                agent_id=self.agent_id,
                session_id=self._session_id or "unknown",
                agent_name=self.name,
                capabilities=self.capabilities,
                config={
                    "max_turns": self.max_turns,
                    "timeout_ms": self.timeout_ms,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to audit agent start: {e}")

    async def _audit_agent_completed(
        self,
        result: Any,
        execution_time_ms: int,
    ) -> None:
        """Log agent completion to audit log."""
        if not self._audit_logger:
            return
        try:
            await self._audit_logger.log_agent_completed(
                agent_id=self.agent_id,
                session_id=self._session_id or "unknown",
                result=result,
                turns_taken=self._context.turn_number if self._context else 0,
                execution_time_ms=execution_time_ms,
            )
        except Exception as e:
            logger.warning(f"Failed to audit agent completion: {e}")

    async def _audit_agent_failed(
        self,
        error: str,
        execution_time_ms: int,
    ) -> None:
        """Log agent failure to audit log."""
        if not self._audit_logger:
            return
        try:
            await self._audit_logger.log_agent_failed(
                agent_id=self.agent_id,
                session_id=self._session_id or "unknown",
                error=error,
                turns_taken=self._context.turn_number if self._context else 0,
                execution_time_ms=execution_time_ms,
            )
        except Exception as e:
            logger.warning(f"Failed to audit agent failure: {e}")

    async def audit_decision(
        self,
        decision_type: str,
        rationale: str,
        selected_option: str,
        confidence: float,
        alternatives: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Log a decision to the audit log.

        Args:
            decision_type: Type of decision (tool_selection, model_selection, etc.)
            rationale: Why this decision was made
            selected_option: The chosen option
            confidence: Confidence score (0-1)
            alternatives: Other options considered
        """
        if not self._audit_logger:
            return

        try:
            from titan.persistence.models import AuditEventType, DecisionType

            # Map string to DecisionType
            type_map = {
                "tool_selection": DecisionType.TOOL_SELECTION,
                "model_selection": DecisionType.MODEL_SELECTION,
                "topology_selection": DecisionType.TOPOLOGY_SELECTION,
                "task_delegation": DecisionType.TASK_DELEGATION,
                "error_recovery": DecisionType.ERROR_RECOVERY,
                "budget_allocation": DecisionType.BUDGET_ALLOCATION,
            }
            dt = type_map.get(decision_type, DecisionType.TOOL_SELECTION)

            # Create audit event for the decision
            event = await self._audit_logger.log_event(
                event_type=AuditEventType.AGENT_COMPLETED,
                action=f"Decision: {decision_type}",
                agent_id=self.agent_id,
                session_id=self._session_id,
                output_data={
                    "decision_type": decision_type,
                    "selected": selected_option,
                    "confidence": confidence,
                },
            )

            # Log the decision linked to the event
            await self._audit_logger.log_decision(
                audit_event_id=event.id,
                decision_type=dt,
                rationale=rationale,
                selected_option=selected_option,
                confidence=confidence,
                alternatives=alternatives,
            )
        except Exception as e:
            logger.warning(f"Failed to audit decision: {e}")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} '{self.name}' id={self.agent_id} state={self.state.value}>"

    def __enter__(self) -> BaseAgent:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        pass
