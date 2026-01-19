"""
Agent State Serialization - Save and restore agent state.

Enables:
- Checkpointing agent progress
- Migration between runtimes
- Fault recovery
- State inspection
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("titan.migration.state")


@dataclass
class StateSnapshot:
    """
    Immutable snapshot of agent state at a point in time.

    Used for:
    - Checkpointing
    - Migration
    - Recovery
    - Auditing
    """

    id: str
    timestamp: datetime
    agent_id: str
    agent_type: str

    # Core state
    task: str
    context: dict[str, Any]
    memory: list[dict[str, Any]]

    # Progress
    turn_number: int
    tool_calls_made: int
    llm_calls_made: int

    # Status
    status: str  # running, paused, completed, failed
    last_action: str
    last_result: Any

    # Checksum for integrity
    checksum: str = ""

    def __post_init__(self) -> None:
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute checksum of state content."""
        content = json.dumps({
            "agent_id": self.agent_id,
            "task": self.task,
            "context": self.context,
            "memory": self.memory,
            "turn_number": self.turn_number,
            "status": self.status,
        }, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def verify(self) -> bool:
        """Verify state integrity."""
        return self.checksum == self._compute_checksum()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "task": self.task,
            "context": self.context,
            "memory": self.memory,
            "turn_number": self.turn_number,
            "tool_calls_made": self.tool_calls_made,
            "llm_calls_made": self.llm_calls_made,
            "status": self.status,
            "last_action": self.last_action,
            "last_result": str(self.last_result)[:1000] if self.last_result else None,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateSnapshot:
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            task=data["task"],
            context=data.get("context", {}),
            memory=data.get("memory", []),
            turn_number=data.get("turn_number", 0),
            tool_calls_made=data.get("tool_calls_made", 0),
            llm_calls_made=data.get("llm_calls_made", 0),
            status=data.get("status", "unknown"),
            last_action=data.get("last_action", ""),
            last_result=data.get("last_result"),
            checksum=data.get("checksum", ""),
        )


@dataclass
class AgentState:
    """
    Mutable agent state that can be serialized and restored.

    Manages:
    - Current state
    - Snapshot history
    - State transitions
    """

    agent_id: str
    agent_type: str
    task: str

    # Mutable state
    context: dict[str, Any] = field(default_factory=dict)
    memory: list[dict[str, Any]] = field(default_factory=list)

    # Progress
    turn_number: int = 0
    tool_calls_made: int = 0
    llm_calls_made: int = 0

    # Status
    status: str = "pending"
    last_action: str = ""
    last_result: Any = None

    # Snapshot history
    _snapshots: list[StateSnapshot] = field(default_factory=list)
    _snapshot_limit: int = 10

    def snapshot(self) -> StateSnapshot:
        """
        Create a snapshot of current state.

        Automatically manages snapshot history.
        """
        import uuid

        snap = StateSnapshot(
            id=f"snap_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            task=self.task,
            context=dict(self.context),
            memory=list(self.memory),
            turn_number=self.turn_number,
            tool_calls_made=self.tool_calls_made,
            llm_calls_made=self.llm_calls_made,
            status=self.status,
            last_action=self.last_action,
            last_result=self.last_result,
        )

        # Add to history
        self._snapshots.append(snap)

        # Trim history
        if len(self._snapshots) > self._snapshot_limit:
            self._snapshots = self._snapshots[-self._snapshot_limit:]

        logger.debug(f"Created snapshot {snap.id} for agent {self.agent_id}")
        return snap

    def restore(self, snapshot: StateSnapshot) -> bool:
        """
        Restore state from a snapshot.

        Returns True if successful.
        """
        # Verify integrity
        if not snapshot.verify():
            logger.error(f"Snapshot {snapshot.id} failed integrity check")
            return False

        # Restore state
        self.context = dict(snapshot.context)
        self.memory = list(snapshot.memory)
        self.turn_number = snapshot.turn_number
        self.tool_calls_made = snapshot.tool_calls_made
        self.llm_calls_made = snapshot.llm_calls_made
        self.status = snapshot.status
        self.last_action = snapshot.last_action
        self.last_result = snapshot.last_result

        logger.info(f"Restored agent {self.agent_id} from snapshot {snapshot.id}")
        return True

    def get_latest_snapshot(self) -> StateSnapshot | None:
        """Get the most recent snapshot."""
        return self._snapshots[-1] if self._snapshots else None

    def get_snapshots(self) -> list[StateSnapshot]:
        """Get all snapshots."""
        return list(self._snapshots)

    def update_progress(
        self,
        action: str,
        result: Any = None,
        increment_turn: bool = False,
    ) -> None:
        """Update progress tracking."""
        self.last_action = action
        self.last_result = result

        if increment_turn:
            self.turn_number += 1

    def record_tool_call(self) -> None:
        """Record a tool call."""
        self.tool_calls_made += 1

    def record_llm_call(self) -> None:
        """Record an LLM call."""
        self.llm_calls_made += 1

    def add_memory(self, entry: dict[str, Any]) -> None:
        """Add a memory entry."""
        self.memory.append({
            "timestamp": datetime.now().isoformat(),
            **entry,
        })

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "task": self.task,
            "context": self.context,
            "memory": self.memory,
            "turn_number": self.turn_number,
            "tool_calls_made": self.tool_calls_made,
            "llm_calls_made": self.llm_calls_made,
            "status": self.status,
            "last_action": self.last_action,
            "last_result": str(self.last_result)[:1000] if self.last_result else None,
            "snapshots": [s.to_dict() for s in self._snapshots],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        """Deserialize from dictionary."""
        state = cls(
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            task=data["task"],
            context=data.get("context", {}),
            memory=data.get("memory", []),
            turn_number=data.get("turn_number", 0),
            tool_calls_made=data.get("tool_calls_made", 0),
            llm_calls_made=data.get("llm_calls_made", 0),
            status=data.get("status", "pending"),
            last_action=data.get("last_action", ""),
            last_result=data.get("last_result"),
        )

        # Restore snapshots
        for snap_data in data.get("snapshots", []):
            state._snapshots.append(StateSnapshot.from_dict(snap_data))

        return state

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> AgentState:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
