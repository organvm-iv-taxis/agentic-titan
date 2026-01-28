"""
Checkpoint Manager - Save and restore agent state.

Enables:
- Conversation checkpointing
- Agent state persistence
- Session recovery from saved state
- Progress tracking across restarts

Reference: vendor/cli/gemini-cli conversation checkpointing patterns
"""

from __future__ import annotations

import gzip
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("titan.persistence.checkpoint")


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class Checkpoint:
    """
    A checkpoint of agent/conversation state.

    Contains all information needed to resume from this point.
    """

    # Identification
    checkpoint_id: str
    session_id: str
    agent_id: str | None = None

    # Timestamp
    created_at: datetime = field(default_factory=datetime.now)
    name: str = ""
    description: str = ""

    # Conversation state
    messages: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""

    # Agent state
    agent_state: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    # Execution state
    turn_number: int = 0
    max_turns: int = 20
    decisions_logged: list[dict[str, Any]] = field(default_factory=list)

    # Tool state
    tool_results: list[dict[str, Any]] = field(default_factory=list)

    # Memory references
    memory_ids: list[str] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "created_at": self.created_at.isoformat(),
            "name": self.name,
            "description": self.description,
            "messages": self.messages,
            "system_prompt": self.system_prompt,
            "agent_state": self.agent_state,
            "context": self.context,
            "turn_number": self.turn_number,
            "max_turns": self.max_turns,
            "decisions_logged": self.decisions_logged,
            "tool_results": self.tool_results,
            "memory_ids": self.memory_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Create from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            session_id=data["session_id"],
            agent_id=data.get("agent_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            name=data.get("name", ""),
            description=data.get("description", ""),
            messages=data.get("messages", []),
            system_prompt=data.get("system_prompt", ""),
            agent_state=data.get("agent_state", {}),
            context=data.get("context", {}),
            turn_number=data.get("turn_number", 0),
            max_turns=data.get("max_turns", 20),
            decisions_logged=data.get("decisions_logged", []),
            tool_results=data.get("tool_results", []),
            memory_ids=data.get("memory_ids", []),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Checkpoint Manager
# ============================================================================


class CheckpointManager:
    """
    Manages checkpoint storage and retrieval.

    Supports:
    - File-based storage (JSON/pickle)
    - Automatic compression
    - Checkpoint rotation (keep last N)
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = ".titan/checkpoints",
        max_checkpoints: int = 10,
        compress: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.compress = compress

        # Ensure directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        checkpoint: Checkpoint,
        format: str = "json",
    ) -> Path:
        """
        Save a checkpoint to disk.

        Args:
            checkpoint: Checkpoint to save
            format: Storage format ("json" or "pickle")

        Returns:
            Path to saved checkpoint
        """
        timestamp = checkpoint.created_at.strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{checkpoint.session_id}_{timestamp}"

        if format == "json":
            filename += ".json"
            if self.compress:
                filename += ".gz"

            path = self.checkpoint_dir / filename
            data = json.dumps(checkpoint.to_dict(), indent=2, default=str)

            if self.compress:
                with gzip.open(path, "wt", encoding="utf-8") as f:
                    f.write(data)
            else:
                path.write_text(data)

        else:  # pickle
            filename += ".pkl"
            if self.compress:
                filename += ".gz"

            path = self.checkpoint_dir / filename

            if self.compress:
                with gzip.open(path, "wb") as f:
                    pickle.dump(checkpoint, f)
            else:
                with open(path, "wb") as f:
                    pickle.dump(checkpoint, f)

        logger.info(f"Saved checkpoint: {path}")

        # Rotate old checkpoints
        self._rotate_checkpoints(checkpoint.session_id)

        return path

    def load(self, path: str | Path) -> Checkpoint:
        """
        Load a checkpoint from disk.

        Args:
            path: Path to checkpoint file

        Returns:
            Loaded Checkpoint
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        is_compressed = path.suffix == ".gz"
        is_pickle = ".pkl" in path.name

        if is_pickle:
            if is_compressed:
                with gzip.open(path, "rb") as f:
                    checkpoint = pickle.load(f)
            else:
                with open(path, "rb") as f:
                    checkpoint = pickle.load(f)
        else:
            if is_compressed:
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = json.loads(path.read_text())
            checkpoint = Checkpoint.from_dict(data)

        logger.info(f"Loaded checkpoint: {path}")
        return checkpoint

    def list_checkpoints(
        self,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List available checkpoints.

        Args:
            session_id: Filter by session ID

        Returns:
            List of checkpoint metadata
        """
        checkpoints = []

        for path in self.checkpoint_dir.iterdir():
            if not path.name.startswith("checkpoint_"):
                continue

            # Parse filename
            parts = path.name.replace(".json", "").replace(".pkl", "").replace(".gz", "").split("_")
            if len(parts) >= 3:
                cp_session = parts[1]
                cp_timestamp = "_".join(parts[2:])

                if session_id and cp_session != session_id:
                    continue

                checkpoints.append({
                    "path": str(path),
                    "session_id": cp_session,
                    "timestamp": cp_timestamp,
                    "size_bytes": path.stat().st_size,
                    "format": "pickle" if ".pkl" in path.name else "json",
                    "compressed": path.suffix == ".gz",
                })

        # Sort by timestamp descending
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints

    def get_latest(self, session_id: str) -> Checkpoint | None:
        """
        Get the latest checkpoint for a session.

        Args:
            session_id: Session ID

        Returns:
            Latest Checkpoint or None
        """
        checkpoints = self.list_checkpoints(session_id)
        if checkpoints:
            return self.load(checkpoints[0]["path"])
        return None

    def delete(self, path: str | Path) -> bool:
        """
        Delete a checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            True if deleted
        """
        path = Path(path)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted checkpoint: {path}")
            return True
        return False

    def clear_session(self, session_id: str) -> int:
        """
        Clear all checkpoints for a session.

        Args:
            session_id: Session ID

        Returns:
            Number of checkpoints deleted
        """
        count = 0
        for cp in self.list_checkpoints(session_id):
            if self.delete(cp["path"]):
                count += 1
        return count

    def _rotate_checkpoints(self, session_id: str) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        checkpoints = self.list_checkpoints(session_id)

        if len(checkpoints) > self.max_checkpoints:
            for cp in checkpoints[self.max_checkpoints:]:
                self.delete(cp["path"])


# ============================================================================
# Convenience Functions
# ============================================================================

_default_manager: CheckpointManager | None = None


def get_manager(checkpoint_dir: str | Path | None = None) -> CheckpointManager:
    """Get or create the default checkpoint manager."""
    global _default_manager

    if _default_manager is None or checkpoint_dir:
        _default_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir or ".titan/checkpoints"
        )

    return _default_manager


def create_checkpoint(
    session_id: str,
    *,
    agent_id: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    system_prompt: str = "",
    agent_state: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    turn_number: int = 0,
    name: str = "",
    description: str = "",
    metadata: dict[str, Any] | None = None,
) -> Checkpoint:
    """
    Create a new checkpoint.

    Args:
        session_id: Session identifier
        agent_id: Optional agent identifier
        messages: Conversation messages
        system_prompt: System prompt
        agent_state: Agent state dict
        context: Context dict
        turn_number: Current turn number
        name: Checkpoint name
        description: Checkpoint description
        metadata: Additional metadata

    Returns:
        New Checkpoint
    """
    import uuid

    checkpoint = Checkpoint(
        checkpoint_id=f"cp-{uuid.uuid4().hex[:8]}",
        session_id=session_id,
        agent_id=agent_id,
        name=name,
        description=description,
        messages=messages or [],
        system_prompt=system_prompt,
        agent_state=agent_state or {},
        context=context or {},
        turn_number=turn_number,
        metadata=metadata or {},
    )

    return checkpoint


def save_checkpoint(checkpoint: Checkpoint) -> Path:
    """Save a checkpoint using the default manager."""
    manager = get_manager()
    return manager.save(checkpoint)


def restore_checkpoint(
    session_id: str,
    checkpoint_path: str | Path | None = None,
) -> Checkpoint | None:
    """
    Restore a checkpoint.

    Args:
        session_id: Session ID to restore
        checkpoint_path: Specific checkpoint path (optional)

    Returns:
        Restored Checkpoint or None if not found
    """
    manager = get_manager()

    if checkpoint_path:
        return manager.load(checkpoint_path)

    return manager.get_latest(session_id)
