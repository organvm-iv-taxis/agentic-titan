"""
Episodic Learning - Learn from past agent interactions.

Based on iGOR pattern: Every interaction is a learning signal.
No explicit feedback required - outcomes inform future decisions.

Components:
- Episode: A single agent interaction with context, action, result
- EpisodeOutcome: Success/failure indicators with confidence
- EpisodicMemory: Vector-backed storage for semantic retrieval
- LearningSignal: Implicit feedback extracted from outcomes
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger("titan.learning.episodic")


# ============================================================================
# Episode Types
# ============================================================================


class EpisodeOutcome(str, Enum):
    """Outcome classification for episodes."""

    SUCCESS = "success"  # Task completed successfully
    PARTIAL = "partial"  # Partially successful
    FAILURE = "failure"  # Task failed
    TIMEOUT = "timeout"  # Task timed out
    CANCELLED = "cancelled"  # Task was cancelled
    UNKNOWN = "unknown"  # Outcome not determined


@dataclass
class LearningSignal:
    """
    Implicit learning signal extracted from an episode.

    No explicit feedback required - signals inferred from:
    - Outcome success/failure
    - Time taken vs expected
    - Tool usage patterns
    - User corrections (if any)
    """

    # Confidence that this approach should be repeated (0-1)
    repeat_confidence: float = 0.5

    # Factors that contributed to outcome
    positive_factors: list[str] = field(default_factory=list)
    negative_factors: list[str] = field(default_factory=list)

    # Specific lessons
    lessons: list[str] = field(default_factory=list)

    # Suggested adjustments for similar tasks
    adjustments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repeat_confidence": self.repeat_confidence,
            "positive_factors": self.positive_factors,
            "negative_factors": self.negative_factors,
            "lessons": self.lessons,
            "adjustments": self.adjustments,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearningSignal:
        return cls(
            repeat_confidence=data.get("repeat_confidence", 0.5),
            positive_factors=data.get("positive_factors", []),
            negative_factors=data.get("negative_factors", []),
            lessons=data.get("lessons", []),
            adjustments=data.get("adjustments", {}),
        )


@dataclass
class Episode:
    """
    A recorded agent episode (interaction + outcome).

    Captures everything needed to learn from past experience:
    - What was the task?
    - What context was available?
    - What actions were taken?
    - What was the result?
    - What can we learn?
    """

    id: str = field(default_factory=lambda: f"ep_{uuid.uuid4().hex[:12]}")

    # Task context
    task: str = ""
    task_type: str = ""  # research, code, review, etc.
    agent_type: str = ""

    # Environment context
    context: dict[str, Any] = field(default_factory=dict)

    # Actions taken
    actions: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    llm_calls: int = 0

    # Results
    outcome: EpisodeOutcome = EpisodeOutcome.UNKNOWN
    result: Any = None
    error: str | None = None

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: datetime | None = None
    duration_seconds: float = 0.0

    # Learning
    signal: LearningSignal | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def complete(
        self,
        outcome: EpisodeOutcome,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Mark episode as complete."""
        self.ended_at = datetime.now()
        self.duration_seconds = (self.ended_at - self.started_at).total_seconds()
        self.outcome = outcome
        self.result = result
        self.error = error

        # Generate learning signal
        self.signal = self._extract_signal()

    def add_action(self, action_type: str, details: dict[str, Any]) -> None:
        """Record an action taken."""
        self.actions.append({
            "type": action_type,
            "timestamp": datetime.now().isoformat(),
            **details,
        })

    def add_tool_call(self, tool_name: str) -> None:
        """Record a tool call."""
        self.tool_calls.append(tool_name)

    def _extract_signal(self) -> LearningSignal:
        """Extract learning signal from episode outcome."""
        signal = LearningSignal()

        # Base confidence on outcome
        confidence_map = {
            EpisodeOutcome.SUCCESS: 0.9,
            EpisodeOutcome.PARTIAL: 0.6,
            EpisodeOutcome.FAILURE: 0.2,
            EpisodeOutcome.TIMEOUT: 0.3,
            EpisodeOutcome.CANCELLED: 0.4,
            EpisodeOutcome.UNKNOWN: 0.5,
        }
        signal.repeat_confidence = confidence_map.get(self.outcome, 0.5)

        # Analyze positive factors
        if self.outcome == EpisodeOutcome.SUCCESS:
            if len(self.tool_calls) <= 3:
                signal.positive_factors.append("efficient_tool_usage")
            if self.duration_seconds < 60:
                signal.positive_factors.append("fast_completion")
            if self.llm_calls <= 2:
                signal.positive_factors.append("minimal_llm_calls")

        # Analyze negative factors
        if self.outcome in [EpisodeOutcome.FAILURE, EpisodeOutcome.TIMEOUT]:
            if len(self.tool_calls) > 10:
                signal.negative_factors.append("excessive_tool_usage")
            if self.duration_seconds > 300:
                signal.negative_factors.append("slow_execution")
            if self.error:
                signal.negative_factors.append(f"error:{self.error[:50]}")

        # Extract lessons
        if self.outcome == EpisodeOutcome.SUCCESS:
            unique_tools = set(self.tool_calls)
            if unique_tools:
                signal.lessons.append(f"effective_tools:{','.join(unique_tools)}")

        if self.outcome == EpisodeOutcome.FAILURE and self.error:
            signal.lessons.append(f"avoid:{self.error[:100]}")

        # Suggest adjustments
        if self.outcome == EpisodeOutcome.TIMEOUT:
            signal.adjustments["increase_timeout"] = True
        if self.outcome == EpisodeOutcome.FAILURE and self.llm_calls > 5:
            signal.adjustments["simplify_approach"] = True

        return signal

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "task": self.task,
            "task_type": self.task_type,
            "agent_type": self.agent_type,
            "context": self.context,
            "actions": self.actions,
            "tool_calls": self.tool_calls,
            "llm_calls": self.llm_calls,
            "outcome": self.outcome.value,
            "result": str(self.result)[:500] if self.result else None,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
            "signal": self.signal.to_dict() if self.signal else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        """Create from dictionary."""
        episode = cls(
            id=data["id"],
            task=data["task"],
            task_type=data.get("task_type", ""),
            agent_type=data.get("agent_type", ""),
            context=data.get("context", {}),
            actions=data.get("actions", []),
            tool_calls=data.get("tool_calls", []),
            llm_calls=data.get("llm_calls", 0),
            outcome=EpisodeOutcome(data.get("outcome", "unknown")),
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )

        if data.get("started_at"):
            episode.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("ended_at"):
            episode.ended_at = datetime.fromisoformat(data["ended_at"])
        if data.get("duration_seconds"):
            episode.duration_seconds = data["duration_seconds"]
        if data.get("signal"):
            episode.signal = LearningSignal.from_dict(data["signal"])

        return episode

    def content_hash(self) -> str:
        """Generate hash for deduplication."""
        content = f"{self.task}:{self.task_type}:{','.join(self.tool_calls)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# ============================================================================
# Episodic Memory
# ============================================================================


class EpisodicMemory:
    """
    Vector-backed episodic memory for semantic retrieval.

    Stores episodes in ChromaDB for:
    - Semantic similarity search
    - Task-type filtering
    - Outcome-based retrieval
    """

    def __init__(
        self,
        collection_name: str = "titan_episodes",
        use_hive_mind: bool = True,
    ) -> None:
        self._collection_name = collection_name
        self._use_hive_mind = use_hive_mind
        self._collection: Any = None
        self._local_cache: dict[str, Episode] = {}

    async def initialize(self) -> None:
        """Initialize the memory store."""
        if self._use_hive_mind:
            try:
                from hive.memory import HiveMind, get_hive_mind

                hive = get_hive_mind()
                await hive.connect()

                # Use hive mind's ChromaDB collection
                # Episodes stored as memories with special type
                logger.info("Episodic memory connected to Hive Mind")

            except Exception as e:
                logger.warning(f"Could not connect to Hive Mind: {e}")
                logger.info("Using local cache for episodic memory")
        else:
            logger.info("Using local cache for episodic memory")

    async def store(self, episode: Episode) -> str:
        """
        Store an episode.

        Returns the episode ID.
        """
        # Always store in local cache
        self._local_cache[episode.id] = episode

        # Try to store in Hive Mind
        if self._use_hive_mind:
            try:
                from hive.memory import get_hive_mind, Memory

                hive = get_hive_mind()

                # Create memory from episode
                memory = Memory(
                    content=json.dumps(episode.to_dict()),
                    importance=self._calculate_importance(episode),
                    metadata={
                        "type": "episode",
                        "task_type": episode.task_type,
                        "agent_type": episode.agent_type,
                        "outcome": episode.outcome.value,
                        "episode_id": episode.id,
                    },
                )

                # Use task as embedding text for semantic search
                memory_id = await hive.remember(
                    f"{episode.task} {episode.task_type}",
                    importance=memory.importance,
                )

                logger.debug(f"Stored episode {episode.id} in Hive Mind: {memory_id}")

            except Exception as e:
                logger.warning(f"Could not store in Hive Mind: {e}")

        logger.info(f"Stored episode {episode.id} ({episode.outcome.value})")
        return episode.id

    async def recall(
        self,
        query: str,
        task_type: str | None = None,
        limit: int = 5,
        min_confidence: float = 0.5,
    ) -> list[Episode]:
        """
        Recall relevant episodes.

        Args:
            query: Semantic search query
            task_type: Filter by task type
            limit: Max episodes to return
            min_confidence: Minimum repeat_confidence threshold

        Returns:
            List of relevant episodes
        """
        episodes: list[Episode] = []

        # Try Hive Mind first
        if self._use_hive_mind:
            try:
                from hive.memory import get_hive_mind

                hive = get_hive_mind()
                memories = await hive.recall(query, k=limit * 2)  # Get extra for filtering

                for mem in memories:
                    if mem.metadata.get("type") == "episode":
                        episode_data = json.loads(mem.content)
                        episode = Episode.from_dict(episode_data)

                        # Apply filters
                        if task_type and episode.task_type != task_type:
                            continue
                        if episode.signal and episode.signal.repeat_confidence < min_confidence:
                            continue

                        episodes.append(episode)

                        if len(episodes) >= limit:
                            break

            except Exception as e:
                logger.warning(f"Hive Mind recall failed: {e}")

        # Fallback to local cache
        if not episodes:
            for episode in self._local_cache.values():
                if task_type and episode.task_type != task_type:
                    continue
                if episode.signal and episode.signal.repeat_confidence < min_confidence:
                    continue

                # Simple keyword matching for local cache
                if any(word.lower() in query.lower() for word in episode.task.split()):
                    episodes.append(episode)

                if len(episodes) >= limit:
                    break

        logger.debug(f"Recalled {len(episodes)} episodes for '{query[:50]}...'")
        return episodes

    async def get_successful_patterns(
        self,
        task_type: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get patterns from successful episodes.

        Returns tool usage patterns and approaches that worked.
        """
        patterns: list[dict[str, Any]] = []

        # Search for successful episodes of this type
        episodes = await self.recall(
            query=task_type,
            task_type=task_type,
            limit=limit,
            min_confidence=0.7,  # Only high-confidence successes
        )

        for episode in episodes:
            if episode.outcome == EpisodeOutcome.SUCCESS and episode.signal:
                patterns.append({
                    "tools_used": list(set(episode.tool_calls)),
                    "duration": episode.duration_seconds,
                    "llm_calls": episode.llm_calls,
                    "positive_factors": episode.signal.positive_factors,
                    "lessons": episode.signal.lessons,
                })

        return patterns

    async def get_failure_lessons(
        self,
        task_type: str,
        limit: int = 5,
    ) -> list[str]:
        """
        Get lessons from failed episodes.

        Returns things to avoid for this task type.
        """
        lessons: list[str] = []

        # Get all episodes from local cache for task type
        for episode in self._local_cache.values():
            if episode.task_type != task_type:
                continue
            if episode.outcome not in [EpisodeOutcome.FAILURE, EpisodeOutcome.TIMEOUT]:
                continue
            if episode.signal:
                lessons.extend(episode.signal.lessons)
                lessons.extend([f"avoid:{f}" for f in episode.signal.negative_factors])

        return list(set(lessons))[:limit]

    def _calculate_importance(self, episode: Episode) -> float:
        """Calculate importance score for episode."""
        # Base importance on outcome
        importance = 0.5

        if episode.outcome == EpisodeOutcome.SUCCESS:
            importance = 0.8
        elif episode.outcome == EpisodeOutcome.FAILURE:
            importance = 0.7  # Failures are also valuable to learn from
        elif episode.outcome == EpisodeOutcome.PARTIAL:
            importance = 0.6

        # Boost for significant lessons
        if episode.signal and len(episode.signal.lessons) > 2:
            importance += 0.1

        return min(importance, 1.0)

    async def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        total = len(self._local_cache)

        outcomes = {}
        task_types = {}

        for episode in self._local_cache.values():
            # Count outcomes
            outcome = episode.outcome.value
            outcomes[outcome] = outcomes.get(outcome, 0) + 1

            # Count task types
            if episode.task_type:
                task_types[episode.task_type] = task_types.get(episode.task_type, 0) + 1

        return {
            "total_episodes": total,
            "outcomes": outcomes,
            "task_types": task_types,
            "using_hive_mind": self._use_hive_mind,
        }
