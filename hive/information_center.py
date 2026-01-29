"""Information Centers for Pattern Aggregation and Broadcast.

Implements crow roost-inspired information centers where hub nodes
aggregate solutions and broadcast patterns to the swarm.

Key concepts:
- Information centers: Hub nodes that collect and distribute knowledge
- Learned patterns: Reusable solution patterns with confidence
- Multi-generational knowledge: Patterns that persist across sessions
- Pattern broadcast: Distributing successful patterns to subscribers

Based on research on:
- Crow roost dynamics and information sharing
- Knowledge aggregation in distributed systems
- Collective memory and multi-generational learning
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from titan.metrics import get_metrics

if TYPE_CHECKING:
    from hive.events import EventBus
    from hive.neighborhood import TopologicalNeighborhood

logger = logging.getLogger("titan.hive.information_center")


class InformationCenterRole(str, Enum):
    """Roles an information center can have."""

    AGGREGATOR = "aggregator"      # Collects solutions from agents
    BROADCASTER = "broadcaster"    # Distributes patterns to agents
    ARCHIVE = "archive"            # Stores multi-generational knowledge


@dataclass
class LearnedPattern:
    """A learned pattern that can be shared across agents.

    Patterns represent reusable solutions or approaches that have
    proven effective. They are aggregated at information centers
    and broadcast to subscribers.
    """

    pattern_id: str
    pattern_type: str             # Category of pattern
    content: dict[str, Any]       # The actual pattern data
    confidence: float = 0.5       # 0-1, how confident we are in this pattern
    generation: int = 0           # Which generation this pattern is from
    usage_count: int = 0          # How many times this pattern has been used
    success_count: int = 0        # How many times usage was successful
    contributor_ids: list[str] = field(default_factory=list)  # Agents who contributed
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate of this pattern."""
        if self.usage_count == 0:
            return 0.5  # No data, assume neutral
        return self.success_count / self.usage_count

    @property
    def effectiveness_score(self) -> float:
        """Calculate overall effectiveness score.

        Combines confidence, success rate, and usage frequency.
        """
        # Weight by usage (more usage = more reliable estimate)
        usage_weight = min(1.0, self.usage_count / 10.0)

        return (
            self.confidence * 0.3 +
            self.success_rate * usage_weight * 0.5 +
            (1.0 - usage_weight) * 0.5 * 0.2  # Prior for low-usage patterns
        )

    def record_usage(self, success: bool) -> None:
        """Record a usage of this pattern."""
        self.usage_count += 1
        if success:
            self.success_count += 1
        self.last_used = datetime.now(timezone.utc)

        # Update confidence based on outcome
        if success:
            self.confidence = min(1.0, self.confidence + 0.05)
        else:
            self.confidence = max(0.0, self.confidence - 0.1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "content": self.content,
            "confidence": self.confidence,
            "generation": self.generation,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "effectiveness_score": self.effectiveness_score,
            "contributor_ids": self.contributor_ids,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedPattern:
        """Create from dictionary."""
        return cls(
            pattern_id=data["pattern_id"],
            pattern_type=data["pattern_type"],
            content=data.get("content", {}),
            confidence=data.get("confidence", 0.5),
            generation=data.get("generation", 0),
            usage_count=data.get("usage_count", 0),
            success_count=data.get("success_count", 0),
            contributor_ids=data.get("contributor_ids", []),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class InformationCenter:
    """An information center that aggregates and broadcasts patterns.

    Based on crow roost research where certain locations become
    information exchange hubs. Agents visit these centers to share
    discoveries and learn from others.
    """

    center_id: str
    role: InformationCenterRole
    agent_ids: list[str] = field(default_factory=list)      # Member agents
    patterns: list[LearnedPattern] = field(default_factory=list)
    subscriber_ids: list[str] = field(default_factory=list)  # Agents subscribed to updates
    generation: int = 0                                       # Current generation
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_member(self, agent_id: str) -> None:
        """Add a member agent to the center."""
        if agent_id not in self.agent_ids:
            self.agent_ids.append(agent_id)

    def remove_member(self, agent_id: str) -> None:
        """Remove a member agent from the center."""
        if agent_id in self.agent_ids:
            self.agent_ids.remove(agent_id)

    def subscribe(self, agent_id: str) -> None:
        """Subscribe an agent to pattern updates."""
        if agent_id not in self.subscriber_ids:
            self.subscriber_ids.append(agent_id)

    def unsubscribe(self, agent_id: str) -> None:
        """Unsubscribe an agent from pattern updates."""
        if agent_id in self.subscriber_ids:
            self.subscriber_ids.remove(agent_id)

    def add_pattern(self, pattern: LearnedPattern) -> None:
        """Add a pattern to the center."""
        # Check for duplicate
        for existing in self.patterns:
            if existing.pattern_id == pattern.pattern_id:
                return
        self.patterns.append(pattern)

    def get_pattern(self, pattern_id: str) -> LearnedPattern | None:
        """Get a pattern by ID."""
        for pattern in self.patterns:
            if pattern.pattern_id == pattern_id:
                return pattern
        return None

    def get_patterns_by_type(self, pattern_type: str) -> list[LearnedPattern]:
        """Get all patterns of a specific type."""
        return [p for p in self.patterns if p.pattern_type == pattern_type]

    def get_best_patterns(self, limit: int = 10) -> list[LearnedPattern]:
        """Get the best patterns by effectiveness."""
        sorted_patterns = sorted(
            self.patterns,
            key=lambda p: p.effectiveness_score,
            reverse=True,
        )
        return sorted_patterns[:limit]

    @property
    def pattern_count(self) -> int:
        """Number of patterns stored."""
        return len(self.patterns)

    @property
    def member_count(self) -> int:
        """Number of member agents."""
        return len(self.agent_ids)

    @property
    def subscriber_count(self) -> int:
        """Number of subscribers."""
        return len(self.subscriber_ids)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "center_id": self.center_id,
            "role": self.role.value,
            "agent_ids": self.agent_ids,
            "pattern_count": self.pattern_count,
            "subscriber_ids": self.subscriber_ids,
            "generation": self.generation,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class InformationCenterManager:
    """Manages information centers for pattern aggregation and broadcast.

    Handles:
    - Creating and destroying information centers
    - Electing centers from candidates
    - Aggregating solutions into patterns
    - Broadcasting patterns to subscribers
    - Archiving generational knowledge
    """

    # Configuration
    MIN_CENTER_MEMBERS = 3
    MAX_PATTERNS_PER_CENTER = 100
    CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to broadcast

    def __init__(
        self,
        neighborhood: TopologicalNeighborhood | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize the information center manager.

        Args:
            neighborhood: Topological neighborhood for member selection.
            event_bus: Event bus for publishing events.
        """
        self._neighborhood = neighborhood
        self._event_bus = event_bus

        self._centers: dict[str, InformationCenter] = {}
        self._agent_center: dict[str, str] = {}  # agent_id -> center_id

        # Generation tracking
        self._current_generation = 0
        self._generation_archives: dict[int, list[LearnedPattern]] = {}

        # Pattern counter
        self._pattern_counter = 0

    @property
    def centers(self) -> list[InformationCenter]:
        """Get all information centers."""
        return list(self._centers.values())

    @property
    def current_generation(self) -> int:
        """Get current generation number."""
        return self._current_generation

    async def create_center(
        self,
        agent_ids: list[str],
        role: InformationCenterRole = InformationCenterRole.AGGREGATOR,
        center_id: str | None = None,
    ) -> InformationCenter:
        """Create a new information center.

        Args:
            agent_ids: Initial member agents.
            role: Role of the center.
            center_id: Optional specific ID.

        Returns:
            The created InformationCenter.
        """
        if center_id is None:
            center_id = f"info_center_{uuid.uuid4().hex[:8]}"

        center = InformationCenter(
            center_id=center_id,
            role=role,
            agent_ids=list(agent_ids),
            generation=self._current_generation,
        )

        self._centers[center_id] = center

        # Map agents to center
        for agent_id in agent_ids:
            self._agent_center[agent_id] = center_id

        # Emit event
        if self._event_bus:
            from hive.events import EventType
            await self._event_bus.emit(
                EventType.INFO_CENTER_CREATED,
                {
                    "center_id": center_id,
                    "role": role.value,
                    "member_count": len(agent_ids),
                },
                source_id="info_center_manager",
            )

        # Record metric
        get_metrics().set_info_centers_active(role.value, len([
            c for c in self._centers.values() if c.role == role
        ]))

        logger.info(f"Created information center: {center_id} ({role.value})")
        return center

    async def destroy_center(self, center_id: str) -> bool:
        """Destroy an information center.

        Args:
            center_id: ID of center to destroy.

        Returns:
            True if destroyed, False if not found.
        """
        center = self._centers.get(center_id)
        if not center:
            return False

        # Remove agent mappings
        for agent_id in center.agent_ids:
            if agent_id in self._agent_center:
                del self._agent_center[agent_id]

        # Archive patterns before destroying
        if center.patterns:
            self._generation_archives.setdefault(
                center.generation, []
            ).extend(center.patterns)

        del self._centers[center_id]

        # Update metrics
        get_metrics().set_info_centers_active(center.role.value, len([
            c for c in self._centers.values() if c.role == center.role
        ]))

        logger.info(f"Destroyed information center: {center_id}")
        return True

    async def elect_center(
        self,
        candidates: list[str],
        criteria: str = "connectivity",
    ) -> InformationCenter:
        """Elect an information center from candidates.

        Uses network topology to select the best hub agent(s).

        Args:
            candidates: Candidate agent IDs.
            criteria: Selection criteria (connectivity, performance, random).

        Returns:
            The elected InformationCenter.
        """
        if not candidates:
            raise ValueError("No candidates provided for election")

        # Score candidates
        scored_candidates: list[tuple[str, float]] = []

        for agent_id in candidates:
            score = await self._score_candidate(agent_id, criteria)
            scored_candidates.append((agent_id, score))

        # Sort by score (highest first)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Select top candidates as center members
        member_count = max(self.MIN_CENTER_MEMBERS, len(candidates) // 5)
        members = [c[0] for c in scored_candidates[:member_count]]

        # Create the center
        center = await self.create_center(
            agent_ids=members,
            role=InformationCenterRole.AGGREGATOR,
        )

        # Emit event
        if self._event_bus:
            from hive.events import EventType
            await self._event_bus.emit(
                EventType.INFO_CENTER_ELECTED,
                {
                    "center_id": center.center_id,
                    "criteria": criteria,
                    "member_ids": members,
                },
                source_id="info_center_manager",
            )

        logger.info(
            f"Elected information center: {center.center_id} "
            f"(criteria={criteria}, members={len(members)})"
        )
        return center

    async def _score_candidate(self, agent_id: str, criteria: str) -> float:
        """Score a candidate for election."""
        if not self._neighborhood:
            return 0.5

        profile = self._neighborhood._profiles.get(agent_id)
        if not profile:
            return 0.0

        if criteria == "connectivity":
            # Score by number of connections
            neighbors = self._neighborhood.get_neighbors(agent_id)
            return len(neighbors) / self._neighborhood.neighbor_count

        elif criteria == "performance":
            # Score by performance
            return profile.performance_score

        elif criteria == "random":
            # Random selection
            import random
            return random.random()

        else:
            # Default: capability count
            return len(profile.capabilities) / 10.0

    async def aggregate_solution(
        self,
        center_id: str,
        solution: dict[str, Any],
        contributor_id: str,
        pattern_type: str = "solution",
        confidence: float = 0.5,
    ) -> LearnedPattern:
        """Aggregate a solution into a learned pattern.

        Args:
            center_id: Information center to add pattern to.
            solution: The solution data to aggregate.
            contributor_id: Agent who contributed.
            pattern_type: Category of the pattern.
            confidence: Initial confidence level.

        Returns:
            The created LearnedPattern.
        """
        center = self._centers.get(center_id)
        if not center:
            raise ValueError(f"Information center not found: {center_id}")

        # Create pattern
        self._pattern_counter += 1
        pattern_id = f"pattern_{self._pattern_counter}"

        pattern = LearnedPattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            content=solution,
            confidence=confidence,
            generation=self._current_generation,
            contributor_ids=[contributor_id],
        )

        # Check for similar existing pattern
        similar = self._find_similar_pattern(center, solution, pattern_type)
        if similar:
            # Merge with existing pattern
            similar.confidence = min(1.0, similar.confidence + 0.1)
            if contributor_id not in similar.contributor_ids:
                similar.contributor_ids.append(contributor_id)
            pattern = similar
        else:
            # Add new pattern
            center.add_pattern(pattern)

            # Trim if too many patterns
            if len(center.patterns) > self.MAX_PATTERNS_PER_CENTER:
                self._trim_patterns(center)

        # Emit event
        if self._event_bus:
            from hive.events import EventType
            await self._event_bus.emit(
                EventType.PATTERN_AGGREGATED,
                {
                    "center_id": center_id,
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern_type,
                    "confidence": pattern.confidence,
                },
                source_id="info_center_manager",
            )

        # Record metric
        get_metrics().pattern_stored(center.role.value)

        logger.debug(
            f"Aggregated pattern {pattern.pattern_id} at {center_id} "
            f"(type={pattern_type}, confidence={pattern.confidence:.2f})"
        )
        return pattern

    def _find_similar_pattern(
        self,
        center: InformationCenter,
        solution: dict[str, Any],
        pattern_type: str,
    ) -> LearnedPattern | None:
        """Find a similar existing pattern."""
        for pattern in center.patterns:
            if pattern.pattern_type != pattern_type:
                continue

            # Simple similarity check - could be more sophisticated
            if pattern.content == solution:
                return pattern

            # Check key overlap
            if set(pattern.content.keys()) == set(solution.keys()):
                # Same structure, might be similar
                matching_values = sum(
                    1 for k, v in solution.items()
                    if pattern.content.get(k) == v
                )
                if matching_values > len(solution) * 0.7:
                    return pattern

        return None

    def _trim_patterns(self, center: InformationCenter) -> int:
        """Trim patterns to stay under limit.

        Removes lowest effectiveness patterns.
        """
        if len(center.patterns) <= self.MAX_PATTERNS_PER_CENTER:
            return 0

        # Sort by effectiveness (lowest first)
        center.patterns.sort(key=lambda p: p.effectiveness_score)

        # Remove lowest performers
        to_remove = len(center.patterns) - self.MAX_PATTERNS_PER_CENTER
        center.patterns = center.patterns[to_remove:]

        return to_remove

    async def broadcast_pattern(
        self,
        center_id: str,
        pattern_id: str,
        target_agents: list[str] | None = None,
    ) -> int:
        """Broadcast a pattern to subscribers.

        Args:
            center_id: Information center holding the pattern.
            pattern_id: Pattern to broadcast.
            target_agents: Specific agents to target (or all subscribers).

        Returns:
            Number of agents pattern was broadcast to.
        """
        center = self._centers.get(center_id)
        if not center:
            raise ValueError(f"Information center not found: {center_id}")

        pattern = center.get_pattern(pattern_id)
        if not pattern:
            raise ValueError(f"Pattern not found: {pattern_id}")

        # Check confidence threshold
        if pattern.confidence < self.CONFIDENCE_THRESHOLD:
            logger.debug(
                f"Pattern {pattern_id} below confidence threshold "
                f"({pattern.confidence:.2f} < {self.CONFIDENCE_THRESHOLD})"
            )
            return 0

        # Determine recipients
        recipients = target_agents or center.subscriber_ids
        if not recipients:
            return 0

        # Emit event
        if self._event_bus:
            from hive.events import EventType
            await self._event_bus.emit(
                EventType.PATTERN_BROADCAST,
                {
                    "center_id": center_id,
                    "pattern_id": pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "recipient_count": len(recipients),
                },
                source_id="info_center_manager",
            )

        # Record metric
        get_metrics().pattern_broadcast()

        logger.info(
            f"Broadcast pattern {pattern_id} to {len(recipients)} agents "
            f"(type={pattern.pattern_type}, confidence={pattern.confidence:.2f})"
        )
        return len(recipients)

    async def archive_generation(self, center_id: str) -> int:
        """Archive current generation patterns and increment generation.

        Args:
            center_id: Information center to archive.

        Returns:
            Number of patterns archived.
        """
        center = self._centers.get(center_id)
        if not center:
            raise ValueError(f"Information center not found: {center_id}")

        # Archive patterns
        archived_count = len(center.patterns)
        if center.patterns:
            self._generation_archives.setdefault(
                self._current_generation, []
            ).extend(center.patterns)

        # Increment generation
        self._current_generation += 1
        center.generation = self._current_generation

        # Keep only high-performing patterns for next generation
        best_patterns = center.get_best_patterns(limit=10)
        center.patterns = []
        for pattern in best_patterns:
            pattern.generation = self._current_generation
            center.patterns.append(pattern)

        # Emit event
        if self._event_bus:
            from hive.events import EventType
            await self._event_bus.emit(
                EventType.GENERATION_ARCHIVED,
                {
                    "center_id": center_id,
                    "archived_count": archived_count,
                    "new_generation": self._current_generation,
                    "retained_count": len(center.patterns),
                },
                source_id="info_center_manager",
            )

        # Record metric
        get_metrics().generation_archived()

        logger.info(
            f"Archived generation {self._current_generation - 1}: "
            f"{archived_count} patterns archived, "
            f"{len(center.patterns)} retained for generation {self._current_generation}"
        )
        return archived_count

    def get_best_pattern(
        self,
        center_id: str,
        pattern_type: str,
    ) -> LearnedPattern | None:
        """Get the best pattern of a type from a center.

        Args:
            center_id: Information center to query.
            pattern_type: Type of pattern to find.

        Returns:
            Best pattern, or None if none found.
        """
        center = self._centers.get(center_id)
        if not center:
            return None

        patterns = center.get_patterns_by_type(pattern_type)
        if not patterns:
            return None

        return max(patterns, key=lambda p: p.effectiveness_score)

    def get_archived_patterns(
        self,
        generation: int | None = None,
        pattern_type: str | None = None,
    ) -> list[LearnedPattern]:
        """Get archived patterns from previous generations.

        Args:
            generation: Specific generation (or all if None).
            pattern_type: Filter by type (or all if None).

        Returns:
            List of archived patterns.
        """
        if generation is not None:
            patterns = self._generation_archives.get(generation, [])
        else:
            patterns = []
            for gen_patterns in self._generation_archives.values():
                patterns.extend(gen_patterns)

        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]

        return patterns

    def get_agent_center(self, agent_id: str) -> InformationCenter | None:
        """Get the information center an agent belongs to."""
        center_id = self._agent_center.get(agent_id)
        if center_id:
            return self._centers.get(center_id)
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "centers": [c.to_dict() for c in self._centers.values()],
            "current_generation": self._current_generation,
            "total_patterns": sum(c.pattern_count for c in self._centers.values()),
            "archived_generations": list(self._generation_archives.keys()),
            "total_archived": sum(
                len(patterns) for patterns in self._generation_archives.values()
            ),
        }
