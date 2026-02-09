"""
Titan Workflows - Temporal Inquiry Tracking

Enables re-inquiry over time with diff tracking to understand
how perspectives and insights evolve across repeated explorations
of the same topic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from titan.workflows.inquiry_engine import InquiryEngine, InquirySession, StageResult

logger = logging.getLogger("titan.workflows.temporal")


class DriftType(StrEnum):
    """Types of drift between inquiry versions."""

    EXPANSION = "expansion"  # New topics/concepts added
    CONTRACTION = "contraction"  # Topics removed or condensed
    REFINEMENT = "refinement"  # Similar content, more precise
    PIVOT = "pivot"  # Significant direction change
    STABLE = "stable"  # Minimal meaningful change


@dataclass
class StageDiff:
    """
    Diff between two versions of a stage result.

    Captures changes in content, key themes, and detected drift.
    """

    stage_name: str
    stage_index: int
    base_summary: str
    comparison_summary: str
    drift_type: DriftType
    drift_score: float  # 0-1, where 1 is maximum drift
    added_themes: list[str] = field(default_factory=list)
    removed_themes: list[str] = field(default_factory=list)
    consistent_themes: list[str] = field(default_factory=list)
    content_similarity: float = 0.0  # 0-1 similarity score
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage_name": self.stage_name,
            "stage_index": self.stage_index,
            "base_summary": self.base_summary,
            "comparison_summary": self.comparison_summary,
            "drift_type": self.drift_type.value,
            "drift_score": self.drift_score,
            "added_themes": self.added_themes,
            "removed_themes": self.removed_themes,
            "consistent_themes": self.consistent_themes,
            "content_similarity": self.content_similarity,
            "metadata": self.metadata,
        }


@dataclass
class InquiryDiff:
    """
    Complete diff between two inquiry sessions.

    Provides stage-by-stage comparison and overall drift analysis.
    """

    base_session_id: str
    comparison_session_id: str
    topic: str
    stage_diffs: list[StageDiff]
    overall_drift_score: float  # 0-1 average drift
    key_changes: list[str]  # Human-readable change descriptions
    drift_summary: DriftType
    base_timestamp: datetime
    comparison_timestamp: datetime
    time_elapsed: float  # Seconds between inquiries
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "base_session_id": self.base_session_id,
            "comparison_session_id": self.comparison_session_id,
            "topic": self.topic,
            "stage_diffs": [s.to_dict() for s in self.stage_diffs],
            "overall_drift_score": self.overall_drift_score,
            "key_changes": self.key_changes,
            "drift_summary": self.drift_summary.value,
            "base_timestamp": self.base_timestamp.isoformat(),
            "comparison_timestamp": self.comparison_timestamp.isoformat(),
            "time_elapsed": self.time_elapsed,
            "metadata": self.metadata,
        }


@dataclass
class TemporalChain:
    """
    A chain of related inquiry sessions over time.

    Tracks multiple re-inquiries on the same topic to understand
    how understanding evolves.
    """

    chain_id: str
    topic: str
    sessions: list[str]  # Chronological session IDs
    created_at: datetime = field(default_factory=datetime.now)
    last_inquiry_at: datetime | None = None
    total_inquiries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chain_id": self.chain_id,
            "topic": self.topic,
            "sessions": self.sessions,
            "created_at": self.created_at.isoformat(),
            "last_inquiry_at": self.last_inquiry_at.isoformat() if self.last_inquiry_at else None,
            "total_inquiries": self.total_inquiries,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalChain:
        """Create from dictionary."""
        created_at = (
            datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
        )
        last_inquiry_at = (
            datetime.fromisoformat(data["last_inquiry_at"]) if data.get("last_inquiry_at") else None
        )

        return cls(
            chain_id=data["chain_id"],
            topic=data["topic"],
            sessions=data["sessions"],
            created_at=created_at,
            last_inquiry_at=last_inquiry_at,
            total_inquiries=data.get("total_inquiries", len(data.get("sessions", []))),
            metadata=data.get("metadata", {}),
        )


class TemporalTracker:
    """
    Tracks inquiry sessions over time and computes diffs.

    Enables temporal analysis of how understanding evolves
    through repeated exploration of topics.
    """

    def __init__(self) -> None:
        """Initialize the temporal tracker."""
        self._chains: dict[str, TemporalChain] = {}
        self._topic_chains: dict[str, str] = {}  # topic -> chain_id mapping

    def create_chain(
        self,
        topic: str,
        session_id: str,
    ) -> TemporalChain:
        """
        Create a new temporal chain for a topic.

        Args:
            topic: The inquiry topic
            session_id: The first session in the chain

        Returns:
            New TemporalChain
        """
        chain_id = f"chain-{uuid4().hex[:12]}"

        chain = TemporalChain(
            chain_id=chain_id,
            topic=topic,
            sessions=[session_id],
            last_inquiry_at=datetime.now(),
            total_inquiries=1,
        )

        self._chains[chain_id] = chain
        self._topic_chains[self._normalize_topic(topic)] = chain_id

        logger.info(f"Created temporal chain {chain_id} for topic: {topic[:50]}")
        return chain

    def add_to_chain(
        self,
        chain_id: str,
        session_id: str,
    ) -> bool:
        """
        Add a session to an existing chain.

        Args:
            chain_id: Chain to add to
            session_id: Session to add

        Returns:
            True if added successfully
        """
        chain = self._chains.get(chain_id)
        if not chain:
            logger.warning(f"Chain not found: {chain_id}")
            return False

        if session_id not in chain.sessions:
            chain.sessions.append(session_id)
            chain.total_inquiries += 1
            chain.last_inquiry_at = datetime.now()
            logger.info(f"Added session {session_id} to chain {chain_id}")

        return True

    def get_chain(self, chain_id: str) -> TemporalChain | None:
        """Get a chain by ID."""
        return self._chains.get(chain_id)

    def get_chain_for_topic(self, topic: str) -> TemporalChain | None:
        """Get chain for a topic (if exists)."""
        normalized = self._normalize_topic(topic)
        chain_id = self._topic_chains.get(normalized)
        if chain_id:
            return self._chains.get(chain_id)
        return None

    def list_chains(self) -> list[TemporalChain]:
        """List all temporal chains."""
        return list(self._chains.values())

    async def compute_diff(
        self,
        base_session: InquirySession,
        comparison_session: InquirySession,
    ) -> InquiryDiff:
        """
        Compute diff between two inquiry sessions.

        Args:
            base_session: The earlier session (baseline)
            comparison_session: The later session to compare

        Returns:
            InquiryDiff with detailed comparison
        """
        logger.info(f"Computing diff between {base_session.id} and {comparison_session.id}")

        stage_diffs: list[StageDiff] = []
        key_changes: list[str] = []

        # Match stages by name
        base_stages = {r.stage_name: r for r in base_session.results}
        comp_stages = {r.stage_name: r for r in comparison_session.results}

        # Process matching stages
        all_stage_names = set(base_stages.keys()) | set(comp_stages.keys())

        for stage_name in sorted(all_stage_names):
            base_result = base_stages.get(stage_name)
            comp_result = comp_stages.get(stage_name)

            if base_result and comp_result:
                # Both have this stage - compute diff
                diff = self._compute_stage_diff(base_result, comp_result)
                stage_diffs.append(diff)

                if diff.drift_score > 0.3:
                    key_changes.append(
                        f"{stage_name}: {diff.drift_type.value} (drift: {diff.drift_score:.0%})"
                    )
            elif base_result and not comp_result:
                # Stage removed
                key_changes.append(f"{stage_name}: Stage removed in comparison")
            elif comp_result and not base_result:
                # Stage added
                key_changes.append(f"{stage_name}: New stage in comparison")

        # Compute overall drift
        if stage_diffs:
            overall_drift = sum(d.drift_score for d in stage_diffs) / len(stage_diffs)
        else:
            overall_drift = 0.0

        # Determine overall drift type
        drift_summary = self._classify_overall_drift(overall_drift, stage_diffs)

        # Calculate time elapsed
        time_elapsed = 0.0
        if base_session.created_at and comparison_session.created_at:
            time_elapsed = (comparison_session.created_at - base_session.created_at).total_seconds()

        return InquiryDiff(
            base_session_id=base_session.id,
            comparison_session_id=comparison_session.id,
            topic=base_session.topic,
            stage_diffs=stage_diffs,
            overall_drift_score=overall_drift,
            key_changes=key_changes,
            drift_summary=drift_summary,
            base_timestamp=base_session.created_at,
            comparison_timestamp=comparison_session.created_at,
            time_elapsed=time_elapsed,
        )

    def _compute_stage_diff(
        self,
        base: StageResult,
        comparison: StageResult,
    ) -> StageDiff:
        """Compute diff between two stage results."""
        # Extract key themes (simple word frequency approach)
        base_themes = self._extract_themes(base.content)
        comp_themes = self._extract_themes(comparison.content)

        added = [t for t in comp_themes if t not in base_themes]
        removed = [t for t in base_themes if t not in comp_themes]
        consistent = [t for t in base_themes if t in comp_themes]

        # Compute content similarity
        similarity = self._compute_similarity(base.content, comparison.content)

        # Determine drift type
        drift_type, drift_score = self._classify_stage_drift(
            similarity, len(added), len(removed), len(consistent)
        )

        return StageDiff(
            stage_name=base.stage_name,
            stage_index=base.stage_index,
            base_summary=base.content[:200],
            comparison_summary=comparison.content[:200],
            drift_type=drift_type,
            drift_score=drift_score,
            added_themes=added[:10],
            removed_themes=removed[:10],
            consistent_themes=consistent[:10],
            content_similarity=similarity,
        )

    def _extract_themes(self, content: str) -> set[str]:
        """Extract key themes from content (simple approach)."""
        # Basic keyword extraction - could be enhanced with NLP
        import re

        words = re.findall(r"\b[a-zA-Z]{4,}\b", content.lower())
        word_freq: dict[str, int] = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1

        # Filter stopwords and get top themes
        stopwords = {
            "this",
            "that",
            "with",
            "from",
            "have",
            "been",
            "were",
            "which",
            "their",
            "about",
            "would",
            "could",
            "there",
            "these",
            "through",
            "being",
            "also",
            "more",
            "other",
            "some",
            "what",
            "when",
            "into",
        }

        themes = sorted(
            [(w, f) for w, f in word_freq.items() if w not in stopwords],
            key=lambda x: x[1],
            reverse=True,
        )

        return {w for w, _ in themes[:20]}

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute basic text similarity (Jaccard on words)."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        return intersection / union if union > 0 else 0.0

    def _classify_stage_drift(
        self,
        similarity: float,
        added_count: int,
        removed_count: int,
        consistent_count: int,
    ) -> tuple[DriftType, float]:
        """Classify the type and magnitude of drift."""
        # High similarity = stable
        if similarity > 0.7:
            return DriftType.STABLE, 0.1

        # More removed than added = contraction
        if removed_count > added_count * 1.5:
            return DriftType.CONTRACTION, 0.4 + (1 - similarity) * 0.3

        # More added than removed = expansion
        if added_count > removed_count * 1.5:
            return DriftType.EXPANSION, 0.3 + (1 - similarity) * 0.3

        # Very low similarity with lots of changes = pivot
        if similarity < 0.2 and added_count > 3 and removed_count > 3:
            return DriftType.PIVOT, 0.8

        # Moderate changes with good consistency = refinement
        if consistent_count > (added_count + removed_count):
            return DriftType.REFINEMENT, 0.2 + (1 - similarity) * 0.2

        # Default
        return DriftType.REFINEMENT, 0.3

    def _classify_overall_drift(
        self,
        overall_score: float,
        stage_diffs: list[StageDiff],
    ) -> DriftType:
        """Classify overall drift type."""
        if not stage_diffs:
            return DriftType.STABLE

        # Count drift types
        type_counts: dict[DriftType, int] = {}
        for d in stage_diffs:
            type_counts[d.drift_type] = type_counts.get(d.drift_type, 0) + 1

        # Return most common type
        most_common = max(type_counts.items(), key=lambda x: x[1])
        return most_common[0]

    def _normalize_topic(self, topic: str) -> str:
        """Normalize topic for matching."""
        return topic.lower().strip()[:200]


# =============================================================================
# Singleton and Factory
# =============================================================================

_tracker: TemporalTracker | None = None


def get_temporal_tracker() -> TemporalTracker:
    """Get the temporal tracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = TemporalTracker()
    return _tracker


async def create_re_inquiry(
    original_session: InquirySession,
    engine: InquiryEngine,
) -> InquirySession:
    """
    Create a re-inquiry session linked to an original.

    Args:
        original_session: The session to re-inquire
        engine: The inquiry engine to use

    Returns:
        New InquirySession with temporal linking
    """
    tracker = get_temporal_tracker()

    # Get or create chain
    chain = tracker.get_chain_for_topic(original_session.topic)
    if not chain:
        chain = tracker.create_chain(original_session.topic, original_session.id)

    # Start new inquiry
    new_session = await engine.start_inquiry(
        original_session.topic,
        original_session.workflow,
    )

    # Set temporal fields
    new_session.parent_session_id = original_session.id
    new_session.chain_id = chain.chain_id
    new_session.version = original_session.version + 1

    # Add to chain
    tracker.add_to_chain(chain.chain_id, new_session.id)

    logger.info(
        f"Created re-inquiry {new_session.id} (v{new_session.version}) "
        f"linked to {original_session.id}"
    )

    return new_session
