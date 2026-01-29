"""Tests for information centers (Phase 16C)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hive.information_center import (
    InformationCenter,
    InformationCenterManager,
    InformationCenterRole,
    LearnedPattern,
)


class TestInformationCenterRole:
    """Tests for InformationCenterRole enum."""

    def test_roles_exist(self):
        """Test all roles are defined."""
        assert InformationCenterRole.AGGREGATOR == "aggregator"
        assert InformationCenterRole.BROADCASTER == "broadcaster"
        assert InformationCenterRole.ARCHIVE == "archive"


class TestLearnedPattern:
    """Tests for LearnedPattern dataclass."""

    def test_pattern_creation(self):
        """Test creating a pattern."""
        pattern = LearnedPattern(
            pattern_id="pattern_1",
            pattern_type="solution",
            content={"approach": "divide_and_conquer"},
            confidence=0.7,
        )

        assert pattern.pattern_id == "pattern_1"
        assert pattern.confidence == 0.7
        assert pattern.usage_count == 0

    def test_success_rate_no_usage(self):
        """Test success rate with no usage."""
        pattern = LearnedPattern(
            pattern_id="test",
            pattern_type="test",
            content={},
        )
        assert pattern.success_rate == 0.5  # Neutral

    def test_success_rate_with_usage(self):
        """Test success rate calculation."""
        pattern = LearnedPattern(
            pattern_id="test",
            pattern_type="test",
            content={},
            usage_count=10,
            success_count=8,
        )
        assert pattern.success_rate == 0.8

    def test_effectiveness_score(self):
        """Test effectiveness score calculation."""
        pattern = LearnedPattern(
            pattern_id="test",
            pattern_type="test",
            content={},
            confidence=0.8,
            usage_count=10,
            success_count=9,
        )

        score = pattern.effectiveness_score
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # High confidence + success

    def test_record_usage_success(self):
        """Test recording successful usage."""
        pattern = LearnedPattern(
            pattern_id="test",
            pattern_type="test",
            content={},
            confidence=0.5,
        )

        pattern.record_usage(success=True)

        assert pattern.usage_count == 1
        assert pattern.success_count == 1
        assert pattern.confidence > 0.5
        assert pattern.last_used is not None

    def test_record_usage_failure(self):
        """Test recording failed usage."""
        pattern = LearnedPattern(
            pattern_id="test",
            pattern_type="test",
            content={},
            confidence=0.5,
        )

        pattern.record_usage(success=False)

        assert pattern.usage_count == 1
        assert pattern.success_count == 0
        assert pattern.confidence < 0.5

    def test_to_dict(self):
        """Test pattern serialization."""
        pattern = LearnedPattern(
            pattern_id="test",
            pattern_type="solution",
            content={"key": "value"},
            confidence=0.6,
            generation=2,
        )

        data = pattern.to_dict()
        assert data["pattern_id"] == "test"
        assert data["pattern_type"] == "solution"
        assert data["confidence"] == 0.6
        assert data["generation"] == 2

    def test_from_dict(self):
        """Test pattern deserialization."""
        data = {
            "pattern_id": "test_2",
            "pattern_type": "approach",
            "content": {"method": "iterative"},
            "confidence": 0.75,
            "generation": 3,
            "usage_count": 5,
            "success_count": 4,
            "contributor_ids": ["agent_1"],
            "created_at": "2024-01-01T00:00:00",
        }

        pattern = LearnedPattern.from_dict(data)
        assert pattern.pattern_id == "test_2"
        assert pattern.confidence == 0.75
        assert pattern.usage_count == 5


class TestInformationCenter:
    """Tests for InformationCenter dataclass."""

    def test_center_creation(self):
        """Test creating an information center."""
        center = InformationCenter(
            center_id="center_1",
            role=InformationCenterRole.AGGREGATOR,
            agent_ids=["agent_1", "agent_2"],
        )

        assert center.center_id == "center_1"
        assert center.role == InformationCenterRole.AGGREGATOR
        assert center.member_count == 2

    def test_add_member(self):
        """Test adding a member."""
        center = InformationCenter(
            center_id="test",
            role=InformationCenterRole.AGGREGATOR,
        )

        center.add_member("agent_1")
        assert "agent_1" in center.agent_ids

        # No duplicates
        center.add_member("agent_1")
        assert center.member_count == 1

    def test_remove_member(self):
        """Test removing a member."""
        center = InformationCenter(
            center_id="test",
            role=InformationCenterRole.AGGREGATOR,
            agent_ids=["agent_1", "agent_2"],
        )

        center.remove_member("agent_1")
        assert "agent_1" not in center.agent_ids
        assert center.member_count == 1

    def test_subscribe(self):
        """Test subscribing to updates."""
        center = InformationCenter(
            center_id="test",
            role=InformationCenterRole.BROADCASTER,
        )

        center.subscribe("subscriber_1")
        assert "subscriber_1" in center.subscriber_ids

    def test_add_pattern(self):
        """Test adding a pattern."""
        center = InformationCenter(
            center_id="test",
            role=InformationCenterRole.AGGREGATOR,
        )

        pattern = LearnedPattern(
            pattern_id="p1",
            pattern_type="solution",
            content={},
        )
        center.add_pattern(pattern)

        assert center.pattern_count == 1

    def test_get_pattern(self):
        """Test getting a pattern by ID."""
        center = InformationCenter(
            center_id="test",
            role=InformationCenterRole.AGGREGATOR,
        )

        pattern = LearnedPattern(
            pattern_id="p1",
            pattern_type="solution",
            content={"test": True},
        )
        center.add_pattern(pattern)

        result = center.get_pattern("p1")
        assert result is not None
        assert result.content["test"] is True

        # Non-existent
        assert center.get_pattern("non_existent") is None

    def test_get_best_patterns(self):
        """Test getting best patterns."""
        center = InformationCenter(
            center_id="test",
            role=InformationCenterRole.AGGREGATOR,
        )

        # Add patterns with different effectiveness
        for i in range(5):
            pattern = LearnedPattern(
                pattern_id=f"p{i}",
                pattern_type="solution",
                content={},
                confidence=0.1 * (i + 1),
                usage_count=10,
                success_count=i + 1,
            )
            center.add_pattern(pattern)

        best = center.get_best_patterns(limit=3)
        assert len(best) == 3
        # Should be sorted by effectiveness
        assert best[0].effectiveness_score >= best[1].effectiveness_score


class TestInformationCenterManager:
    """Tests for InformationCenterManager class."""

    @pytest.fixture
    def mock_neighborhood(self):
        """Create mock neighborhood."""
        neighborhood = MagicMock()
        neighborhood.neighbor_count = 7

        profiles = {}
        for i in range(10):
            profile = MagicMock()
            profile.performance_score = 0.5 + (i * 0.05)
            profile.capabilities = ["code", "research"]
            profiles[f"agent_{i}"] = profile

        neighborhood._profiles = profiles
        return neighborhood

    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        bus = MagicMock()
        bus.emit = AsyncMock()
        return bus

    @pytest.fixture
    def manager(self, mock_neighborhood, mock_event_bus):
        """Create an InformationCenterManager for testing."""
        return InformationCenterManager(
            neighborhood=mock_neighborhood,
            event_bus=mock_event_bus,
        )

    @pytest.mark.asyncio
    async def test_create_center(self, manager):
        """Test creating an information center."""
        center = await manager.create_center(
            agent_ids=["agent_1", "agent_2"],
            role=InformationCenterRole.AGGREGATOR,
        )

        assert center is not None
        assert center.role == InformationCenterRole.AGGREGATOR
        assert center.member_count == 2
        assert center.center_id in [c.center_id for c in manager.centers]

    @pytest.mark.asyncio
    async def test_destroy_center(self, manager):
        """Test destroying an information center."""
        center = await manager.create_center(
            agent_ids=["agent_1"],
            role=InformationCenterRole.AGGREGATOR,
        )

        result = await manager.destroy_center(center.center_id)
        assert result is True
        assert center.center_id not in [c.center_id for c in manager.centers]

    @pytest.mark.asyncio
    async def test_destroy_nonexistent_center(self, manager):
        """Test destroying non-existent center."""
        result = await manager.destroy_center("fake_id")
        assert result is False

    @pytest.mark.asyncio
    async def test_elect_center(self, manager):
        """Test electing an information center."""
        candidates = [f"agent_{i}" for i in range(5)]

        center = await manager.elect_center(
            candidates=candidates,
            criteria="connectivity",
        )

        assert center is not None
        assert center.member_count >= manager.MIN_CENTER_MEMBERS or center.member_count == len(candidates)

    @pytest.mark.asyncio
    async def test_elect_center_by_performance(self, manager):
        """Test electing center by performance criteria."""
        candidates = [f"agent_{i}" for i in range(5)]

        center = await manager.elect_center(
            candidates=candidates,
            criteria="performance",
        )

        assert center is not None

    @pytest.mark.asyncio
    async def test_aggregate_solution(self, manager):
        """Test aggregating a solution into a pattern."""
        center = await manager.create_center(
            agent_ids=["agent_1"],
            role=InformationCenterRole.AGGREGATOR,
        )

        pattern = await manager.aggregate_solution(
            center_id=center.center_id,
            solution={"approach": "recursive"},
            contributor_id="agent_1",
            pattern_type="algorithm",
            confidence=0.6,
        )

        assert pattern is not None
        assert pattern.pattern_type == "algorithm"
        assert "agent_1" in pattern.contributor_ids

    @pytest.mark.asyncio
    async def test_aggregate_similar_solution(self, manager):
        """Test aggregating similar solution merges patterns."""
        center = await manager.create_center(
            agent_ids=["agent_1"],
            role=InformationCenterRole.AGGREGATOR,
        )

        # First solution
        pattern1 = await manager.aggregate_solution(
            center_id=center.center_id,
            solution={"approach": "recursive"},
            contributor_id="agent_1",
        )

        initial_confidence = pattern1.confidence

        # Same solution from different agent
        pattern2 = await manager.aggregate_solution(
            center_id=center.center_id,
            solution={"approach": "recursive"},
            contributor_id="agent_2",
        )

        # Should be same pattern with increased confidence
        assert pattern2.pattern_id == pattern1.pattern_id
        assert pattern2.confidence > initial_confidence
        assert "agent_2" in pattern2.contributor_ids

    @pytest.mark.asyncio
    async def test_broadcast_pattern(self, manager):
        """Test broadcasting a pattern."""
        center = await manager.create_center(
            agent_ids=["agent_1"],
            role=InformationCenterRole.BROADCASTER,
        )

        # Add subscriber
        center.subscribe("agent_2")
        center.subscribe("agent_3")

        # Add pattern with high confidence
        pattern = await manager.aggregate_solution(
            center_id=center.center_id,
            solution={"test": True},
            contributor_id="agent_1",
            confidence=0.8,
        )

        count = await manager.broadcast_pattern(
            center_id=center.center_id,
            pattern_id=pattern.pattern_id,
        )

        assert count == 2  # Two subscribers

    @pytest.mark.asyncio
    async def test_broadcast_pattern_low_confidence(self, manager):
        """Test broadcasting blocked for low confidence patterns."""
        center = await manager.create_center(
            agent_ids=["agent_1"],
            role=InformationCenterRole.BROADCASTER,
        )

        center.subscribe("agent_2")

        # Add pattern with low confidence
        pattern = await manager.aggregate_solution(
            center_id=center.center_id,
            solution={"test": True},
            contributor_id="agent_1",
            confidence=0.1,  # Below threshold
        )

        count = await manager.broadcast_pattern(
            center_id=center.center_id,
            pattern_id=pattern.pattern_id,
        )

        assert count == 0  # Not broadcast

    @pytest.mark.asyncio
    async def test_archive_generation(self, manager):
        """Test archiving a generation."""
        center = await manager.create_center(
            agent_ids=["agent_1"],
            role=InformationCenterRole.ARCHIVE,
        )

        # Add some patterns
        for i in range(15):
            await manager.aggregate_solution(
                center_id=center.center_id,
                solution={"pattern": i},
                contributor_id="agent_1",
                confidence=0.3 + (i * 0.04),
            )

        initial_gen = manager.current_generation
        archived = await manager.archive_generation(center.center_id)

        assert archived == 15
        assert manager.current_generation == initial_gen + 1
        assert center.generation == manager.current_generation
        # Should have retained best patterns
        assert len(center.patterns) <= 10

    def test_get_best_pattern(self, manager):
        """Test getting best pattern by type."""
        # No centers yet
        result = manager.get_best_pattern("fake", "solution")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_best_pattern_with_data(self, manager):
        """Test getting best pattern with actual data."""
        center = await manager.create_center(
            agent_ids=["agent_1"],
            role=InformationCenterRole.AGGREGATOR,
        )

        await manager.aggregate_solution(
            center_id=center.center_id,
            solution={"v": 1},
            contributor_id="agent_1",
            pattern_type="algo",
            confidence=0.3,
        )

        await manager.aggregate_solution(
            center_id=center.center_id,
            solution={"v": 2},
            contributor_id="agent_1",
            pattern_type="algo",
            confidence=0.9,
        )

        best = manager.get_best_pattern(center.center_id, "algo")
        assert best is not None
        assert best.confidence == 0.9

    def test_get_archived_patterns(self, manager):
        """Test getting archived patterns."""
        # Initially empty
        patterns = manager.get_archived_patterns()
        assert len(patterns) == 0

    def test_get_agent_center(self, manager):
        """Test getting agent's center."""
        # Initially no center
        assert manager.get_agent_center("agent_1") is None

    @pytest.mark.asyncio
    async def test_get_agent_center_after_creation(self, manager):
        """Test getting agent's center after creation."""
        center = await manager.create_center(
            agent_ids=["agent_1", "agent_2"],
            role=InformationCenterRole.AGGREGATOR,
        )

        result = manager.get_agent_center("agent_1")
        assert result is not None
        assert result.center_id == center.center_id

    def test_to_dict(self, manager):
        """Test serialization."""
        data = manager.to_dict()

        assert "centers" in data
        assert "current_generation" in data
        assert "total_patterns" in data
        assert "archived_generations" in data
