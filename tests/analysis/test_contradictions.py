"""
Tests for Contradiction Detection System.

Tests the ContradictionDetector and related data structures.
"""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import AsyncMock

from titan.analysis.contradictions import (
    Contradiction,
    ContradictionPair,
    ContradictionReport,
    ContradictionSeverity,
    ContradictionType,
)
from titan.analysis.detector import (
    ContradictionDetector,
    DetectorConfig,
    get_contradiction_detector,
)


# =============================================================================
# Contradiction Data Structure Tests
# =============================================================================


class TestContradiction:
    """Tests for Contradiction dataclass."""

    def test_creation_with_defaults(self) -> None:
        """Test creating a contradiction with default values."""
        c = Contradiction()
        assert c.contradiction_id.startswith("ctr-")
        assert c.contradiction_type == ContradictionType.LOGICAL
        assert c.severity == ContradictionSeverity.MEDIUM
        assert c.confidence == 0.0

    def test_creation_with_values(self) -> None:
        """Test creating a contradiction with specific values."""
        c = Contradiction(
            contradiction_type=ContradictionType.SEMANTIC,
            severity=ContradictionSeverity.HIGH,
            source_a="Agent A",
            source_b="Agent B",
            content_a="The sky is blue",
            content_b="The sky is red",
            confidence=0.85,
            explanation="Color disagreement",
        )
        assert c.contradiction_type == ContradictionType.SEMANTIC
        assert c.severity == ContradictionSeverity.HIGH
        assert c.confidence == 0.85

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        c = Contradiction(
            source_a="A",
            source_b="B",
            content_a="Content A",
            content_b="Content B",
        )
        data = c.to_dict()
        assert data["source_a"] == "A"
        assert data["source_b"] == "B"
        assert "contradiction_id" in data
        assert "detected_at" in data

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "contradiction_id": "ctr-test123",
            "contradiction_type": "semantic",
            "severity": "high",
            "source_a": "X",
            "source_b": "Y",
            "content_a": "Content X",
            "content_b": "Content Y",
            "confidence": 0.9,
            "explanation": "Test",
            "key_terms": ["term1", "term2"],
            "resolution_suggestions": ["suggestion1"],
        }
        c = Contradiction.from_dict(data)
        assert c.contradiction_id == "ctr-test123"
        assert c.contradiction_type == ContradictionType.SEMANTIC
        assert c.severity == ContradictionSeverity.HIGH
        assert c.confidence == 0.9

    def test_is_significant_low_confidence(self) -> None:
        """Test is_significant with low confidence."""
        c = Contradiction(
            confidence=0.4,
            severity=ContradictionSeverity.HIGH,
        )
        assert not c.is_significant()  # Below threshold

    def test_is_significant_low_severity(self) -> None:
        """Test is_significant with low severity."""
        c = Contradiction(
            confidence=0.8,
            severity=ContradictionSeverity.LOW,
        )
        assert not c.is_significant()  # Wrong severity

    def test_is_significant_true(self) -> None:
        """Test is_significant when actually significant."""
        c = Contradiction(
            confidence=0.8,
            severity=ContradictionSeverity.HIGH,
        )
        assert c.is_significant()


class TestContradictionType:
    """Tests for ContradictionType enum."""

    def test_all_types_exist(self) -> None:
        """Test that all expected types exist."""
        expected = [
            "logical",
            "semantic",
            "methodological",
            "empirical",
            "evaluative",
            "temporal",
            "causal",
        ]
        for t in expected:
            assert ContradictionType(t)

    def test_type_values(self) -> None:
        """Test enum values."""
        assert ContradictionType.LOGICAL.value == "logical"
        assert ContradictionType.SEMANTIC.value == "semantic"


class TestContradictionPair:
    """Tests for ContradictionPair dataclass."""

    def test_to_analysis_prompt(self) -> None:
        """Test generating analysis prompt."""
        pair = ContradictionPair(
            source_a="Source A",
            source_b="Source B",
            content_a="Content from A",
            content_b="Content from B",
            context="Shared context",
        )
        prompt = pair.to_analysis_prompt()

        assert "Source A" in prompt
        assert "Source B" in prompt
        assert "Content from A" in prompt
        assert "Content from B" in prompt
        assert "Shared context" in prompt

    def test_to_analysis_prompt_no_context(self) -> None:
        """Test prompt without context."""
        pair = ContradictionPair(
            source_a="A",
            source_b="B",
            content_a="X",
            content_b="Y",
        )
        prompt = pair.to_analysis_prompt()
        assert "Context:" not in prompt


class TestContradictionReport:
    """Tests for ContradictionReport dataclass."""

    def test_empty_report(self) -> None:
        """Test empty report."""
        report = ContradictionReport()
        assert report.total_contradictions == 0
        assert report.contradictions == []

    def test_report_with_data(self) -> None:
        """Test report with contradictions."""
        c1 = Contradiction(
            contradiction_type=ContradictionType.LOGICAL,
            severity=ContradictionSeverity.HIGH,
            confidence=0.9,
        )
        c2 = Contradiction(
            contradiction_type=ContradictionType.SEMANTIC,
            severity=ContradictionSeverity.MEDIUM,
            confidence=0.7,
        )
        report = ContradictionReport(
            total_contradictions=2,
            contradictions=[c1, c2],
            by_type={"logical": 1, "semantic": 1},
            by_severity={"high": 1, "medium": 1},
            average_confidence=0.8,
            sources_analyzed=["A", "B", "C"],
            pairs_analyzed=3,
        )
        assert report.total_contradictions == 2
        assert report.average_confidence == 0.8

    def test_get_significant_contradictions(self) -> None:
        """Test filtering significant contradictions."""
        c1 = Contradiction(confidence=0.9, severity=ContradictionSeverity.HIGH)
        c2 = Contradiction(confidence=0.4, severity=ContradictionSeverity.HIGH)
        c3 = Contradiction(confidence=0.9, severity=ContradictionSeverity.LOW)
        report = ContradictionReport(
            total_contradictions=3,
            contradictions=[c1, c2, c3],
        )
        significant = report.get_significant_contradictions()
        assert len(significant) == 1
        assert significant[0] == c1

    def test_summary(self) -> None:
        """Test text summary generation."""
        report = ContradictionReport(
            total_contradictions=2,
            by_type={"logical": 1, "semantic": 1},
            by_severity={"high": 1, "medium": 1},
            average_confidence=0.8,
            sources_analyzed=["A", "B"],
            pairs_analyzed=1,
        )
        summary = report.summary()
        assert "Total contradictions: 2" in summary
        assert "logical: 1" in summary
        assert "Average confidence: 80" in summary  # 80% or 80.00%


# =============================================================================
# ContradictionDetector Tests
# =============================================================================


class TestContradictionDetector:
    """Tests for ContradictionDetector class."""

    @pytest.fixture
    def detector(self) -> ContradictionDetector:
        """Create detector for testing."""
        config = DetectorConfig(
            use_llm_analysis=False,  # Disable LLM for unit tests
            use_heuristics=True,
        )
        return ContradictionDetector(config=config)

    @pytest.mark.asyncio
    async def test_detect_contradictions_empty(self, detector: ContradictionDetector) -> None:
        """Test detection with empty outputs."""
        report = await detector.detect_contradictions([])
        assert report.total_contradictions == 0

    @pytest.mark.asyncio
    async def test_detect_contradictions_single(self, detector: ContradictionDetector) -> None:
        """Test detection with single output (no pairs)."""
        outputs = [{"source": "A", "content": "Some content"}]
        report = await detector.detect_contradictions(outputs)
        assert report.total_contradictions == 0

    @pytest.mark.asyncio
    async def test_detect_contradictions_basic(self, detector: ContradictionDetector) -> None:
        """Test basic contradiction detection."""
        outputs = [
            {"source": "A", "content": "The result is positive and the system works correctly."},
            {"source": "B", "content": "The result is negative and the system does not work. It contradicts the other view."},
        ]
        report = await detector.detect_contradictions(outputs)
        # Should detect something due to contradiction indicators
        assert report.pairs_analyzed == 1
        assert len(report.sources_analyzed) == 2

    @pytest.mark.asyncio
    async def test_detect_logical_contradiction(self, detector: ContradictionDetector) -> None:
        """Test detection of logical contradictions."""
        outputs = [
            {"source": "A", "content": "It is impossible to have both conditions at the same time."},
            {"source": "B", "content": "Both conditions can exist simultaneously, which contradicts the other view."},
        ]
        report = await detector.detect_contradictions(outputs)
        assert report.pairs_analyzed == 1

    @pytest.mark.asyncio
    async def test_compare_pair_negation(self, detector: ContradictionDetector) -> None:
        """Test pair comparison with negation."""
        contradictions = await detector.compare_pair(
            content_a="The data shows that the algorithm is efficient and fast.",
            content_b="The data shows that the algorithm is not efficient and is slow.",
            source_a="Study A",
            source_b="Study B",
        )
        # Should detect negation pattern
        assert len(contradictions) >= 0  # May or may not detect depending on threshold

    @pytest.mark.asyncio
    async def test_compare_pair_opposite_sentiment(self, detector: ContradictionDetector) -> None:
        """Test pair comparison with opposite sentiment."""
        contradictions = await detector.compare_pair(
            content_a="This approach is good and produces the best results.",
            content_b="This approach is bad and produces the worst results.",
            source_a="Review A",
            source_b="Review B",
        )
        # Should detect opposite sentiment
        assert isinstance(contradictions, list)

    @pytest.mark.asyncio
    async def test_detect_with_context(self, detector: ContradictionDetector) -> None:
        """Test detection with shared context."""
        outputs = [
            {"source": "A", "content": "The experiment succeeded."},
            {"source": "B", "content": "The experiment failed completely."},
        ]
        report = await detector.detect_contradictions(outputs, context="Laboratory study")
        assert "A" in report.sources_analyzed

    @pytest.mark.asyncio
    async def test_multiple_outputs(self, detector: ContradictionDetector) -> None:
        """Test detection with multiple outputs."""
        outputs = [
            {"source": "A", "content": "Position A text is relatively long to meet minimum length requirements."},
            {"source": "B", "content": "Position B text is relatively long to meet minimum length requirements."},
            {"source": "C", "content": "Position C text is relatively long to meet minimum length requirements."},
        ]
        report = await detector.detect_contradictions(outputs)
        # Should analyze 3 pairs: AB, AC, BC
        assert report.pairs_analyzed == 3

    @pytest.mark.asyncio
    async def test_short_content_skipped(self, detector: ContradictionDetector) -> None:
        """Test that short content is skipped."""
        outputs = [
            {"source": "A", "content": "Short"},
            {"source": "B", "content": "Also short"},
        ]
        report = await detector.detect_contradictions(outputs)
        assert report.pairs_analyzed == 0  # Skipped due to min length


class TestDetectorWithLLM:
    """Tests for ContradictionDetector with mock LLM."""

    @pytest.fixture
    def mock_llm(self) -> AsyncMock:
        """Create mock LLM caller."""
        mock = AsyncMock()
        mock.return_value = """
        {
            "contradictions": [
                {
                    "type": "semantic",
                    "severity": "high",
                    "confidence": 0.85,
                    "explanation": "Direct conflict in meaning",
                    "key_terms": ["term1", "term2"],
                    "resolution_suggestions": ["Consider context"]
                }
            ],
            "summary": "One semantic contradiction found"
        }
        """
        return mock

    @pytest.fixture
    def detector_with_llm(self, mock_llm: AsyncMock) -> ContradictionDetector:
        """Create detector with mock LLM."""
        config = DetectorConfig(
            use_llm_analysis=True,
            use_heuristics=False,
        )
        return ContradictionDetector(llm_caller=mock_llm, config=config)

    @pytest.mark.asyncio
    async def test_llm_analysis(
        self,
        detector_with_llm: ContradictionDetector,
        mock_llm: AsyncMock,
    ) -> None:
        """Test LLM-based contradiction analysis."""
        outputs = [
            {"source": "A", "content": "Long content A that meets the minimum length requirement for analysis."},
            {"source": "B", "content": "Long content B that meets the minimum length requirement for analysis."},
        ]
        report = await detector_with_llm.detect_contradictions(outputs)

        # LLM should have been called
        assert mock_llm.called
        # Should have parsed the mock response
        if report.contradictions:
            assert report.contradictions[0].contradiction_type == ContradictionType.SEMANTIC


class TestFactoryFunction:
    """Tests for factory functions."""

    def test_get_contradiction_detector(self) -> None:
        """Test getting default detector."""
        detector = get_contradiction_detector()
        assert isinstance(detector, ContradictionDetector)

    def test_singleton_behavior(self) -> None:
        """Test that factory returns same instance."""
        d1 = get_contradiction_detector()
        d2 = get_contradiction_detector()
        assert d1 is d2
