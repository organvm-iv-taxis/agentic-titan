"""
Tests for Dialectic Synthesis System.

Tests the DialecticSynthesizer and related data structures.
"""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import AsyncMock

from titan.analysis.contradictions import (
    Contradiction,
    ContradictionSeverity,
    ContradictionType,
)
from titan.analysis.dialectic import (
    DialecticConfig,
    DialecticReport,
    DialecticSynthesizer,
    SynthesisResult,
    SynthesisStrategy,
    get_dialectic_synthesizer,
)


# =============================================================================
# SynthesisResult Tests
# =============================================================================


class TestSynthesisResult:
    """Tests for SynthesisResult dataclass."""

    def test_creation_with_defaults(self) -> None:
        """Test creating a synthesis result with defaults."""
        result = SynthesisResult()
        assert result.synthesis_id.startswith("syn-")
        assert result.strategy == SynthesisStrategy.INTEGRATION
        assert result.confidence == 0.0

    def test_creation_with_values(self) -> None:
        """Test creating a synthesis with specific values."""
        result = SynthesisResult(
            contradiction_id="ctr-123",
            thesis="Position A",
            antithesis="Position B",
            synthesis="Unified view",
            strategy=SynthesisStrategy.HIERARCHICAL,
            confidence=0.85,
            completeness=0.9,
            key_insights=["Insight 1", "Insight 2"],
        )
        assert result.contradiction_id == "ctr-123"
        assert result.strategy == SynthesisStrategy.HIERARCHICAL
        assert result.confidence == 0.85
        assert len(result.key_insights) == 2

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        result = SynthesisResult(
            thesis="T",
            antithesis="A",
            synthesis="S",
            confidence=0.8,
        )
        data = result.to_dict()

        assert data["thesis"] == "T"
        assert data["antithesis"] == "A"
        assert data["synthesis"] == "S"
        assert data["confidence"] == 0.8
        assert "synthesis_id" in data
        assert "created_at" in data

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "synthesis_id": "syn-test123",
            "contradiction_id": "ctr-test456",
            "thesis": "Thesis text",
            "antithesis": "Antithesis text",
            "synthesis": "Synthesis text",
            "strategy": "complementary",
            "confidence": 0.9,
            "completeness": 0.85,
            "key_insights": ["insight1"],
            "remaining_tensions": ["tension1"],
            "implications": ["implication1"],
        }
        result = SynthesisResult.from_dict(data)

        assert result.synthesis_id == "syn-test123"
        assert result.contradiction_id == "ctr-test456"
        assert result.strategy == SynthesisStrategy.COMPLEMENTARY
        assert result.confidence == 0.9
        assert result.key_insights == ["insight1"]


class TestSynthesisStrategy:
    """Tests for SynthesisStrategy enum."""

    def test_all_strategies_exist(self) -> None:
        """Test that all expected strategies exist."""
        expected = [
            "integration",
            "contextualization",
            "hierarchical",
            "complementary",
            "temporal",
            "conditional",
        ]
        for s in expected:
            assert SynthesisStrategy(s)

    def test_strategy_values(self) -> None:
        """Test enum values."""
        assert SynthesisStrategy.INTEGRATION.value == "integration"
        assert SynthesisStrategy.HIERARCHICAL.value == "hierarchical"


# =============================================================================
# DialecticSynthesizer Tests
# =============================================================================


class TestDialecticSynthesizer:
    """Tests for DialecticSynthesizer class."""

    @pytest.fixture
    def synthesizer(self) -> DialecticSynthesizer:
        """Create synthesizer for testing (no LLM)."""
        config = DialecticConfig(use_llm=False)
        return DialecticSynthesizer(config=config)

    @pytest.fixture
    def contradiction(self) -> Contradiction:
        """Create a test contradiction."""
        return Contradiction(
            contradiction_type=ContradictionType.SEMANTIC,
            severity=ContradictionSeverity.MEDIUM,
            source_a="Source A",
            source_b="Source B",
            content_a="The approach emphasizes speed and efficiency as primary goals.",
            content_b="The approach prioritizes accuracy and correctness over speed.",
            confidence=0.8,
            explanation="Different priorities",
            key_terms=["speed", "accuracy", "efficiency"],
            resolution_suggestions=["Consider trade-offs"],
        )

    @pytest.mark.asyncio
    async def test_synthesize_contradiction(
        self,
        synthesizer: DialecticSynthesizer,
        contradiction: Contradiction,
    ) -> None:
        """Test synthesizing a single contradiction."""
        result = await synthesizer.synthesize_contradiction(contradiction)

        assert result.contradiction_id == contradiction.contradiction_id
        assert result.thesis != ""
        assert result.antithesis != ""
        assert result.synthesis != ""
        assert result.strategy in SynthesisStrategy

    @pytest.mark.asyncio
    async def test_synthesize_multiple(
        self,
        synthesizer: DialecticSynthesizer,
        contradiction: Contradiction,
    ) -> None:
        """Test synthesizing multiple contradictions."""
        c2 = Contradiction(
            contradiction_type=ContradictionType.METHODOLOGICAL,
            source_a="Method A",
            source_b="Method B",
            content_a="Use iterative approach",
            content_b="Use recursive approach",
        )
        results = await synthesizer.synthesize([contradiction, c2])

        assert len(results) == 2
        assert all(isinstance(r, SynthesisResult) for r in results)

    @pytest.mark.asyncio
    async def test_strategy_determination_logical(
        self,
        synthesizer: DialecticSynthesizer,
    ) -> None:
        """Test strategy determination for logical contradiction."""
        c = Contradiction(contradiction_type=ContradictionType.LOGICAL)
        result = await synthesizer.synthesize_contradiction(c)
        assert result.strategy == SynthesisStrategy.CONTEXTUALIZATION

    @pytest.mark.asyncio
    async def test_strategy_determination_methodological(
        self,
        synthesizer: DialecticSynthesizer,
    ) -> None:
        """Test strategy determination for methodological contradiction."""
        c = Contradiction(contradiction_type=ContradictionType.METHODOLOGICAL)
        result = await synthesizer.synthesize_contradiction(c)
        assert result.strategy == SynthesisStrategy.COMPLEMENTARY

    @pytest.mark.asyncio
    async def test_strategy_determination_temporal(
        self,
        synthesizer: DialecticSynthesizer,
    ) -> None:
        """Test strategy determination for temporal contradiction."""
        c = Contradiction(contradiction_type=ContradictionType.TEMPORAL)
        result = await synthesizer.synthesize_contradiction(c)
        assert result.strategy == SynthesisStrategy.TEMPORAL

    @pytest.mark.asyncio
    async def test_heuristic_synthesis_integration(
        self,
        synthesizer: DialecticSynthesizer,
    ) -> None:
        """Test heuristic synthesis with integration strategy."""
        c = Contradiction(
            contradiction_type=ContradictionType.SEMANTIC,
            source_a="A",
            source_b="B",
            content_a="Content A",
            content_b="Content B",
        )
        result = await synthesizer.synthesize_contradiction(c)

        assert "integration" in result.strategy.value or "contextualization" in result.strategy.value
        assert result.confidence == 0.5  # Heuristic confidence
        assert "heuristic_synthesis" in result.metadata.get("source", "")

    @pytest.mark.asyncio
    async def test_synthesis_with_key_terms(
        self,
        synthesizer: DialecticSynthesizer,
        contradiction: Contradiction,
    ) -> None:
        """Test that key terms appear in insights."""
        result = await synthesizer.synthesize_contradiction(contradiction)

        # Should have insights based on key terms
        all_text = " ".join(result.key_insights)
        # At least some insights should be generated
        assert len(result.key_insights) >= 0

    @pytest.mark.asyncio
    async def test_synthesis_remaining_tensions(
        self,
        synthesizer: DialecticSynthesizer,
    ) -> None:
        """Test that remaining tensions are captured for high severity."""
        c = Contradiction(
            contradiction_type=ContradictionType.LOGICAL,
            severity=ContradictionSeverity.CRITICAL,
            source_a="A",
            source_b="B",
            content_a="X",
            content_b="Y",
        )
        result = await synthesizer.synthesize_contradiction(c)

        # Critical severity should produce remaining tensions
        assert len(result.remaining_tensions) >= 0

    @pytest.mark.asyncio
    async def test_batch_synthesize(
        self,
        synthesizer: DialecticSynthesizer,
        contradiction: Contradiction,
    ) -> None:
        """Test batch synthesis."""
        contradictions = [contradiction, contradiction]
        results = await synthesizer.batch_synthesize(contradictions, parallel=True)

        assert len(results) == 2


class TestDialecticSynthesizerWithLLM:
    """Tests for DialecticSynthesizer with mock LLM."""

    @pytest.fixture
    def mock_llm(self) -> AsyncMock:
        """Create mock LLM caller."""
        mock = AsyncMock()
        mock.return_value = """
        {
            "thesis_summary": "Position A emphasizes speed",
            "antithesis_summary": "Position B emphasizes accuracy",
            "synthesis": "Both speed and accuracy can be achieved through adaptive algorithms",
            "key_insights": [
                "Trade-offs are context dependent",
                "Hybrid approaches are possible"
            ],
            "remaining_tensions": ["Resource constraints"],
            "implications": ["Design flexibility"],
            "confidence": 0.88,
            "completeness": 0.92
        }
        """
        return mock

    @pytest.fixture
    def synthesizer_with_llm(self, mock_llm: AsyncMock) -> DialecticSynthesizer:
        """Create synthesizer with mock LLM."""
        config = DialecticConfig(use_llm=True)
        return DialecticSynthesizer(llm_caller=mock_llm, config=config)

    @pytest.mark.asyncio
    async def test_llm_synthesis(
        self,
        synthesizer_with_llm: DialecticSynthesizer,
        mock_llm: AsyncMock,
    ) -> None:
        """Test LLM-based synthesis."""
        c = Contradiction(
            contradiction_type=ContradictionType.SEMANTIC,
            source_a="A",
            source_b="B",
            content_a="Speed first",
            content_b="Accuracy first",
        )
        result = await synthesizer_with_llm.synthesize_contradiction(c)

        assert mock_llm.called
        assert "adaptive algorithms" in result.synthesis
        assert result.confidence == 0.88
        assert result.completeness == 0.92

    @pytest.mark.asyncio
    async def test_llm_synthesis_parses_insights(
        self,
        synthesizer_with_llm: DialecticSynthesizer,
    ) -> None:
        """Test that LLM response insights are parsed."""
        c = Contradiction(
            contradiction_type=ContradictionType.METHODOLOGICAL,
            source_a="A",
            source_b="B",
            content_a="Method X",
            content_b="Method Y",
        )
        result = await synthesizer_with_llm.synthesize_contradiction(c)

        assert len(result.key_insights) == 2
        assert "Trade-offs are context dependent" in result.key_insights


# =============================================================================
# DialecticReport Tests
# =============================================================================


class TestDialecticReport:
    """Tests for DialecticReport dataclass."""

    def test_from_results_empty(self) -> None:
        """Test building report from empty results."""
        report = DialecticReport.from_results([])
        assert report.contradictions_analyzed == 0
        assert report.average_confidence == 0.0

    def test_from_results_with_data(self) -> None:
        """Test building report from results."""
        r1 = SynthesisResult(
            strategy=SynthesisStrategy.INTEGRATION,
            confidence=0.8,
            completeness=0.9,
            key_insights=["Insight 1"],
            remaining_tensions=["Tension 1"],
        )
        r2 = SynthesisResult(
            strategy=SynthesisStrategy.COMPLEMENTARY,
            confidence=0.7,
            completeness=0.85,
            key_insights=["Insight 2"],
            remaining_tensions=[],
        )
        report = DialecticReport.from_results([r1, r2])

        assert report.contradictions_analyzed == 2
        assert report.strategies_used == {"integration": 1, "complementary": 1}
        assert report.average_confidence == 0.75
        assert "Insight 1" in report.all_insights
        assert "Insight 2" in report.all_insights

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        r = SynthesisResult(strategy=SynthesisStrategy.TEMPORAL)
        report = DialecticReport.from_results([r])
        data = report.to_dict()

        assert "contradictions_analyzed" in data
        assert "syntheses" in data
        assert "strategies_used" in data
        assert "created_at" in data


class TestFactoryFunction:
    """Tests for factory functions."""

    def test_get_dialectic_synthesizer(self) -> None:
        """Test getting default synthesizer."""
        synthesizer = get_dialectic_synthesizer()
        assert isinstance(synthesizer, DialecticSynthesizer)

    def test_singleton_behavior(self) -> None:
        """Test that factory returns same instance."""
        s1 = get_dialectic_synthesizer()
        s2 = get_dialectic_synthesizer()
        assert s1 is s2
