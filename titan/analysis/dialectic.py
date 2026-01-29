"""
Titan Analysis - Dialectic Synthesizer

Synthesizes contradictions into higher-order understanding using
the thesis-antithesis-synthesis dialectic pattern.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from titan.analysis.contradictions import (
    Contradiction,
    ContradictionSeverity,
    ContradictionType,
)

logger = logging.getLogger("titan.analysis.dialectic")


class SynthesisStrategy(str, Enum):
    """Strategies for synthesizing contradictions."""

    INTEGRATION = "integration"  # Combine both perspectives into unified view
    CONTEXTUALIZATION = "contextualization"  # Show both are valid in different contexts
    HIERARCHICAL = "hierarchical"  # One subsumes the other at a higher level
    COMPLEMENTARY = "complementary"  # Both address different aspects
    TEMPORAL = "temporal"  # Contradiction resolved by temporal sequence
    CONDITIONAL = "conditional"  # Both valid under different conditions


@dataclass
class SynthesisResult:
    """Result of dialectic synthesis of a contradiction."""

    synthesis_id: str = field(default_factory=lambda: f"syn-{uuid.uuid4().hex[:12]}")
    contradiction_id: str = ""  # The contradiction being synthesized

    # Dialectic components
    thesis: str = ""  # First position (content_a perspective)
    antithesis: str = ""  # Second position (content_b perspective)
    synthesis: str = ""  # Unified understanding

    # Strategy and quality
    strategy: SynthesisStrategy = SynthesisStrategy.INTEGRATION
    confidence: float = 0.0  # Confidence in synthesis quality (0-1)
    completeness: float = 0.0  # How well synthesis addresses both sides (0-1)

    # Insights
    key_insights: list[str] = field(default_factory=list)
    remaining_tensions: list[str] = field(default_factory=list)
    implications: list[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "synthesis_id": self.synthesis_id,
            "contradiction_id": self.contradiction_id,
            "thesis": self.thesis,
            "antithesis": self.antithesis,
            "synthesis": self.synthesis,
            "strategy": self.strategy.value,
            "confidence": self.confidence,
            "completeness": self.completeness,
            "key_insights": self.key_insights,
            "remaining_tensions": self.remaining_tensions,
            "implications": self.implications,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SynthesisResult":
        """Create from dictionary."""
        return cls(
            synthesis_id=data.get("synthesis_id", ""),
            contradiction_id=data.get("contradiction_id", ""),
            thesis=data.get("thesis", ""),
            antithesis=data.get("antithesis", ""),
            synthesis=data.get("synthesis", ""),
            strategy=SynthesisStrategy(data.get("strategy", "integration")),
            confidence=data.get("confidence", 0.0),
            completeness=data.get("completeness", 0.0),
            key_insights=data.get("key_insights", []),
            remaining_tensions=data.get("remaining_tensions", []),
            implications=data.get("implications", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DialecticConfig:
    """Configuration for DialecticSynthesizer."""

    # LLM settings
    use_llm: bool = True
    llm_model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 1500

    # Synthesis settings
    default_strategy: SynthesisStrategy = SynthesisStrategy.INTEGRATION
    min_confidence: float = 0.5

    # Output settings
    include_remaining_tensions: bool = True
    max_insights: int = 5


class DialecticSynthesizer:
    """
    Synthesizes contradictions using dialectic methodology.

    Applies the Hegelian thesis-antithesis-synthesis pattern to
    create higher-order understanding from conflicting perspectives.
    """

    def __init__(
        self,
        llm_caller: Callable[[str, str], Any] | None = None,
        config: DialecticConfig | None = None,
    ) -> None:
        """
        Initialize synthesizer.

        Args:
            llm_caller: Async function to call LLM
            config: Synthesizer configuration
        """
        self._llm_caller = llm_caller
        self._config = config or DialecticConfig()

    async def synthesize(
        self,
        contradictions: list[Contradiction],
    ) -> list[SynthesisResult]:
        """
        Synthesize multiple contradictions.

        Args:
            contradictions: List of contradictions to synthesize

        Returns:
            List of synthesis results, one per contradiction
        """
        results: list[SynthesisResult] = []

        for contradiction in contradictions:
            result = await self.synthesize_contradiction(contradiction)
            results.append(result)

        return results

    async def synthesize_contradiction(
        self,
        contradiction: Contradiction,
    ) -> SynthesisResult:
        """
        Synthesize a single contradiction.

        Args:
            contradiction: The contradiction to synthesize

        Returns:
            SynthesisResult with dialectic analysis
        """
        # Determine best synthesis strategy
        strategy = self._determine_strategy(contradiction)

        if self._config.use_llm and self._llm_caller:
            return await self._llm_synthesis(contradiction, strategy)
        else:
            return self._heuristic_synthesis(contradiction, strategy)

    def _determine_strategy(self, contradiction: Contradiction) -> SynthesisStrategy:
        """Determine best synthesis strategy based on contradiction type."""
        strategy_map = {
            ContradictionType.LOGICAL: SynthesisStrategy.CONTEXTUALIZATION,
            ContradictionType.SEMANTIC: SynthesisStrategy.INTEGRATION,
            ContradictionType.METHODOLOGICAL: SynthesisStrategy.COMPLEMENTARY,
            ContradictionType.EMPIRICAL: SynthesisStrategy.CONDITIONAL,
            ContradictionType.EVALUATIVE: SynthesisStrategy.HIERARCHICAL,
            ContradictionType.TEMPORAL: SynthesisStrategy.TEMPORAL,
            ContradictionType.CAUSAL: SynthesisStrategy.CONDITIONAL,
        }
        return strategy_map.get(contradiction.contradiction_type, self._config.default_strategy)

    async def _llm_synthesis(
        self,
        contradiction: Contradiction,
        strategy: SynthesisStrategy,
    ) -> SynthesisResult:
        """Use LLM for dialectic synthesis."""
        prompt = self._build_synthesis_prompt(contradiction, strategy)

        try:
            response = await self._llm_caller(prompt, self._config.llm_model)
            content = response if isinstance(response, str) else str(response)
            return self._parse_synthesis_response(content, contradiction, strategy)

        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")
            return self._heuristic_synthesis(contradiction, strategy)

    def _build_synthesis_prompt(
        self,
        contradiction: Contradiction,
        strategy: SynthesisStrategy,
    ) -> str:
        """Build prompt for LLM synthesis."""
        strategy_instructions = {
            SynthesisStrategy.INTEGRATION: "Create a unified view that incorporates the valid aspects of both perspectives.",
            SynthesisStrategy.CONTEXTUALIZATION: "Show how both perspectives are valid in different contexts or domains.",
            SynthesisStrategy.HIERARCHICAL: "Identify if one perspective operates at a higher level that encompasses the other.",
            SynthesisStrategy.COMPLEMENTARY: "Demonstrate how both perspectives address different aspects of the same phenomenon.",
            SynthesisStrategy.TEMPORAL: "Explain how the contradiction is resolved through temporal sequence or evolution.",
            SynthesisStrategy.CONDITIONAL: "Identify the conditions under which each perspective holds true.",
        }

        return f"""Perform a dialectic synthesis of the following contradiction:

THESIS (Position A - from {contradiction.source_a}):
{contradiction.content_a}

ANTITHESIS (Position B - from {contradiction.source_b}):
{contradiction.content_b}

Contradiction Type: {contradiction.contradiction_type.value}
Current Explanation: {contradiction.explanation}

SYNTHESIS STRATEGY: {strategy.value}
Instructions: {strategy_instructions.get(strategy, "")}

Provide a dialectic synthesis that transcends the opposition. Respond in JSON:
{{
    "thesis_summary": "Brief summary of first position",
    "antithesis_summary": "Brief summary of second position",
    "synthesis": "The higher-order understanding that reconciles both",
    "key_insights": ["insight1", "insight2", "insight3"],
    "remaining_tensions": ["tension1"],
    "implications": ["implication1", "implication2"],
    "confidence": 0.0-1.0,
    "completeness": 0.0-1.0
}}"""

    def _parse_synthesis_response(
        self,
        response: str,
        contradiction: Contradiction,
        strategy: SynthesisStrategy,
    ) -> SynthesisResult:
        """Parse LLM synthesis response."""
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                return self._heuristic_synthesis(contradiction, strategy)

            data = json.loads(json_match.group())

            return SynthesisResult(
                contradiction_id=contradiction.contradiction_id,
                thesis=data.get("thesis_summary", contradiction.content_a[:200]),
                antithesis=data.get("antithesis_summary", contradiction.content_b[:200]),
                synthesis=data.get("synthesis", ""),
                strategy=strategy,
                confidence=data.get("confidence", 0.7),
                completeness=data.get("completeness", 0.7),
                key_insights=data.get("key_insights", [])[:self._config.max_insights],
                remaining_tensions=data.get("remaining_tensions", []) if self._config.include_remaining_tensions else [],
                implications=data.get("implications", []),
                metadata={"source": "llm_synthesis"},
            )

        except json.JSONDecodeError:
            logger.warning("Failed to parse synthesis response as JSON")
            return self._heuristic_synthesis(contradiction, strategy)

    def _heuristic_synthesis(
        self,
        contradiction: Contradiction,
        strategy: SynthesisStrategy,
    ) -> SynthesisResult:
        """Generate heuristic synthesis without LLM."""
        # Generate synthesis based on strategy
        synthesis_templates = {
            SynthesisStrategy.INTEGRATION: (
                f"While {contradiction.source_a} emphasizes one aspect and "
                f"{contradiction.source_b} emphasizes another, both perspectives "
                "can be integrated into a more comprehensive understanding that "
                "acknowledges the validity of each within a broader framework."
            ),
            SynthesisStrategy.CONTEXTUALIZATION: (
                f"The apparent contradiction between {contradiction.source_a} and "
                f"{contradiction.source_b} resolves when we recognize that each "
                "perspective applies within its appropriate context or domain."
            ),
            SynthesisStrategy.HIERARCHICAL: (
                f"Upon deeper analysis, one perspective may operate at a level "
                "that encompasses the other, suggesting a hierarchical relationship "
                "rather than direct opposition."
            ),
            SynthesisStrategy.COMPLEMENTARY: (
                f"Rather than contradicting each other, {contradiction.source_a} and "
                f"{contradiction.source_b} address different facets of the same "
                "phenomenon, providing complementary insights."
            ),
            SynthesisStrategy.TEMPORAL: (
                "The contradiction may be resolved through temporal understanding, "
                "where different perspectives apply at different stages or times."
            ),
            SynthesisStrategy.CONDITIONAL: (
                "Both perspectives may be correct under different conditions. "
                "Identifying these conditions resolves the apparent contradiction."
            ),
        }

        synthesis = synthesis_templates.get(
            strategy,
            "Further analysis needed to synthesize these perspectives."
        )

        # Generate insights from key terms
        insights = []
        if contradiction.key_terms:
            insights.append(
                f"Key concepts involved: {', '.join(contradiction.key_terms[:3])}"
            )
        if contradiction.resolution_suggestions:
            insights.append(
                f"Suggested resolution path: {contradiction.resolution_suggestions[0]}"
            )

        # Identify remaining tensions
        tensions = []
        if contradiction.severity in (ContradictionSeverity.HIGH, ContradictionSeverity.CRITICAL):
            tensions.append(
                "Deep structural tension remains between fundamental assumptions"
            )

        return SynthesisResult(
            contradiction_id=contradiction.contradiction_id,
            thesis=f"{contradiction.source_a}'s perspective: {contradiction.content_a[:150]}...",
            antithesis=f"{contradiction.source_b}'s perspective: {contradiction.content_b[:150]}...",
            synthesis=synthesis,
            strategy=strategy,
            confidence=0.5,  # Heuristic synthesis has lower confidence
            completeness=0.4,
            key_insights=insights,
            remaining_tensions=tensions if self._config.include_remaining_tensions else [],
            implications=[],
            metadata={"source": "heuristic_synthesis"},
        )

    async def batch_synthesize(
        self,
        contradictions: list[Contradiction],
        parallel: bool = True,
    ) -> list[SynthesisResult]:
        """
        Synthesize multiple contradictions, optionally in parallel.

        Args:
            contradictions: Contradictions to synthesize
            parallel: Whether to run syntheses in parallel

        Returns:
            List of synthesis results
        """
        if parallel:
            tasks = [
                self.synthesize_contradiction(c)
                for c in contradictions
            ]
            return await asyncio.gather(*tasks)
        else:
            return await self.synthesize(contradictions)


@dataclass
class DialecticReport:
    """Complete report of dialectic analysis."""

    contradictions_analyzed: int = 0
    syntheses: list[SynthesisResult] = field(default_factory=list)
    strategies_used: dict[str, int] = field(default_factory=dict)
    average_confidence: float = 0.0
    average_completeness: float = 0.0
    all_insights: list[str] = field(default_factory=list)
    all_remaining_tensions: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_results(cls, results: list[SynthesisResult]) -> "DialecticReport":
        """Build report from synthesis results."""
        strategies: dict[str, int] = {}
        all_insights: list[str] = []
        all_tensions: list[str] = []

        for r in results:
            strategies[r.strategy.value] = strategies.get(r.strategy.value, 0) + 1
            all_insights.extend(r.key_insights)
            all_tensions.extend(r.remaining_tensions)

        avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0.0
        avg_completeness = sum(r.completeness for r in results) / len(results) if results else 0.0

        return cls(
            contradictions_analyzed=len(results),
            syntheses=results,
            strategies_used=strategies,
            average_confidence=avg_confidence,
            average_completeness=avg_completeness,
            all_insights=list(set(all_insights)),
            all_remaining_tensions=list(set(all_tensions)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contradictions_analyzed": self.contradictions_analyzed,
            "syntheses": [s.to_dict() for s in self.syntheses],
            "strategies_used": self.strategies_used,
            "average_confidence": self.average_confidence,
            "average_completeness": self.average_completeness,
            "all_insights": self.all_insights,
            "all_remaining_tensions": self.all_remaining_tensions,
            "created_at": self.created_at.isoformat(),
        }


# Factory function
_default_synthesizer: DialecticSynthesizer | None = None


def get_dialectic_synthesizer() -> DialecticSynthesizer:
    """Get default dialectic synthesizer instance."""
    global _default_synthesizer
    if _default_synthesizer is None:
        _default_synthesizer = DialecticSynthesizer()
    return _default_synthesizer
