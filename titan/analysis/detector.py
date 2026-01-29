"""
Titan Analysis - Contradiction Detector

Detects semantic contradictions between agent outputs using
LLM-based analysis and keyword matching heuristics.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from titan.analysis.contradictions import (
    Contradiction,
    ContradictionPair,
    ContradictionReport,
    ContradictionSeverity,
    ContradictionType,
)

if TYPE_CHECKING:
    from adapters.router import LLMRouter

logger = logging.getLogger("titan.analysis.detector")


# Patterns that suggest contradictions
CONTRADICTION_PATTERNS = {
    ContradictionType.LOGICAL: [
        r"\bnot\b.*\bsame\b",
        r"\bopposite\b",
        r"\bcontradicts?\b",
        r"\binconsistent\b",
        r"\bimpossible\b.*\bboth\b",
    ],
    ContradictionType.TEMPORAL: [
        r"\bbefore\b.*\bafter\b",
        r"\bsimultaneous\b.*\bsequential\b",
        r"\bfirst\b.*\blast\b",
    ],
    ContradictionType.CAUSAL: [
        r"\bcauses?\b.*\bdoes not cause\b",
        r"\bleads to\b.*\bprevents\b",
        r"\bresults in\b.*\bno effect\b",
    ],
}

# Words that often indicate contradiction
CONTRADICTION_INDICATORS = [
    "however",
    "but",
    "although",
    "whereas",
    "contrary",
    "opposite",
    "disagree",
    "conflict",
    "contradict",
    "inconsistent",
    "incompatible",
]


@dataclass
class DetectorConfig:
    """Configuration for ContradictionDetector."""

    # Analysis thresholds
    min_confidence_threshold: float = 0.3  # Minimum confidence to report
    high_confidence_threshold: float = 0.7  # Threshold for high confidence

    # Content processing
    max_content_length: int = 2000  # Max chars to analyze per source
    min_content_length: int = 50  # Min chars for meaningful analysis

    # LLM settings
    use_llm_analysis: bool = True  # Use LLM for deep analysis
    llm_model: str = "claude-3-5-sonnet-20241022"
    max_llm_tokens: int = 1000

    # Heuristics
    use_heuristics: bool = True  # Use pattern matching heuristics
    heuristic_weight: float = 0.3  # Weight of heuristics vs LLM


class ContradictionDetector:
    """
    Detects contradictions between agent outputs.

    Uses a combination of:
    1. Pattern matching heuristics for quick detection
    2. LLM-based semantic analysis for deeper understanding
    3. Keyword overlap and divergence analysis
    """

    def __init__(
        self,
        llm_caller: Callable[[str, str], Any] | None = None,
        config: DetectorConfig | None = None,
    ) -> None:
        """
        Initialize detector.

        Args:
            llm_caller: Async function to call LLM: (prompt, model) -> response
            config: Detector configuration
        """
        self._llm_caller = llm_caller
        self._config = config or DetectorConfig()
        self._analysis_cache: dict[str, list[Contradiction]] = {}

    async def detect_contradictions(
        self,
        outputs: list[dict[str, Any]],
        context: str | None = None,
    ) -> ContradictionReport:
        """
        Detect contradictions across multiple outputs.

        Args:
            outputs: List of output dicts with 'source' and 'content' keys
            context: Optional shared context for all outputs

        Returns:
            ContradictionReport with all detected contradictions
        """
        if len(outputs) < 2:
            return ContradictionReport()

        all_contradictions: list[Contradiction] = []
        pairs_analyzed = 0
        sources = [o.get("source", f"source_{i}") for i, o in enumerate(outputs)]

        # Compare all pairs
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                source_a = outputs[i].get("source", f"source_{i}")
                source_b = outputs[j].get("source", f"source_{j}")
                content_a = outputs[i].get("content", "")
                content_b = outputs[j].get("content", "")

                if not content_a or not content_b:
                    continue

                # Truncate if needed
                if len(content_a) > self._config.max_content_length:
                    content_a = content_a[: self._config.max_content_length] + "..."
                if len(content_b) > self._config.max_content_length:
                    content_b = content_b[: self._config.max_content_length] + "..."

                # Skip if too short
                if (
                    len(content_a) < self._config.min_content_length
                    or len(content_b) < self._config.min_content_length
                ):
                    continue

                pairs_analyzed += 1

                # Detect contradictions in this pair
                pair_contradictions = await self.compare_pair(
                    content_a=content_a,
                    content_b=content_b,
                    source_a=source_a,
                    source_b=source_b,
                    context=context,
                )
                all_contradictions.extend(pair_contradictions)

        # Build report
        return self._build_report(all_contradictions, sources, pairs_analyzed)

    async def compare_pair(
        self,
        content_a: str,
        content_b: str,
        source_a: str = "A",
        source_b: str = "B",
        context: str | None = None,
    ) -> list[Contradiction]:
        """
        Compare two content pieces for contradictions.

        Args:
            content_a: First content
            content_b: Second content
            source_a: Source identifier for first content
            source_b: Source identifier for second content
            context: Optional context

        Returns:
            List of detected contradictions
        """
        contradictions: list[Contradiction] = []

        # Run heuristic analysis
        if self._config.use_heuristics:
            heuristic_results = self._heuristic_analysis(
                content_a, content_b, source_a, source_b
            )
            contradictions.extend(heuristic_results)

        # Run LLM analysis
        if self._config.use_llm_analysis and self._llm_caller:
            llm_results = await self._llm_analysis(
                content_a, content_b, source_a, source_b, context
            )
            contradictions.extend(llm_results)

        # Merge and dedupe
        merged = self._merge_contradictions(contradictions)

        # Filter by confidence
        filtered = [
            c
            for c in merged
            if c.confidence >= self._config.min_confidence_threshold
        ]

        return filtered

    def _heuristic_analysis(
        self,
        content_a: str,
        content_b: str,
        source_a: str,
        source_b: str,
    ) -> list[Contradiction]:
        """Use pattern matching for quick contradiction detection."""
        contradictions: list[Contradiction] = []
        combined = f"{content_a}\n{content_b}".lower()

        # Check for contradiction indicator words
        indicator_count = sum(1 for ind in CONTRADICTION_INDICATORS if ind in combined)

        # Check for type-specific patterns
        for ctype, patterns in CONTRADICTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    # Calculate confidence based on indicator presence
                    confidence = min(0.5 + (indicator_count * 0.1), 0.8)

                    contradictions.append(
                        Contradiction(
                            contradiction_type=ctype,
                            severity=ContradictionSeverity.MEDIUM,
                            source_a=source_a,
                            source_b=source_b,
                            content_a=content_a[:500],
                            content_b=content_b[:500],
                            confidence=confidence * self._config.heuristic_weight,
                            explanation=f"Pattern match detected: {pattern}",
                            key_terms=self._extract_key_terms(content_a, content_b),
                            metadata={"detection_method": "heuristic"},
                        )
                    )
                    break  # One contradiction per type from heuristics

        # Check for direct negation patterns
        negation_score = self._check_negation_patterns(content_a, content_b)
        if negation_score > 0.3:
            contradictions.append(
                Contradiction(
                    contradiction_type=ContradictionType.LOGICAL,
                    severity=ContradictionSeverity.HIGH if negation_score > 0.6 else ContradictionSeverity.MEDIUM,
                    source_a=source_a,
                    source_b=source_b,
                    content_a=content_a[:500],
                    content_b=content_b[:500],
                    confidence=negation_score * self._config.heuristic_weight,
                    explanation="Direct negation pattern detected",
                    key_terms=self._extract_key_terms(content_a, content_b),
                    metadata={"detection_method": "negation_analysis"},
                )
            )

        return contradictions

    async def _llm_analysis(
        self,
        content_a: str,
        content_b: str,
        source_a: str,
        source_b: str,
        context: str | None,
    ) -> list[Contradiction]:
        """Use LLM for deep semantic contradiction analysis."""
        if not self._llm_caller:
            return []

        pair = ContradictionPair(
            source_a=source_a,
            source_b=source_b,
            content_a=content_a,
            content_b=content_b,
            context=context,
        )

        prompt = f"""{pair.to_analysis_prompt()}

Respond in JSON format:
{{
    "contradictions": [
        {{
            "type": "logical|semantic|methodological|empirical|evaluative|temporal|causal",
            "severity": "low|medium|high|critical",
            "confidence": 0.0-1.0,
            "explanation": "Brief explanation",
            "key_terms": ["term1", "term2"],
            "resolution_suggestions": ["suggestion1", "suggestion2"]
        }}
    ],
    "summary": "Brief overall summary"
}}

If no contradictions are found, return: {{"contradictions": [], "summary": "No contradictions found"}}"""

        try:
            response = await self._llm_caller(prompt, self._config.llm_model)
            content = response if isinstance(response, str) else str(response)

            # Parse JSON from response
            contradictions = self._parse_llm_response(
                content, source_a, source_b, content_a, content_b
            )
            return contradictions

        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return []

    def _parse_llm_response(
        self,
        response: str,
        source_a: str,
        source_b: str,
        content_a: str,
        content_b: str,
    ) -> list[Contradiction]:
        """Parse LLM response into Contradiction objects."""
        contradictions: list[Contradiction] = []

        try:
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                return []

            data = json.loads(json_match.group())
            raw_contradictions = data.get("contradictions", [])

            for raw in raw_contradictions:
                try:
                    ctype = ContradictionType(raw.get("type", "semantic"))
                except ValueError:
                    ctype = ContradictionType.SEMANTIC

                try:
                    severity = ContradictionSeverity(raw.get("severity", "medium"))
                except ValueError:
                    severity = ContradictionSeverity.MEDIUM

                contradictions.append(
                    Contradiction(
                        contradiction_type=ctype,
                        severity=severity,
                        source_a=source_a,
                        source_b=source_b,
                        content_a=content_a[:500],
                        content_b=content_b[:500],
                        confidence=raw.get("confidence", 0.7),
                        explanation=raw.get("explanation", ""),
                        key_terms=raw.get("key_terms", []),
                        resolution_suggestions=raw.get("resolution_suggestions", []),
                        metadata={"detection_method": "llm_analysis"},
                    )
                )

        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
        except Exception as e:
            logger.warning(f"Error parsing LLM response: {e}")

        return contradictions

    def _check_negation_patterns(self, content_a: str, content_b: str) -> float:
        """Check for direct negation between content pieces."""
        content_a_lower = content_a.lower()
        content_b_lower = content_b.lower()

        # Extract key sentences
        sentences_a = re.split(r"[.!?]", content_a_lower)
        sentences_b = re.split(r"[.!?]", content_b_lower)

        score = 0.0

        for sent_a in sentences_a:
            sent_a = sent_a.strip()
            if len(sent_a) < 20:
                continue

            for sent_b in sentences_b:
                sent_b = sent_b.strip()
                if len(sent_b) < 20:
                    continue

                # Check for negation patterns
                if self._is_negation(sent_a, sent_b):
                    score = max(score, 0.7)
                elif self._has_opposite_sentiment(sent_a, sent_b):
                    score = max(score, 0.5)

        return score

    def _is_negation(self, sent_a: str, sent_b: str) -> bool:
        """Check if one sentence is roughly the negation of another."""
        negation_words = ["not", "never", "no", "none", "neither", "cannot", "won't", "don't", "doesn't", "isn't", "aren't"]

        words_a = set(sent_a.split())
        words_b = set(sent_b.split())

        # Check for shared content words
        shared = words_a & words_b
        shared = {w for w in shared if len(w) > 3 and w not in negation_words}

        if len(shared) < 2:
            return False

        # Check if one has negation and other doesn't
        neg_in_a = any(neg in sent_a for neg in negation_words)
        neg_in_b = any(neg in sent_b for neg in negation_words)

        return neg_in_a != neg_in_b

    def _has_opposite_sentiment(self, sent_a: str, sent_b: str) -> bool:
        """Check for opposite sentiment indicators."""
        positive = ["good", "better", "best", "positive", "correct", "right", "true", "agree", "support"]
        negative = ["bad", "worse", "worst", "negative", "incorrect", "wrong", "false", "disagree", "oppose"]

        pos_a = any(p in sent_a for p in positive)
        neg_a = any(n in sent_a for n in negative)
        pos_b = any(p in sent_b for p in positive)
        neg_b = any(n in sent_b for n in negative)

        # Opposite sentiment if one is positive and other is negative
        return (pos_a and neg_b) or (neg_a and pos_b)

    def _extract_key_terms(self, content_a: str, content_b: str) -> list[str]:
        """Extract key terms from content."""
        # Simple extraction: find common significant words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                      "have", "has", "had", "do", "does", "did", "will", "would", "could",
                      "should", "may", "might", "must", "shall", "can", "to", "of", "in",
                      "for", "on", "with", "at", "by", "from", "or", "and", "but", "if",
                      "then", "than", "so", "as", "it", "its", "this", "that", "these",
                      "those", "they", "them", "their", "we", "us", "our", "you", "your"}

        words_a = set(re.findall(r"\b\w{4,}\b", content_a.lower()))
        words_b = set(re.findall(r"\b\w{4,}\b", content_b.lower()))

        common = (words_a & words_b) - stop_words
        return list(common)[:10]

    def _merge_contradictions(
        self,
        contradictions: list[Contradiction],
    ) -> list[Contradiction]:
        """Merge similar contradictions, keeping highest confidence."""
        if not contradictions:
            return []

        # Group by type
        by_type: dict[ContradictionType, list[Contradiction]] = {}
        for c in contradictions:
            by_type.setdefault(c.contradiction_type, []).append(c)

        # For each type, keep the one with highest confidence
        merged: list[Contradiction] = []
        for ctype, clist in by_type.items():
            best = max(clist, key=lambda x: x.confidence)

            # Merge resolution suggestions from all
            all_suggestions: list[str] = []
            for c in clist:
                all_suggestions.extend(c.resolution_suggestions)

            best.resolution_suggestions = list(set(all_suggestions))[:5]
            merged.append(best)

        return merged

    def _build_report(
        self,
        contradictions: list[Contradiction],
        sources: list[str],
        pairs_analyzed: int,
    ) -> ContradictionReport:
        """Build a contradiction report from analysis results."""
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}

        for c in contradictions:
            by_type[c.contradiction_type.value] = by_type.get(c.contradiction_type.value, 0) + 1
            by_severity[c.severity.value] = by_severity.get(c.severity.value, 0) + 1

        avg_confidence = (
            sum(c.confidence for c in contradictions) / len(contradictions)
            if contradictions else 0.0
        )

        return ContradictionReport(
            total_contradictions=len(contradictions),
            by_type=by_type,
            by_severity=by_severity,
            contradictions=contradictions,
            average_confidence=avg_confidence,
            sources_analyzed=sources,
            pairs_analyzed=pairs_analyzed,
        )


# Factory function
_default_detector: ContradictionDetector | None = None


def get_contradiction_detector() -> ContradictionDetector:
    """Get default contradiction detector instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = ContradictionDetector()
    return _default_detector
