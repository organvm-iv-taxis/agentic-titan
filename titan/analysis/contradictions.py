"""
Titan Analysis - Contradiction Types and Data Structures

Defines the types and structures for representing contradictions
detected between agent outputs or inquiry stages.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ContradictionType(str, Enum):
    """Types of contradictions that can be detected."""

    LOGICAL = "logical"  # A and not-A (direct logical contradiction)
    SEMANTIC = "semantic"  # Conflicting meanings or interpretations
    METHODOLOGICAL = "methodological"  # Different approaches to same problem
    EMPIRICAL = "empirical"  # Conflicting factual claims or evidence
    EVALUATIVE = "evaluative"  # Different value judgments
    TEMPORAL = "temporal"  # Contradictions about timing or sequence
    CAUSAL = "causal"  # Conflicting causal claims


class ContradictionSeverity(str, Enum):
    """Severity levels for contradictions."""

    LOW = "low"  # Minor inconsistency, doesn't affect conclusions
    MEDIUM = "medium"  # Notable contradiction, may affect some conclusions
    HIGH = "high"  # Significant contradiction, affects core conclusions
    CRITICAL = "critical"  # Fundamental contradiction, invalidates one position


@dataclass
class Contradiction:
    """
    A detected contradiction between two pieces of content.

    Represents a conflict between outputs from different agents, stages,
    or perspectives within the inquiry system.
    """

    contradiction_id: str = field(default_factory=lambda: f"ctr-{uuid.uuid4().hex[:12]}")
    contradiction_type: ContradictionType = ContradictionType.LOGICAL
    severity: ContradictionSeverity = ContradictionSeverity.MEDIUM

    # Source information
    source_a: str = ""  # Identifier for first source (agent/stage name)
    source_b: str = ""  # Identifier for second source
    content_a: str = ""  # The conflicting content from source A
    content_b: str = ""  # The conflicting content from source B

    # Analysis
    confidence: float = 0.0  # Confidence that this is a real contradiction (0-1)
    explanation: str = ""  # Human-readable explanation of the contradiction
    key_terms: list[str] = field(default_factory=list)  # Key terms involved

    # Resolution suggestions
    resolution_suggestions: list[str] = field(default_factory=list)
    preferred_resolution: str | None = None

    # Metadata
    detected_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "contradiction_id": self.contradiction_id,
            "contradiction_type": self.contradiction_type.value,
            "severity": self.severity.value,
            "source_a": self.source_a,
            "source_b": self.source_b,
            "content_a": self.content_a,
            "content_b": self.content_b,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "key_terms": self.key_terms,
            "resolution_suggestions": self.resolution_suggestions,
            "preferred_resolution": self.preferred_resolution,
            "detected_at": self.detected_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Contradiction":
        """Create from dictionary."""
        return cls(
            contradiction_id=data.get("contradiction_id", ""),
            contradiction_type=ContradictionType(data.get("contradiction_type", "logical")),
            severity=ContradictionSeverity(data.get("severity", "medium")),
            source_a=data.get("source_a", ""),
            source_b=data.get("source_b", ""),
            content_a=data.get("content_a", ""),
            content_b=data.get("content_b", ""),
            confidence=data.get("confidence", 0.0),
            explanation=data.get("explanation", ""),
            key_terms=data.get("key_terms", []),
            resolution_suggestions=data.get("resolution_suggestions", []),
            preferred_resolution=data.get("preferred_resolution"),
            metadata=data.get("metadata", {}),
        )

    def is_significant(self, threshold: float = 0.6) -> bool:
        """Check if contradiction is significant enough to warrant attention."""
        return self.confidence >= threshold and self.severity in (
            ContradictionSeverity.HIGH,
            ContradictionSeverity.CRITICAL,
        )


@dataclass
class ContradictionPair:
    """A pair of content items to analyze for contradictions."""

    source_a: str
    source_b: str
    content_a: str
    content_b: str
    context: str | None = None  # Optional shared context

    def to_analysis_prompt(self) -> str:
        """Generate a prompt for contradiction analysis."""
        context_part = f"\nContext: {self.context}" if self.context else ""
        return f"""Analyze these two statements for contradictions:

Statement A (from {self.source_a}):
{self.content_a}

Statement B (from {self.source_b}):
{self.content_b}{context_part}

Identify any contradictions between these statements. For each contradiction found:
1. Type: logical, semantic, methodological, empirical, evaluative, temporal, or causal
2. Severity: low, medium, high, or critical
3. Explanation: Brief description of the conflict
4. Key terms involved
5. Possible resolutions"""


@dataclass
class ContradictionReport:
    """Summary report of contradictions in an analysis."""

    total_contradictions: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)
    contradictions: list[Contradiction] = field(default_factory=list)
    analysis_timestamp: datetime = field(default_factory=datetime.now)

    # Statistics
    average_confidence: float = 0.0
    sources_analyzed: list[str] = field(default_factory=list)
    pairs_analyzed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_contradictions": self.total_contradictions,
            "by_type": self.by_type,
            "by_severity": self.by_severity,
            "contradictions": [c.to_dict() for c in self.contradictions],
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "average_confidence": self.average_confidence,
            "sources_analyzed": self.sources_analyzed,
            "pairs_analyzed": self.pairs_analyzed,
        }

    def get_significant_contradictions(self, threshold: float = 0.6) -> list[Contradiction]:
        """Get only significant contradictions."""
        return [c for c in self.contradictions if c.is_significant(threshold)]

    def summary(self) -> str:
        """Generate a text summary of the report."""
        lines = [
            f"Contradiction Analysis Report",
            f"=" * 40,
            f"Total contradictions: {self.total_contradictions}",
            f"Sources analyzed: {len(self.sources_analyzed)}",
            f"Pairs analyzed: {self.pairs_analyzed}",
            f"Average confidence: {self.average_confidence:.2%}",
            "",
            "By Type:",
        ]
        for ctype, count in self.by_type.items():
            lines.append(f"  - {ctype}: {count}")

        lines.extend(["", "By Severity:"])
        for sev, count in self.by_severity.items():
            lines.append(f"  - {sev}: {count}")

        return "\n".join(lines)
