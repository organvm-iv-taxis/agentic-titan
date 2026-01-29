"""
Titan Analysis - Semantic Analysis and Contradiction Detection

Provides tools for analyzing agent outputs, detecting contradictions,
and performing dialectic synthesis of conflicting perspectives.

Modules:
- contradictions: Contradiction types and data structures
- detector: ContradictionDetector for semantic analysis
- dialectic: DialecticSynthesizer for resolving contradictions
"""

from titan.analysis.contradictions import (
    ContradictionType,
    Contradiction,
    ContradictionSeverity,
)
from titan.analysis.detector import ContradictionDetector
from titan.analysis.dialectic import DialecticSynthesizer, SynthesisResult

__all__ = [
    "ContradictionType",
    "Contradiction",
    "ContradictionSeverity",
    "ContradictionDetector",
    "DialecticSynthesizer",
    "SynthesisResult",
]
