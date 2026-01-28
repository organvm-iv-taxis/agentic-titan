"""
Titan Learning - Local learning and style adaptation.

Provides:
- LocalTrainer: Train on local code patterns
- StyleAdapter: Adapt to user coding style
- PatternExtractor: Extract coding patterns
"""

from titan.learning.local_trainer import (
    LocalTrainer,
    TrainingConfig,
    TrainingResult,
    StyleAdapter,
    CodingPattern,
    extract_patterns,
)

__all__ = [
    "LocalTrainer",
    "TrainingConfig",
    "TrainingResult",
    "StyleAdapter",
    "CodingPattern",
    "extract_patterns",
]
