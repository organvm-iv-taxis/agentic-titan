"""
Learning Module - Episodic learning for agents.

Provides:
- Episode: Record of agent interaction
- EpisodicMemory: Storage and retrieval of episodes
- LearningAgent: Base class for learning-enabled agents
- Entity extraction for memory enrichment
"""

from agents.learning.episodic import (
    Episode,
    EpisodeOutcome,
    EpisodicMemory,
    LearningSignal,
)
from agents.learning.learner import LearningAgent
from agents.learning.entity_extractor import (
    ExtractedEntities,
    extract_entities,
    extract_entities_pattern,
    extract_keywords,
)

__all__ = [
    "Episode",
    "EpisodeOutcome",
    "EpisodicMemory",
    "LearningSignal",
    "LearningAgent",
    # Entity extraction
    "ExtractedEntities",
    "extract_entities",
    "extract_entities_pattern",
    "extract_keywords",
]
