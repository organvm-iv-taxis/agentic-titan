"""
Learning Module - Episodic learning for agents.

Provides:
- Episode: Record of agent interaction
- EpisodicMemory: Storage and retrieval of episodes
- LearningAgent: Base class for learning-enabled agents
"""

from agents.learning.episodic import (
    Episode,
    EpisodeOutcome,
    EpisodicMemory,
    LearningSignal,
)
from agents.learning.learner import LearningAgent

__all__ = [
    "Episode",
    "EpisodeOutcome",
    "EpisodicMemory",
    "LearningSignal",
    "LearningAgent",
]
