"""
Decision Module - Multi-agent voting and consensus.

Provides:
- VotingSession: Manage agent votes
- ConsensusEngine: Aggregate votes into decisions
- DecisionProtocol: Different voting strategies
"""

from hive.decision.voting import (
    Vote,
    VotingSession,
    VotingStrategy,
    VotingResult,
)
from hive.decision.consensus import (
    ConsensusEngine,
    ConsensusConfig,
    ConsensusResult,
)

__all__ = [
    "Vote",
    "VotingSession",
    "VotingStrategy",
    "VotingResult",
    "ConsensusEngine",
    "ConsensusConfig",
    "ConsensusResult",
]
