"""
Voting System - Multi-agent voting mechanisms.

Based on coliseum pattern:
- Democratic voting
- Weighted voting (expertise-based)
- Ranked choice voting
- Approval voting

Each vote includes:
- Choice(s) selected
- Confidence level
- Optional reasoning
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger("titan.decision.voting")


# ============================================================================
# Types
# ============================================================================


class VotingStrategy(str, Enum):
    """Voting strategies."""

    MAJORITY = "majority"  # Simple majority wins
    WEIGHTED = "weighted"  # Votes weighted by agent expertise
    RANKED_CHOICE = "ranked_choice"  # Ranked elimination
    APPROVAL = "approval"  # Approve any number of choices
    UNANIMOUS = "unanimous"  # Require 100% agreement


@dataclass
class Vote:
    """A single agent's vote."""

    agent_id: str
    agent_name: str

    # Primary choice
    choice: str

    # For ranked choice: list of choices in preference order
    ranked_choices: list[str] = field(default_factory=list)

    # For approval voting: all approved choices
    approved_choices: list[str] = field(default_factory=list)

    # Confidence in choice (0-1)
    confidence: float = 1.0

    # Weight multiplier for weighted voting
    weight: float = 1.0

    # Optional reasoning
    reasoning: str = ""

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "choice": self.choice,
            "ranked_choices": self.ranked_choices,
            "approved_choices": self.approved_choices,
            "confidence": self.confidence,
            "weight": self.weight,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class VotingResult:
    """Result of a voting session."""

    # Winning choice(s)
    winner: str
    winning_choices: list[str] = field(default_factory=list)

    # Vote counts
    vote_counts: dict[str, float] = field(default_factory=dict)
    total_votes: int = 0

    # Consensus metrics
    consensus_reached: bool = False
    consensus_strength: float = 0.0  # 0-1, how strong the agreement is

    # Breakdown
    strategy_used: VotingStrategy = VotingStrategy.MAJORITY
    rounds_needed: int = 1  # For ranked choice

    # Details
    votes: list[Vote] = field(default_factory=list)
    eliminated: list[str] = field(default_factory=list)  # For ranked choice

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "winning_choices": self.winning_choices,
            "vote_counts": self.vote_counts,
            "total_votes": self.total_votes,
            "consensus_reached": self.consensus_reached,
            "consensus_strength": self.consensus_strength,
            "strategy_used": self.strategy_used.value,
            "rounds_needed": self.rounds_needed,
            "eliminated": self.eliminated,
        }


# ============================================================================
# Voting Session
# ============================================================================


class VotingSession:
    """
    Manages a voting session among multiple agents.

    Usage:
        session = VotingSession(
            question="Which approach should we use?",
            choices=["A", "B", "C"],
            strategy=VotingStrategy.WEIGHTED,
        )

        # Agents cast votes
        session.cast_vote(Vote(agent_id="a1", choice="A", weight=1.5))
        session.cast_vote(Vote(agent_id="a2", choice="B", weight=1.0))

        # Tally results
        result = session.tally()
    """

    def __init__(
        self,
        question: str,
        choices: list[str],
        *,
        strategy: VotingStrategy = VotingStrategy.MAJORITY,
        timeout_seconds: int = 300,
        min_votes: int = 1,
        quorum: float = 0.5,  # Minimum participation
    ) -> None:
        self.id = f"vote_{uuid.uuid4().hex[:8]}"
        self.question = question
        self.choices = choices
        self.strategy = strategy
        self.timeout_seconds = timeout_seconds
        self.min_votes = min_votes
        self.quorum = quorum

        self._votes: dict[str, Vote] = {}  # agent_id -> vote
        self._expected_voters: set[str] = set()
        self._created_at = datetime.now()
        self._closed = False

    def add_expected_voter(self, agent_id: str) -> None:
        """Add an agent to the expected voters list."""
        self._expected_voters.add(agent_id)

    def cast_vote(self, vote: Vote) -> bool:
        """
        Cast a vote.

        Returns True if vote was accepted, False if rejected.
        """
        if self._closed:
            logger.warning(f"Vote rejected: session {self.id} is closed")
            return False

        if self._is_timed_out():
            self._closed = True
            logger.warning(f"Vote rejected: session {self.id} timed out")
            return False

        # Validate choice
        if vote.choice not in self.choices:
            logger.warning(f"Invalid choice '{vote.choice}' not in {self.choices}")
            return False

        # For ranked choice, validate all choices
        if self.strategy == VotingStrategy.RANKED_CHOICE:
            if vote.ranked_choices:
                for choice in vote.ranked_choices:
                    if choice not in self.choices:
                        logger.warning(f"Invalid ranked choice: {choice}")
                        return False
            else:
                # Use primary choice as single-item ranking
                vote.ranked_choices = [vote.choice]

        # For approval voting
        if self.strategy == VotingStrategy.APPROVAL:
            if vote.approved_choices:
                for choice in vote.approved_choices:
                    if choice not in self.choices:
                        logger.warning(f"Invalid approved choice: {choice}")
                        return False
            else:
                # Use primary choice as only approved
                vote.approved_choices = [vote.choice]

        # Record vote (overwrite if already voted)
        self._votes[vote.agent_id] = vote
        logger.info(
            f"Vote cast: {vote.agent_name} -> {vote.choice} "
            f"(confidence={vote.confidence}, weight={vote.weight})"
        )

        return True

    def get_vote(self, agent_id: str) -> Vote | None:
        """Get an agent's vote."""
        return self._votes.get(agent_id)

    def get_votes(self) -> list[Vote]:
        """Get all votes."""
        return list(self._votes.values())

    def has_quorum(self) -> bool:
        """Check if quorum is reached."""
        if not self._expected_voters:
            return len(self._votes) >= self.min_votes

        participation = len(self._votes) / len(self._expected_voters)
        return participation >= self.quorum

    def is_complete(self) -> bool:
        """Check if voting is complete."""
        if self._closed:
            return True

        # All expected voters have voted
        if self._expected_voters:
            if self._expected_voters == set(self._votes.keys()):
                return True

        return self._is_timed_out()

    def close(self) -> None:
        """Close voting session."""
        self._closed = True

    def tally(self) -> VotingResult:
        """
        Tally votes and determine winner.

        Uses the configured voting strategy.
        """
        if not self._votes:
            return VotingResult(
                winner="",
                total_votes=0,
                consensus_reached=False,
            )

        votes = list(self._votes.values())

        if self.strategy == VotingStrategy.MAJORITY:
            return self._tally_majority(votes)
        elif self.strategy == VotingStrategy.WEIGHTED:
            return self._tally_weighted(votes)
        elif self.strategy == VotingStrategy.RANKED_CHOICE:
            return self._tally_ranked_choice(votes)
        elif self.strategy == VotingStrategy.APPROVAL:
            return self._tally_approval(votes)
        elif self.strategy == VotingStrategy.UNANIMOUS:
            return self._tally_unanimous(votes)
        else:
            return self._tally_majority(votes)

    def _tally_majority(self, votes: list[Vote]) -> VotingResult:
        """Simple majority voting."""
        counts: dict[str, float] = {choice: 0 for choice in self.choices}

        for vote in votes:
            counts[vote.choice] += 1

        winner = max(counts.keys(), key=lambda c: counts[c])
        total = len(votes)
        winning_count = counts[winner]

        return VotingResult(
            winner=winner,
            winning_choices=[winner],
            vote_counts=counts,
            total_votes=total,
            consensus_reached=winning_count > total / 2,
            consensus_strength=winning_count / total if total > 0 else 0,
            strategy_used=VotingStrategy.MAJORITY,
            votes=votes,
        )

    def _tally_weighted(self, votes: list[Vote]) -> VotingResult:
        """Weighted voting (expertise-based)."""
        counts: dict[str, float] = {choice: 0 for choice in self.choices}
        total_weight = 0.0

        for vote in votes:
            effective_weight = vote.weight * vote.confidence
            counts[vote.choice] += effective_weight
            total_weight += effective_weight

        winner = max(counts.keys(), key=lambda c: counts[c])
        winning_weight = counts[winner]

        return VotingResult(
            winner=winner,
            winning_choices=[winner],
            vote_counts=counts,
            total_votes=len(votes),
            consensus_reached=winning_weight > total_weight / 2,
            consensus_strength=winning_weight / total_weight if total_weight > 0 else 0,
            strategy_used=VotingStrategy.WEIGHTED,
            votes=votes,
        )

    def _tally_ranked_choice(self, votes: list[Vote]) -> VotingResult:
        """Instant-runoff ranked choice voting."""
        # Each vote tracks current active ranking
        active_rankings = [
            list(v.ranked_choices) if v.ranked_choices else [v.choice]
            for v in votes
        ]
        remaining_choices = set(self.choices)
        eliminated: list[str] = []
        rounds = 0

        while len(remaining_choices) > 1:
            rounds += 1

            # Count first-choice votes
            counts: dict[str, int] = {c: 0 for c in remaining_choices}
            for ranking in active_rankings:
                # Find first non-eliminated choice
                for choice in ranking:
                    if choice in remaining_choices:
                        counts[choice] += 1
                        break

            # Check for majority
            total = len(votes)
            for choice, count in counts.items():
                if count > total / 2:
                    return VotingResult(
                        winner=choice,
                        winning_choices=[choice],
                        vote_counts={c: float(counts.get(c, 0)) for c in self.choices},
                        total_votes=total,
                        consensus_reached=True,
                        consensus_strength=count / total,
                        strategy_used=VotingStrategy.RANKED_CHOICE,
                        rounds_needed=rounds,
                        votes=votes,
                        eliminated=eliminated,
                    )

            # Eliminate lowest
            min_count = min(counts.values())
            losers = [c for c, count in counts.items() if count == min_count]
            loser = losers[0]  # Tie-break: first alphabetically
            remaining_choices.remove(loser)
            eliminated.append(loser)

        # Last remaining choice wins
        winner = list(remaining_choices)[0] if remaining_choices else ""
        final_counts = {c: float(0) for c in self.choices}
        for ranking in active_rankings:
            for choice in ranking:
                if choice == winner:
                    final_counts[choice] += 1
                    break

        return VotingResult(
            winner=winner,
            winning_choices=[winner],
            vote_counts=final_counts,
            total_votes=len(votes),
            consensus_reached=True,
            consensus_strength=final_counts.get(winner, 0) / len(votes) if votes else 0,
            strategy_used=VotingStrategy.RANKED_CHOICE,
            rounds_needed=rounds,
            votes=votes,
            eliminated=eliminated,
        )

    def _tally_approval(self, votes: list[Vote]) -> VotingResult:
        """Approval voting - voters approve multiple options."""
        counts: dict[str, float] = {choice: 0 for choice in self.choices}

        for vote in votes:
            approved = vote.approved_choices or [vote.choice]
            for choice in approved:
                if choice in counts:
                    counts[choice] += 1

        winner = max(counts.keys(), key=lambda c: counts[c])
        total = len(votes)

        # Can have multiple winners in approval voting
        max_approvals = counts[winner]
        winners = [c for c, count in counts.items() if count == max_approvals]

        return VotingResult(
            winner=winner,
            winning_choices=winners,
            vote_counts=counts,
            total_votes=total,
            consensus_reached=max_approvals > total * 0.5,
            consensus_strength=max_approvals / total if total > 0 else 0,
            strategy_used=VotingStrategy.APPROVAL,
            votes=votes,
        )

    def _tally_unanimous(self, votes: list[Vote]) -> VotingResult:
        """Unanimous voting - requires 100% agreement."""
        counts: dict[str, float] = {choice: 0 for choice in self.choices}

        for vote in votes:
            counts[vote.choice] += 1

        winner = max(counts.keys(), key=lambda c: counts[c])
        total = len(votes)
        winning_count = counts[winner]

        is_unanimous = winning_count == total

        return VotingResult(
            winner=winner if is_unanimous else "",
            winning_choices=[winner] if is_unanimous else [],
            vote_counts=counts,
            total_votes=total,
            consensus_reached=is_unanimous,
            consensus_strength=1.0 if is_unanimous else winning_count / total,
            strategy_used=VotingStrategy.UNANIMOUS,
            votes=votes,
        )

    def _is_timed_out(self) -> bool:
        """Check if session has timed out."""
        elapsed = datetime.now() - self._created_at
        return elapsed > timedelta(seconds=self.timeout_seconds)

    def get_status(self) -> dict[str, Any]:
        """Get session status."""
        return {
            "id": self.id,
            "question": self.question,
            "choices": self.choices,
            "strategy": self.strategy.value,
            "votes_cast": len(self._votes),
            "expected_voters": len(self._expected_voters),
            "has_quorum": self.has_quorum(),
            "is_complete": self.is_complete(),
            "closed": self._closed,
            "age_seconds": (datetime.now() - self._created_at).total_seconds(),
        }
