"""
Consensus Engine - Coordinate multi-agent consensus building.

Higher-level abstraction over voting:
- Manage multiple voting sessions
- Handle conflict resolution
- Support iterative consensus building
- Track decision history
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Awaitable

from hive.decision.voting import (
    Vote,
    VotingSession,
    VotingStrategy,
    VotingResult,
)

logger = logging.getLogger("titan.decision.consensus")


# ============================================================================
# Types
# ============================================================================


@dataclass
class ConsensusConfig:
    """Configuration for consensus building."""

    # Voting strategy
    strategy: VotingStrategy = VotingStrategy.MAJORITY

    # Timeout for gathering votes (seconds)
    timeout: int = 300

    # Minimum participation required
    quorum: float = 0.5

    # For iterative consensus: max rounds
    max_rounds: int = 3

    # Minimum consensus strength to accept
    min_consensus_strength: float = 0.5

    # Whether to allow discussion between rounds
    allow_discussion: bool = True

    # Whether to reveal votes before final tally
    blind_voting: bool = False


@dataclass
class ConsensusResult:
    """Result of consensus building."""

    # Was consensus reached?
    reached: bool = False

    # Final decision (if reached)
    decision: str = ""
    alternatives: list[str] = field(default_factory=list)

    # Voting result
    voting_result: VotingResult | None = None

    # Consensus metrics
    strength: float = 0.0  # 0-1
    participation: float = 0.0  # 0-1
    rounds_used: int = 1

    # Metadata
    question: str = ""
    choices: list[str] = field(default_factory=list)
    agent_votes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reached": self.reached,
            "decision": self.decision,
            "alternatives": self.alternatives,
            "strength": self.strength,
            "participation": self.participation,
            "rounds_used": self.rounds_used,
            "question": self.question,
            "choices": self.choices,
            "agent_votes": self.agent_votes,
            "voting_result": self.voting_result.to_dict() if self.voting_result else None,
        }


# ============================================================================
# Consensus Engine
# ============================================================================


class ConsensusEngine:
    """
    Engine for building multi-agent consensus.

    Coordinates:
    - Creating voting sessions
    - Gathering votes from agents
    - Tallying results
    - Handling iterative refinement
    - Recording decision history
    """

    def __init__(
        self,
        config: ConsensusConfig | None = None,
    ) -> None:
        self._config = config or ConsensusConfig()
        self._active_sessions: dict[str, VotingSession] = {}
        self._history: list[ConsensusResult] = []

        # Callbacks for agent integration
        self._vote_requesters: dict[str, Callable[..., Awaitable[Vote]]] = {}

    def register_voter(
        self,
        agent_id: str,
        vote_callback: Callable[..., Awaitable[Vote]],
    ) -> None:
        """
        Register an agent as a voter.

        The callback will be called with (question, choices) and should
        return a Vote object.
        """
        self._vote_requesters[agent_id] = vote_callback
        logger.info(f"Registered voter: {agent_id}")

    def unregister_voter(self, agent_id: str) -> None:
        """Unregister an agent."""
        self._vote_requesters.pop(agent_id, None)

    async def build_consensus(
        self,
        question: str,
        choices: list[str],
        *,
        voter_ids: list[str] | None = None,
        config: ConsensusConfig | None = None,
    ) -> ConsensusResult:
        """
        Build consensus among agents.

        Args:
            question: The question to decide
            choices: Available options
            voter_ids: Specific agents to poll (None = all registered)
            config: Override default config

        Returns:
            ConsensusResult with decision and metrics
        """
        cfg = config or self._config
        voters = voter_ids or list(self._vote_requesters.keys())

        if not voters:
            logger.warning("No voters available for consensus")
            return ConsensusResult(
                reached=False,
                question=question,
                choices=choices,
            )

        logger.info(f"Building consensus: '{question}' with {len(voters)} voters")

        # Iterative consensus building
        best_result: ConsensusResult | None = None

        for round_num in range(1, cfg.max_rounds + 1):
            logger.info(f"Consensus round {round_num}/{cfg.max_rounds}")

            # Create voting session
            session = VotingSession(
                question=question,
                choices=choices,
                strategy=cfg.strategy,
                timeout_seconds=cfg.timeout,
                quorum=cfg.quorum,
            )

            for voter_id in voters:
                session.add_expected_voter(voter_id)

            self._active_sessions[session.id] = session

            # Gather votes
            votes = await self._gather_votes(
                session,
                voters,
                previous_result=best_result,
            )

            # Tally
            voting_result = session.tally()

            # Build result
            result = ConsensusResult(
                reached=voting_result.consensus_reached,
                decision=voting_result.winner,
                alternatives=voting_result.winning_choices,
                voting_result=voting_result,
                strength=voting_result.consensus_strength,
                participation=len(votes) / len(voters) if voters else 0,
                rounds_used=round_num,
                question=question,
                choices=choices,
                agent_votes={v.agent_id: v.choice for v in votes},
            )

            # Check if consensus is strong enough
            if result.reached and result.strength >= cfg.min_consensus_strength:
                logger.info(
                    f"Consensus reached: '{result.decision}' "
                    f"(strength={result.strength:.2f})"
                )
                self._history.append(result)
                return result

            best_result = result

            # Check if we should continue
            if round_num < cfg.max_rounds and cfg.allow_discussion:
                # Give agents info about current standings
                logger.info(f"Round {round_num} incomplete, continuing...")
            else:
                break

        # Use best result even if not ideal
        final_result = best_result or ConsensusResult(
            reached=False,
            question=question,
            choices=choices,
        )

        self._history.append(final_result)
        return final_result

    async def _gather_votes(
        self,
        session: VotingSession,
        voter_ids: list[str],
        previous_result: ConsensusResult | None = None,
    ) -> list[Vote]:
        """Gather votes from all voters."""
        tasks = []

        for voter_id in voter_ids:
            callback = self._vote_requesters.get(voter_id)
            if callback:
                tasks.append(
                    self._request_vote(
                        voter_id,
                        callback,
                        session,
                        previous_result,
                    )
                )

        # Gather with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self._config.timeout,
            )

            for result in results:
                if isinstance(result, Vote):
                    session.cast_vote(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Vote request failed: {result}")

        except asyncio.TimeoutError:
            logger.warning("Vote gathering timed out")

        return session.get_votes()

    async def _request_vote(
        self,
        voter_id: str,
        callback: Callable[..., Awaitable[Vote]],
        session: VotingSession,
        previous_result: ConsensusResult | None,
    ) -> Vote:
        """Request a vote from a single voter."""
        try:
            # Build context for voter
            context = {
                "question": session.question,
                "choices": session.choices,
                "strategy": session.strategy.value,
            }

            if previous_result and not self._config.blind_voting:
                context["previous_votes"] = previous_result.agent_votes
                context["current_leader"] = previous_result.decision

            # Call voter's callback
            vote = await callback(
                session.question,
                session.choices,
                context,
            )

            return vote

        except Exception as e:
            logger.error(f"Error requesting vote from {voter_id}: {e}")
            raise

    async def quick_poll(
        self,
        question: str,
        choices: list[str],
        voter_ids: list[str] | None = None,
    ) -> str:
        """
        Quick poll - single round majority vote.

        Returns winning choice or empty string if no votes.
        """
        result = await self.build_consensus(
            question,
            choices,
            voter_ids=voter_ids,
            config=ConsensusConfig(
                strategy=VotingStrategy.MAJORITY,
                timeout=60,
                max_rounds=1,
            ),
        )
        return result.decision

    async def weighted_decision(
        self,
        question: str,
        choices: list[str],
        weights: dict[str, float],
    ) -> ConsensusResult:
        """
        Make a weighted decision.

        Args:
            question: Question to decide
            choices: Options
            weights: agent_id -> weight mapping

        Returns:
            ConsensusResult
        """
        # Create special vote requesters that include weights
        original_requesters = dict(self._vote_requesters)

        async def weighted_vote_wrapper(
            q: str, c: list[str], ctx: dict[str, Any]
        ) -> Vote:
            voter_id = ctx.get("voter_id", "")
            callback = original_requesters.get(voter_id)
            if callback:
                vote = await callback(q, c, ctx)
                vote.weight = weights.get(voter_id, 1.0)
                return vote
            raise ValueError(f"No callback for {voter_id}")

        try:
            # Override with weighted wrappers
            for voter_id in weights:
                if voter_id in self._vote_requesters:
                    self._vote_requesters[voter_id] = weighted_vote_wrapper

            return await self.build_consensus(
                question,
                choices,
                voter_ids=list(weights.keys()),
                config=ConsensusConfig(
                    strategy=VotingStrategy.WEIGHTED,
                    max_rounds=1,
                ),
            )

        finally:
            # Restore original
            self._vote_requesters = original_requesters

    def get_history(self, limit: int = 10) -> list[ConsensusResult]:
        """Get recent decision history."""
        return self._history[-limit:]

    def get_active_sessions(self) -> list[dict[str, Any]]:
        """Get active voting sessions."""
        return [s.get_status() for s in self._active_sessions.values()]

    def clear_history(self) -> None:
        """Clear decision history."""
        self._history.clear()


# ============================================================================
# Helper Functions
# ============================================================================


def create_simple_vote(
    agent_id: str,
    agent_name: str,
    choice: str,
    confidence: float = 1.0,
    reasoning: str = "",
) -> Vote:
    """Create a simple vote."""
    return Vote(
        agent_id=agent_id,
        agent_name=agent_name,
        choice=choice,
        confidence=confidence,
        reasoning=reasoning,
    )


def create_ranked_vote(
    agent_id: str,
    agent_name: str,
    ranked_choices: list[str],
    confidence: float = 1.0,
) -> Vote:
    """Create a ranked choice vote."""
    return Vote(
        agent_id=agent_id,
        agent_name=agent_name,
        choice=ranked_choices[0] if ranked_choices else "",
        ranked_choices=ranked_choices,
        confidence=confidence,
    )


def create_approval_vote(
    agent_id: str,
    agent_name: str,
    approved_choices: list[str],
) -> Vote:
    """Create an approval vote."""
    return Vote(
        agent_id=agent_id,
        agent_name=agent_name,
        choice=approved_choices[0] if approved_choices else "",
        approved_choices=approved_choices,
    )
