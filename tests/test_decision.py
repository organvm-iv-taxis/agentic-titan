"""
Tests for voting and consensus systems.
"""

import pytest

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
    create_simple_vote,
    create_ranked_vote,
    create_approval_vote,
)


# ============================================================================
# Vote Tests
# ============================================================================


class TestVote:
    """Test Vote dataclass."""

    def test_create_vote(self) -> None:
        """Test basic vote creation."""
        vote = Vote(
            agent_id="agent_1",
            agent_name="Alice",
            choice="A",
            confidence=0.9,
        )

        assert vote.agent_id == "agent_1"
        assert vote.choice == "A"
        assert vote.confidence == 0.9
        assert vote.weight == 1.0

    def test_vote_to_dict(self) -> None:
        """Test vote serialization."""
        vote = Vote(
            agent_id="a1",
            agent_name="Bob",
            choice="B",
            reasoning="Best option",
        )

        d = vote.to_dict()

        assert d["agent_id"] == "a1"
        assert d["choice"] == "B"
        assert d["reasoning"] == "Best option"


# ============================================================================
# Voting Session Tests
# ============================================================================


class TestVotingSession:
    """Test VotingSession."""

    def test_create_session(self) -> None:
        """Test session creation."""
        session = VotingSession(
            question="Which option?",
            choices=["A", "B", "C"],
        )

        assert session.question == "Which option?"
        assert session.choices == ["A", "B", "C"]
        assert session.strategy == VotingStrategy.MAJORITY

    def test_cast_vote(self) -> None:
        """Test casting a vote."""
        session = VotingSession(
            question="Pick one",
            choices=["X", "Y"],
        )

        vote = Vote(agent_id="a1", agent_name="Alice", choice="X")
        result = session.cast_vote(vote)

        assert result is True
        assert session.get_vote("a1") == vote

    def test_reject_invalid_choice(self) -> None:
        """Test rejecting invalid choices."""
        session = VotingSession(
            question="Pick one",
            choices=["A", "B"],
        )

        vote = Vote(agent_id="a1", agent_name="Alice", choice="Z")
        result = session.cast_vote(vote)

        assert result is False

    def test_overwrite_vote(self) -> None:
        """Test that re-voting overwrites previous vote."""
        session = VotingSession(
            question="Pick",
            choices=["A", "B"],
        )

        vote1 = Vote(agent_id="a1", agent_name="Alice", choice="A")
        vote2 = Vote(agent_id="a1", agent_name="Alice", choice="B")

        session.cast_vote(vote1)
        session.cast_vote(vote2)

        assert session.get_vote("a1").choice == "B"
        assert len(session.get_votes()) == 1


class TestMajorityVoting:
    """Test majority voting strategy."""

    def test_clear_majority(self) -> None:
        """Test with clear majority."""
        session = VotingSession(
            question="Pick",
            choices=["A", "B", "C"],
            strategy=VotingStrategy.MAJORITY,
        )

        session.cast_vote(Vote(agent_id="1", agent_name="a", choice="A"))
        session.cast_vote(Vote(agent_id="2", agent_name="b", choice="A"))
        session.cast_vote(Vote(agent_id="3", agent_name="c", choice="B"))

        result = session.tally()

        assert result.winner == "A"
        assert result.vote_counts["A"] == 2
        assert result.vote_counts["B"] == 1
        assert result.consensus_reached is True

    def test_no_majority(self) -> None:
        """Test with no clear majority."""
        session = VotingSession(
            question="Pick",
            choices=["A", "B", "C"],
            strategy=VotingStrategy.MAJORITY,
        )

        session.cast_vote(Vote(agent_id="1", agent_name="a", choice="A"))
        session.cast_vote(Vote(agent_id="2", agent_name="b", choice="B"))
        session.cast_vote(Vote(agent_id="3", agent_name="c", choice="C"))

        result = session.tally()

        # Still picks one, but no consensus
        assert result.winner in ["A", "B", "C"]
        assert result.consensus_reached is False


class TestWeightedVoting:
    """Test weighted voting strategy."""

    def test_weighted_votes(self) -> None:
        """Test weighted voting."""
        session = VotingSession(
            question="Pick",
            choices=["A", "B"],
            strategy=VotingStrategy.WEIGHTED,
        )

        # Agent with weight 2.0 votes A
        session.cast_vote(Vote(agent_id="1", agent_name="expert", choice="A", weight=2.0))
        # Two agents with weight 1.0 vote B
        session.cast_vote(Vote(agent_id="2", agent_name="normal1", choice="B", weight=1.0))
        session.cast_vote(Vote(agent_id="3", agent_name="normal2", choice="B", weight=1.0))

        result = session.tally()

        # A has weight 2.0, B has weight 2.0 - tie, A wins (first in choices)
        assert result.vote_counts["A"] == 2.0
        assert result.vote_counts["B"] == 2.0

    def test_confidence_affects_weight(self) -> None:
        """Test that confidence multiplies weight."""
        session = VotingSession(
            question="Pick",
            choices=["A", "B"],
            strategy=VotingStrategy.WEIGHTED,
        )

        # High weight but low confidence
        session.cast_vote(Vote(
            agent_id="1", agent_name="unsure",
            choice="A", weight=2.0, confidence=0.5,
        ))
        # Normal weight, high confidence
        session.cast_vote(Vote(
            agent_id="2", agent_name="sure",
            choice="B", weight=1.0, confidence=1.0,
        ))

        result = session.tally()

        # A: 2.0 * 0.5 = 1.0
        # B: 1.0 * 1.0 = 1.0
        assert result.vote_counts["A"] == 1.0
        assert result.vote_counts["B"] == 1.0


class TestRankedChoiceVoting:
    """Test ranked choice voting."""

    def test_first_round_majority(self) -> None:
        """Test ranked choice with first-round majority."""
        session = VotingSession(
            question="Rank",
            choices=["A", "B", "C"],
            strategy=VotingStrategy.RANKED_CHOICE,
        )

        session.cast_vote(create_ranked_vote("1", "a", ["A", "B", "C"]))
        session.cast_vote(create_ranked_vote("2", "b", ["A", "C", "B"]))
        session.cast_vote(create_ranked_vote("3", "c", ["B", "A", "C"]))

        result = session.tally()

        assert result.winner == "A"
        assert result.rounds_needed == 1

    def test_instant_runoff(self) -> None:
        """Test instant runoff elimination."""
        session = VotingSession(
            question="Rank",
            choices=["A", "B", "C"],
            strategy=VotingStrategy.RANKED_CHOICE,
        )

        # No first-round majority
        session.cast_vote(create_ranked_vote("1", "a", ["A", "B", "C"]))
        session.cast_vote(create_ranked_vote("2", "b", ["B", "A", "C"]))
        session.cast_vote(create_ranked_vote("3", "c", ["C", "B", "A"]))
        session.cast_vote(create_ranked_vote("4", "d", ["C", "A", "B"]))
        session.cast_vote(create_ranked_vote("5", "e", ["B", "C", "A"]))

        result = session.tally()

        # Should eliminate lowest and redistribute
        assert result.winner in ["A", "B", "C"]
        assert len(result.eliminated) > 0


class TestApprovalVoting:
    """Test approval voting."""

    def test_approval_voting(self) -> None:
        """Test basic approval voting."""
        session = VotingSession(
            question="Approve",
            choices=["A", "B", "C"],
            strategy=VotingStrategy.APPROVAL,
        )

        session.cast_vote(create_approval_vote("1", "a", ["A", "B"]))
        session.cast_vote(create_approval_vote("2", "b", ["B", "C"]))
        session.cast_vote(create_approval_vote("3", "c", ["A", "B"]))

        result = session.tally()

        # B approved by all 3
        assert result.winner == "B"
        assert result.vote_counts["B"] == 3
        assert result.vote_counts["A"] == 2
        assert result.vote_counts["C"] == 1


class TestUnanimousVoting:
    """Test unanimous voting."""

    def test_unanimous_success(self) -> None:
        """Test unanimous agreement."""
        session = VotingSession(
            question="Agree",
            choices=["A", "B"],
            strategy=VotingStrategy.UNANIMOUS,
        )

        session.cast_vote(Vote(agent_id="1", agent_name="a", choice="A"))
        session.cast_vote(Vote(agent_id="2", agent_name="b", choice="A"))
        session.cast_vote(Vote(agent_id="3", agent_name="c", choice="A"))

        result = session.tally()

        assert result.winner == "A"
        assert result.consensus_reached is True
        assert result.consensus_strength == 1.0

    def test_unanimous_failure(self) -> None:
        """Test when unanimity not reached."""
        session = VotingSession(
            question="Agree",
            choices=["A", "B"],
            strategy=VotingStrategy.UNANIMOUS,
        )

        session.cast_vote(Vote(agent_id="1", agent_name="a", choice="A"))
        session.cast_vote(Vote(agent_id="2", agent_name="b", choice="A"))
        session.cast_vote(Vote(agent_id="3", agent_name="c", choice="B"))

        result = session.tally()

        assert result.winner == ""
        assert result.consensus_reached is False


# ============================================================================
# Consensus Engine Tests
# ============================================================================


class TestConsensusEngine:
    """Test ConsensusEngine."""

    @pytest.fixture
    def engine(self) -> ConsensusEngine:
        return ConsensusEngine()

    def test_create_engine(self, engine: ConsensusEngine) -> None:
        """Test engine creation."""
        assert engine._config is not None
        assert len(engine._vote_requesters) == 0

    @pytest.mark.asyncio
    async def test_no_voters(self, engine: ConsensusEngine) -> None:
        """Test with no registered voters."""
        result = await engine.build_consensus(
            "Question?",
            ["A", "B"],
        )

        assert result.reached is False
        assert result.decision == ""

    @pytest.mark.asyncio
    async def test_with_voters(self, engine: ConsensusEngine) -> None:
        """Test with registered voters."""
        async def voter1(q: str, c: list[str], ctx: dict) -> Vote:
            return create_simple_vote("v1", "Voter1", "A")

        async def voter2(q: str, c: list[str], ctx: dict) -> Vote:
            return create_simple_vote("v2", "Voter2", "A")

        engine.register_voter("v1", voter1)
        engine.register_voter("v2", voter2)

        result = await engine.build_consensus(
            "Pick one",
            ["A", "B"],
        )

        assert result.reached is True
        assert result.decision == "A"

    @pytest.mark.asyncio
    async def test_quick_poll(self, engine: ConsensusEngine) -> None:
        """Test quick poll."""
        async def voter(q: str, c: list[str], ctx: dict) -> Vote:
            return create_simple_vote("v1", "Voter", "B")

        engine.register_voter("v1", voter)

        result = await engine.quick_poll("Quick?", ["A", "B"])

        assert result == "B"

    def test_history(self, engine: ConsensusEngine) -> None:
        """Test history tracking."""
        engine._history.append(ConsensusResult(
            reached=True,
            decision="A",
        ))

        history = engine.get_history()

        assert len(history) == 1
        assert history[0].decision == "A"


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestHelperFunctions:
    """Test helper functions."""

    def test_create_simple_vote(self) -> None:
        """Test simple vote creation."""
        vote = create_simple_vote("a1", "Alice", "X", 0.8, "Good choice")

        assert vote.agent_id == "a1"
        assert vote.choice == "X"
        assert vote.confidence == 0.8
        assert vote.reasoning == "Good choice"

    def test_create_ranked_vote(self) -> None:
        """Test ranked vote creation."""
        vote = create_ranked_vote("a1", "Alice", ["X", "Y", "Z"])

        assert vote.choice == "X"
        assert vote.ranked_choices == ["X", "Y", "Z"]

    def test_create_approval_vote(self) -> None:
        """Test approval vote creation."""
        vote = create_approval_vote("a1", "Alice", ["X", "Y"])

        assert vote.choice == "X"
        assert vote.approved_choices == ["X", "Y"]
