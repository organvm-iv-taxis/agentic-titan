"""Tests for digital agent archetypes: SwarmIntelligence, DAO."""

from __future__ import annotations

import pytest

from agents.archetypes.swarm_intelligence import SwarmIntelligenceAgent, SwarmAlgorithm
from agents.archetypes.dao import DAOAgent, ProposalType, ProposalStatus, VoteChoice
from agents.framework.base_agent import AgentState


class TestSwarmIntelligenceAgent:
    """Tests for SwarmIntelligenceAgent (PSO/ACO inspired)."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(SwarmIntelligenceAgent)
        # Name may include algorithm type
        assert "swarm" in agent.name
        assert len(agent.capabilities) > 0

    def test_constructor_with_algorithm(self, agent_factory):
        """Test initialization with algorithm parameter."""
        agent = agent_factory(
            SwarmIntelligenceAgent,
            algorithm=SwarmAlgorithm.PSO,
        )
        assert agent.algorithm == SwarmAlgorithm.PSO

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(SwarmIntelligenceAgent)
        await agent.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic swarm intelligence work."""
        agent = agent_factory(SwarmIntelligenceAgent)
        await agent.initialize()
        result = await agent.work()
        assert result is not None

    def test_algorithm_enum_values(self):
        """Test that expected algorithms exist."""
        expected_algorithms = ["PSO", "ACO", "HYBRID"]
        for algo_name in expected_algorithms:
            assert hasattr(SwarmAlgorithm, algo_name)


class TestDAOAgent:
    """Tests for DAOAgent (Decentralized Autonomous Organization)."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(DAOAgent)
        # Name may include member ID
        assert "dao" in agent.name
        assert len(agent.capabilities) > 0

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(DAOAgent)
        await agent.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic DAO governance work."""
        agent = agent_factory(DAOAgent)
        await agent.initialize()
        result = await agent.work()
        assert result is not None

    def test_proposal_type_enum_values(self):
        """Test that expected proposal types exist."""
        expected_types = ["GOVERNANCE", "TREASURY", "MEMBERSHIP", "EXECUTION", "CONSTITUTIONAL"]
        for type_name in expected_types:
            assert hasattr(ProposalType, type_name)

    def test_proposal_status_enum_values(self):
        """Test that expected proposal statuses exist."""
        expected_statuses = ["DRAFT", "ACTIVE", "PASSED", "REJECTED", "EXECUTED"]
        for status_name in expected_statuses:
            assert hasattr(ProposalStatus, status_name)

    def test_vote_choice_enum_values(self):
        """Test that expected vote choices exist."""
        expected_choices = ["FOR", "AGAINST", "ABSTAIN"]
        for choice_name in expected_choices:
            assert hasattr(VoteChoice, choice_name)
