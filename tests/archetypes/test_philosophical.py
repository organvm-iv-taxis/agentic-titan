"""Tests for philosophical agent archetypes: Assemblage, ActorNetwork."""

from __future__ import annotations

import pytest

from agents.archetypes.assemblage import AssemblageAgent, TerritorialState, ComponentType
from agents.archetypes.actor_network import ActorNetworkAgent, TranslationType
from agents.framework.base_agent import AgentState


class TestAssemblageAgent:
    """Tests for AssemblageAgent (Deleuze & Guattari inspired)."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(AssemblageAgent)
        assert agent.name == "assemblage"
        assert len(agent.capabilities) > 0

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(AssemblageAgent)
        await agent.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic assemblage work."""
        agent = agent_factory(AssemblageAgent)
        await agent.initialize()
        result = await agent.work()
        assert result is not None

    def test_territorial_state_enum_values(self):
        """Test that expected territorial states exist."""
        expected_states = ["DETERRITORIALIZED", "RETERRITORIALIZING", "TERRITORIALIZED", "STRATIFIED"]
        for state_name in expected_states:
            assert hasattr(TerritorialState, state_name)

    def test_component_type_enum_values(self):
        """Test that expected component types exist."""
        expected_types = ["MATERIAL", "EXPRESSIVE", "MACHINIC", "ENUNCIATIVE"]
        for type_name in expected_types:
            assert hasattr(ComponentType, type_name)


class TestActorNetworkAgent:
    """Tests for ActorNetworkAgent (Actor-Network Theory inspired)."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(ActorNetworkAgent)
        assert agent.name == "actor_network"
        assert len(agent.capabilities) > 0

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(ActorNetworkAgent)
        await agent.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic actor-network work."""
        agent = agent_factory(ActorNetworkAgent)
        await agent.initialize()
        result = await agent.work()
        assert result is not None

    def test_translation_type_enum_values(self):
        """Test that expected translation types exist."""
        expected_types = ["PROBLEMATIZATION", "INTERESSEMENT", "ENROLLMENT", "MOBILIZATION"]
        for type_name in expected_types:
            assert hasattr(TranslationType, type_name)
