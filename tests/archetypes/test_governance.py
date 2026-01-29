"""Tests for governance agent archetypes: Jury, Executive, Legislative, Judicial, Bureaucracy."""

from __future__ import annotations

import pytest

from agents.archetypes.jury import JuryAgent
from agents.archetypes.government import ExecutiveAgent, LegislativeAgent, JudicialAgent
from agents.archetypes.bureaucracy import BureaucracyAgent
from agents.framework.base_agent import AgentState


class TestJuryAgent:
    """Tests for JuryAgent."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(JuryAgent)
        assert agent.name == "jury"
        assert len(agent.capabilities) > 0

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(JuryAgent)
        await agent.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic jury deliberation work."""
        agent = agent_factory(JuryAgent)
        await agent.initialize()
        result = await agent.work()
        assert result is not None


class TestExecutiveAgent:
    """Tests for ExecutiveAgent."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(ExecutiveAgent)
        assert agent.name == "executive"
        assert len(agent.capabilities) > 0

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(ExecutiveAgent)
        await agent.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic executive work."""
        agent = agent_factory(ExecutiveAgent)
        await agent.initialize()
        result = await agent.work()
        assert result is not None


class TestLegislativeAgent:
    """Tests for LegislativeAgent."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(LegislativeAgent)
        assert agent.name == "legislative"
        assert len(agent.capabilities) > 0

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(LegislativeAgent)
        await agent.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic legislative work."""
        agent = agent_factory(LegislativeAgent)
        await agent.initialize()
        result = await agent.work()
        assert result is not None


class TestJudicialAgent:
    """Tests for JudicialAgent."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(JudicialAgent)
        assert agent.name == "judicial"
        assert len(agent.capabilities) > 0

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(JudicialAgent)
        await agent.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic judicial work."""
        agent = agent_factory(JudicialAgent)
        await agent.initialize()
        result = await agent.work()
        assert result is not None


class TestBureaucracyAgent:
    """Tests for BureaucracyAgent."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(BureaucracyAgent)
        # Name includes caste type, so just check it starts with bureaucracy
        assert "bureaucracy" in agent.name
        assert len(agent.capabilities) > 0

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(BureaucracyAgent)
        await agent.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic bureaucratic work."""
        agent = agent_factory(BureaucracyAgent)
        await agent.initialize()
        result = await agent.work()
        assert result is not None
