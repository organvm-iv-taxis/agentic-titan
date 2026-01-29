"""Tests for professional agent archetypes: CFO, DevOps, Security, DataEngineer, ProductManager."""

from __future__ import annotations

import pytest

from agents.archetypes.cfo import CFOAgent
from agents.archetypes.devops import DevOpsAgent
from agents.archetypes.security_analyst import SecurityAnalystAgent
from agents.archetypes.data_engineer import DataEngineerAgent
from agents.archetypes.product_manager import ProductManagerAgent
from agents.framework.base_agent import AgentState


class TestCFOAgent:
    """Tests for CFOAgent."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(CFOAgent)
        assert agent.name == "cfo"
        # Check that it has budget-related capabilities
        assert any(c in ["budget", "cost_optimization", "reporting"] for c in agent.capabilities)

    def test_constructor_with_session_budget(self, agent_factory):
        """Test initialization with session budget."""
        agent = agent_factory(
            CFOAgent,
            session_budget_usd=500.0,
        )
        assert agent._session_budget_usd == 500.0

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(CFOAgent)
        await agent.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic CFO work."""
        agent = agent_factory(CFOAgent)
        await agent.initialize()
        result = await agent.work()
        assert result is not None


class TestDevOpsAgent:
    """Tests for DevOpsAgent."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(DevOpsAgent)
        assert agent.name == "devops"
        # Check that it has infrastructure-related capabilities
        assert len(agent.capabilities) > 0

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(DevOpsAgent)
        await agent.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic DevOps work."""
        agent = agent_factory(DevOpsAgent)
        await agent.initialize()
        result = await agent.work()
        assert result is not None


class TestSecurityAnalystAgent:
    """Tests for SecurityAnalystAgent."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(SecurityAnalystAgent)
        assert agent.name == "security_analyst"
        # Check that it has security-related capabilities
        assert len(agent.capabilities) > 0

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(SecurityAnalystAgent)
        await agent.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic security analysis work."""
        agent = agent_factory(SecurityAnalystAgent)
        await agent.initialize()
        result = await agent.work()
        assert result is not None


class TestDataEngineerAgent:
    """Tests for DataEngineerAgent."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(DataEngineerAgent)
        assert agent.name == "data_engineer"
        # Check that it has data-related capabilities
        assert len(agent.capabilities) > 0

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(DataEngineerAgent)
        await agent.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic data engineering work."""
        agent = agent_factory(DataEngineerAgent)
        await agent.initialize()
        result = await agent.work()
        assert result is not None


class TestProductManagerAgent:
    """Tests for ProductManagerAgent."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(ProductManagerAgent)
        assert agent.name == "product_manager"
        # Check that it has product-related capabilities
        assert len(agent.capabilities) > 0

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(ProductManagerAgent)
        await agent.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic product management work."""
        agent = agent_factory(ProductManagerAgent)
        await agent.initialize()
        result = await agent.work()
        assert result is not None
