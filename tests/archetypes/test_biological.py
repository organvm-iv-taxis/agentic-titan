"""Tests for biological agent archetypes: EusocialColony, Cell."""

from __future__ import annotations

import pytest

from agents.archetypes.eusocial import EusocialColonyAgent, CasteType
from agents.archetypes.cell import CellAgent, SignalType, CellType, CellState
from agents.framework.base_agent import AgentState


class TestEusocialColonyAgent:
    """Tests for EusocialColonyAgent."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(EusocialColonyAgent)
        # Name includes caste type (e.g., colony_worker)
        assert "colony" in agent.name or "eusocial" in agent.name
        assert len(agent.capabilities) > 0

    def test_constructor_with_caste(self, agent_factory):
        """Test initialization with caste parameter."""
        agent = agent_factory(
            EusocialColonyAgent,
            caste=CasteType.QUEEN,
        )
        assert agent.caste == CasteType.QUEEN

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(EusocialColonyAgent)
        await agent.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic colony work."""
        agent = agent_factory(EusocialColonyAgent)
        await agent.initialize()
        result = await agent.work()
        assert result is not None

    def test_caste_enum_values(self):
        """Test that all expected castes exist."""
        expected_castes = ["QUEEN", "WORKER", "SOLDIER", "NURSE", "FORAGER"]
        for caste_name in expected_castes:
            assert hasattr(CasteType, caste_name)

    @pytest.mark.asyncio
    async def test_different_castes_can_be_created(self, agent_factory, mock_llm_router):
        """Test that different castes can be created."""
        queen = agent_factory(EusocialColonyAgent, caste=CasteType.QUEEN)
        worker = agent_factory(EusocialColonyAgent, caste=CasteType.WORKER)

        assert queen.caste != worker.caste


class TestCellAgent:
    """Tests for CellAgent."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(CellAgent)
        # Name includes cell type (e.g., cell_somatic)
        assert "cell" in agent.name
        assert len(agent.capabilities) > 0

    def test_constructor_with_cell_type(self, agent_factory):
        """Test initialization with cell type parameter."""
        agent = agent_factory(
            CellAgent,
            cell_type=CellType.STEM,
        )
        assert agent.cell_type == CellType.STEM

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(CellAgent)
        await agent.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic cell work."""
        agent = agent_factory(CellAgent)
        await agent.initialize()
        result = await agent.work()
        assert result is not None

    def test_signal_type_enum_values(self):
        """Test that all expected signal types exist."""
        expected_signals = ["GROWTH", "APOPTOSIS", "DIFFERENTIATION", "SURVIVAL"]
        for signal_name in expected_signals:
            assert hasattr(SignalType, signal_name)

    def test_cell_type_enum_values(self):
        """Test that expected cell types exist."""
        expected_types = ["STEM", "SOMATIC", "GERMLINE", "SIGNALING"]
        for type_name in expected_types:
            assert hasattr(CellType, type_name)

    def test_cell_state_enum_values(self):
        """Test that expected cell states exist."""
        expected_states = ["HEALTHY", "STRESSED", "DAMAGED", "APOPTOTIC", "DEAD"]
        for state_name in expected_states:
            assert hasattr(CellState, state_name)
