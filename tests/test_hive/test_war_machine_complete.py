"""Tests for complete war machine operations (Phase 16B)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hive.machines import (
    MachineDynamics,
    MachineOperation,
    MachineState,
    MachineType,
    OperationType,
)


class TestWarMachineOperations:
    """Tests for war machine operations in MachineDynamics."""

    @pytest.fixture
    def mock_topology_with_territories(self):
        """Create mock topology with territories."""
        topology = MagicMock()

        # Set up territories
        territory1 = MagicMock()
        territory1.agent_ids = ["agent_1", "agent_2"]
        territory1.stability = 1.0

        territory2 = MagicMock()
        territory2.agent_ids = ["agent_3", "agent_4"]
        territory2.stability = 0.8

        topology._territories = {
            "territory_1": territory1,
            "territory_2": territory2,
        }

        # Set up nodes
        node1 = MagicMock()
        node1.agent_id = "agent_1"
        node1.role = "worker"
        node1.metadata = {}

        node2 = MagicMock()
        node2.agent_id = "agent_2"
        node2.role = "worker"
        node2.metadata = {}

        topology.nodes = {
            "agent_1": node1,
            "agent_2": node2,
            "agent_3": MagicMock(agent_id="agent_3", metadata={}),
            "agent_4": MagicMock(agent_id="agent_4", metadata={}),
        }

        # Mock methods
        topology.rupture = MagicMock(return_value=["agent_1"])
        topology.dissolve_territory = MagicMock(return_value=["agent_1", "agent_2"])
        topology.flux_cycle = AsyncMock(return_value={})

        return topology

    @pytest.fixture
    def mock_deterritorialized_topology(self):
        """Create mock deterritorialized topology."""
        topology = MagicMock()

        # Set up nodes with metadata
        nodes = {}
        for i in range(5):
            node = MagicMock()
            node.agent_id = f"agent_{i}"
            node.role = "worker"
            node.metadata = {"fixed_position": True, "locked": True}
            nodes[f"agent_{i}"] = node

        topology.nodes = nodes
        topology.flux_cycle = AsyncMock(return_value={"agent_0": "explorer"})

        return topology

    @pytest.fixture
    def dynamics(self, mock_topology_with_territories):
        """Create MachineDynamics with registered topology."""
        dynamics = MachineDynamics(initial_balance=0.0)
        dynamics.register_topology("territory_topo", mock_topology_with_territories)
        return dynamics

    @pytest.mark.asyncio
    async def test_smooth_operation_all_territories(self, dynamics):
        """Test smooth operation affects all territories."""
        operation = await dynamics.smooth_operation(territory_id=None)

        assert operation.operation_type == OperationType.SMOOTH
        assert operation.machine_type == MachineType.WAR
        assert operation.success is True
        assert dynamics._state.striation_level < 0.5  # Reduced

    @pytest.mark.asyncio
    async def test_smooth_operation_specific_territory(self, dynamics, mock_topology_with_territories):
        """Test smooth operation on specific territory."""
        operation = await dynamics.smooth_operation(territory_id="territory_1")

        assert operation.success is True
        # Territory stability should be reduced
        assert mock_topology_with_territories._territories["territory_1"].stability < 1.0

    @pytest.mark.asyncio
    async def test_smooth_operation_updates_state(self, dynamics):
        """Test smooth operation updates machine state."""
        initial_striation = dynamics._state.striation_level
        initial_war = dynamics._state.war_intensity

        await dynamics.smooth_operation()

        assert dynamics._state.striation_level < initial_striation
        assert dynamics._state.war_intensity > initial_war

    @pytest.mark.asyncio
    async def test_line_of_flight(self, dynamics, mock_topology_with_territories):
        """Test line of flight operation."""
        # Set up topology with the method
        mock_topology_with_territories.initiate_line_of_flight = MagicMock(return_value=True)
        mock_topology_with_territories._agent_territory = {
            "agent_1": "territory_1",
        }

        operation = await dynamics.line_of_flight(
            agent_id="agent_1",
            escape_vector="exploration",
            duration=300.0,
        )

        assert operation.operation_type == OperationType.LINE_OF_FLIGHT
        assert operation.target_agents == ["agent_1"]
        assert "escape_vector" in operation.metadata

    @pytest.mark.asyncio
    async def test_line_of_flight_updates_escape_rate(self, dynamics, mock_topology_with_territories):
        """Test line of flight updates escape rate."""
        mock_topology_with_territories.initiate_line_of_flight = MagicMock(return_value=True)

        initial_escape = dynamics._state.escape_rate

        await dynamics.line_of_flight(
            agent_id="agent_1",
            escape_vector="innovation",
        )

        assert dynamics._state.escape_rate >= initial_escape

    @pytest.mark.asyncio
    async def test_nomadize_operation(self, dynamics, mock_deterritorialized_topology):
        """Test nomadize operation."""
        dynamics.register_topology("deter_topo", mock_deterritorialized_topology)

        operation = await dynamics.nomadize_operation(target_agents=None)

        assert operation.operation_type == OperationType.NOMADIZE
        assert operation.machine_type == MachineType.WAR
        # Should have affected some agents
        assert len(operation.target_agents) > 0

    @pytest.mark.asyncio
    async def test_nomadize_removes_fixed_markers(self, dynamics, mock_deterritorialized_topology):
        """Test nomadize removes fixed position markers."""
        dynamics.register_topology("deter_topo", mock_deterritorialized_topology)

        await dynamics.nomadize_operation()

        # Check that fixed_position was removed from metadata
        for node in mock_deterritorialized_topology.nodes.values():
            # The pop should have been called
            assert "fixed_position" not in node.metadata or node.metadata.get("fixed_position") is None

    @pytest.mark.asyncio
    async def test_nomadize_specific_agents(self, dynamics, mock_deterritorialized_topology):
        """Test nomadize with specific target agents."""
        dynamics.register_topology("deter_topo", mock_deterritorialized_topology)

        operation = await dynamics.nomadize_operation(target_agents=["agent_0", "agent_1"])

        assert "agent_0" in operation.target_agents or "agent_1" in operation.target_agents

    @pytest.mark.asyncio
    async def test_deterritorialize_operation(self, dynamics):
        """Test deterritorialize operation."""
        operation = await dynamics.deterritorialize_operation(territory_id="territory_1")

        assert operation.operation_type == OperationType.DETERRITORIALIZE
        assert operation.machine_type == MachineType.WAR
        assert "territory" in operation.metadata

    @pytest.mark.asyncio
    async def test_deterritorialize_updates_state(self, dynamics):
        """Test deterritorialize updates machine state."""
        initial_striation = dynamics._state.striation_level
        initial_state_intensity = dynamics._state.state_intensity

        await dynamics.deterritorialize_operation(territory_id="territory_1")

        assert dynamics._state.striation_level < initial_striation
        assert dynamics._state.state_intensity < initial_state_intensity

    @pytest.mark.asyncio
    async def test_operation_records_to_history(self, dynamics):
        """Test operations are recorded to history."""
        initial_count = len(dynamics._operations)

        await dynamics.smooth_operation()
        await dynamics.nomadize_operation()

        assert len(dynamics._operations) == initial_count + 2

    def test_get_operation_stats(self, dynamics):
        """Test getting operation statistics."""
        stats = dynamics.get_operation_stats()

        assert "total_operations" in stats
        assert "state_operations" in stats
        assert "war_operations" in stats
        assert "current_balance" in stats
        assert "dominant_machine" in stats

    @pytest.mark.asyncio
    async def test_war_operations_affect_balance(self, dynamics):
        """Test war operations shift balance toward war machine."""
        initial_balance = dynamics._state.balance

        await dynamics.smooth_operation()
        await dynamics.nomadize_operation()

        # War operations should decrease state intensity or increase war intensity
        # resulting in lower balance (toward war machine)
        assert dynamics._state.war_intensity >= 0.5


class TestMachineStateBalance:
    """Tests for machine state and balance."""

    def test_initial_balance(self):
        """Test initial balanced state."""
        dynamics = MachineDynamics(initial_balance=0.0)
        assert dynamics.state.balance == 0.0
        assert dynamics.dominant_machine == MachineType.HYBRID

    def test_state_dominant_threshold(self):
        """Test state machine dominance threshold."""
        dynamics = MachineDynamics(initial_balance=0.8)
        assert dynamics.dominant_machine == MachineType.STATE

    def test_war_dominant_threshold(self):
        """Test war machine dominance threshold."""
        dynamics = MachineDynamics(initial_balance=-0.8)
        assert dynamics.dominant_machine == MachineType.WAR

    def test_adjust_balance_positive(self):
        """Test adjusting balance toward state machine."""
        dynamics = MachineDynamics(initial_balance=0.0)
        new_balance = dynamics.adjust_balance(0.3)

        assert new_balance > 0.0
        assert dynamics._state.state_intensity > 0.5

    def test_adjust_balance_negative(self):
        """Test adjusting balance toward war machine."""
        dynamics = MachineDynamics(initial_balance=0.0)
        new_balance = dynamics.adjust_balance(-0.3)

        assert new_balance < 0.0
        assert dynamics._state.war_intensity > 0.5


class TestMachineOperation:
    """Tests for MachineOperation dataclass."""

    def test_operation_creation(self):
        """Test creating an operation."""
        operation = MachineOperation(
            operation_type=OperationType.SMOOTH,
            machine_type=MachineType.WAR,
            target_agents=["agent_1", "agent_2"],
            success=True,
        )

        assert operation.operation_type == OperationType.SMOOTH
        assert operation.machine_type == MachineType.WAR
        assert len(operation.target_agents) == 2

    def test_operation_to_dict(self):
        """Test operation serialization."""
        operation = MachineOperation(
            operation_type=OperationType.NOMADIZE,
            machine_type=MachineType.WAR,
            target_agents=["agent_1"],
            success=True,
            metadata={"reason": "exploration"},
        )

        data = operation.to_dict()
        assert data["operation_type"] == "nomadize"
        assert data["machine_type"] == "war"
        assert data["metadata"]["reason"] == "exploration"
