"""Tests for complete assembly transitions (Phase 16B)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hive.assembly import (
    AssemblyEvent,
    AssemblyManager,
    AssemblyState,
    DeterritorializationType,
    StabilityMetrics,
    TerritorizationType,
)


class TestAssemblyState:
    """Tests for AssemblyState enum."""

    def test_states_exist(self):
        """Test all states are defined."""
        assert AssemblyState.STABLE == "stable"
        assert AssemblyState.UNSTABLE == "unstable"
        assert AssemblyState.TRANSITIONING == "transitioning"
        assert AssemblyState.RUPTURED == "ruptured"
        assert AssemblyState.CRYSTALLIZED == "crystallized"


class TestTerritorizationType:
    """Tests for TerritorizationType enum."""

    def test_types_exist(self):
        """Test all types are defined."""
        assert TerritorizationType.CODING == "coding"
        assert TerritorizationType.STRATIFICATION == "stratification"
        assert TerritorizationType.SEGMENTATION == "segmentation"
        assert TerritorizationType.CAPTURE == "capture"


class TestDeterritorializationType:
    """Tests for DeterritorializationType enum."""

    def test_types_exist(self):
        """Test all types are defined."""
        assert DeterritorializationType.DECODING == "decoding"
        assert DeterritorializationType.SMOOTHING == "smoothing"
        assert DeterritorializationType.NOMADIZATION == "nomadization"
        assert DeterritorializationType.FLIGHT == "flight"


class TestStabilityMetrics:
    """Tests for StabilityMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = StabilityMetrics()
        assert metrics.connection_density == 0.0
        assert metrics.role_consistency == 0.0
        assert metrics.task_completion_rate == 0.0

    def test_overall_stability(self):
        """Test overall stability calculation."""
        high_stability = StabilityMetrics(
            connection_density=0.8,
            role_consistency=0.9,
            communication_flow=0.8,
            task_completion_rate=0.9,
            defection_rate=0.1,
            coordination_overhead=0.2,
        )
        assert high_stability.overall_stability > 0.6

        low_stability = StabilityMetrics(
            connection_density=0.2,
            role_consistency=0.3,
            communication_flow=0.2,
            task_completion_rate=0.3,
            defection_rate=0.5,
            coordination_overhead=0.8,
        )
        assert low_stability.overall_stability < high_stability.overall_stability

    def test_suggested_state_stable(self):
        """Test suggesting stable state."""
        metrics = StabilityMetrics(
            connection_density=0.85,
            role_consistency=0.9,
            communication_flow=0.85,
            task_completion_rate=0.9,
            defection_rate=0.02,
            coordination_overhead=0.05,
        )
        # Overall stability should be in the stable range (0.7-0.9)
        stability = metrics.overall_stability
        assert 0.7 <= stability < 0.9, f"Expected stability in [0.7, 0.9), got {stability}"
        assert metrics.suggested_state == AssemblyState.STABLE

    def test_suggested_state_crystallized(self):
        """Test suggesting crystallized state."""
        metrics = StabilityMetrics(
            connection_density=1.0,
            role_consistency=1.0,
            communication_flow=1.0,
            task_completion_rate=1.0,
            defection_rate=0.0,
            coordination_overhead=0.0,
        )
        # Overall stability should be > 0.9 for crystallized
        stability = metrics.overall_stability
        assert stability > 0.9, f"Expected stability > 0.9, got {stability}"
        assert metrics.suggested_state == AssemblyState.CRYSTALLIZED

    def test_suggested_state_ruptured(self):
        """Test suggesting ruptured state."""
        metrics = StabilityMetrics(
            connection_density=0.1,
            role_consistency=0.1,
            communication_flow=0.1,
            task_completion_rate=0.1,
            defection_rate=0.8,
            coordination_overhead=0.9,
        )
        assert metrics.suggested_state == AssemblyState.RUPTURED


class TestAssemblyEvent:
    """Tests for AssemblyEvent dataclass."""

    def test_event_creation(self):
        """Test creating an event."""
        event = AssemblyEvent(
            event_type="territorialize_coding",
            previous_state=AssemblyState.UNSTABLE,
            new_state=AssemblyState.STABLE,
            trigger="stability_evaluation",
            affected_agents=["agent_1", "agent_2"],
        )

        assert event.event_type == "territorialize_coding"
        assert len(event.affected_agents) == 2

    def test_event_to_dict(self):
        """Test event serialization."""
        event = AssemblyEvent(
            event_type="deterritorialize_decoding",
            previous_state=AssemblyState.CRYSTALLIZED,
            new_state=AssemblyState.UNSTABLE,
            trigger="manual",
        )

        data = event.to_dict()
        assert data["event_type"] == "deterritorialize_decoding"
        assert data["previous_state"] == "crystallized"


class TestAssemblyManager:
    """Tests for AssemblyManager class."""

    @pytest.fixture
    def mock_topology_with_nodes(self):
        """Create mock topology with nodes."""
        topology = MagicMock()

        # Set up nodes
        nodes = {}
        for i in range(10):
            node = MagicMock()
            node.agent_id = f"agent_{i}"
            node.role = "worker" if i > 0 else None
            node.neighbors = []
            node.child_ids = []
            nodes[f"agent_{i}"] = node

        topology.nodes = nodes
        return topology

    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        bus = MagicMock()
        bus.emit = AsyncMock()
        return bus

    @pytest.fixture
    def manager(self, mock_topology_with_nodes, mock_event_bus):
        """Create an AssemblyManager for testing."""
        mgr = AssemblyManager(
            assembly_id="test_assembly",
            evaluation_interval=1.0,
            event_bus=mock_event_bus,
        )
        mgr.register_topology("main", mock_topology_with_nodes)
        return mgr

    def test_initial_state(self, manager):
        """Test initial manager state."""
        assert manager.state == AssemblyState.STABLE
        assert manager.assembly_id == "test_assembly"

    @pytest.mark.asyncio
    async def test_evaluate_stability(self, manager):
        """Test evaluating stability metrics."""
        metrics = await manager.evaluate_stability()

        assert isinstance(metrics, StabilityMetrics)
        assert 0.0 <= metrics.connection_density <= 1.0
        assert 0.0 <= metrics.overall_stability <= 1.0

    def test_should_territorialize_when_unstable(self, manager):
        """Test territorialize recommendation when unstable."""
        manager._state = AssemblyState.UNSTABLE
        manager._metrics = StabilityMetrics(
            connection_density=0.2,
            role_consistency=0.2,
            task_completion_rate=0.2,
            defection_rate=0.3,
        )

        assert manager.should_territorialize() is True

    def test_should_not_territorialize_when_stable(self, manager):
        """Test no territorialize when already stable."""
        manager._state = AssemblyState.STABLE
        manager._metrics = StabilityMetrics(
            connection_density=0.8,
            role_consistency=0.8,
            task_completion_rate=0.8,
            defection_rate=0.05,
        )

        assert manager.should_territorialize() is False

    def test_should_deterritorialize_when_crystallized(self, manager):
        """Test deterritorialize recommendation when crystallized."""
        manager._state = AssemblyState.CRYSTALLIZED
        manager._metrics = StabilityMetrics(
            connection_density=0.95,
            coordination_overhead=0.8,
        )

        assert manager.should_deterritorialize() is True

    def test_should_not_deterritorialize_when_unstable(self, manager):
        """Test no deterritorialize when already unstable."""
        manager._state = AssemblyState.UNSTABLE
        manager._metrics = StabilityMetrics(
            connection_density=0.3,
        )

        assert manager.should_deterritorialize() is False

    @pytest.mark.asyncio
    async def test_territorialize_coding(self, manager, mock_topology_with_nodes):
        """Test coding territorialization."""
        result = await manager.territorialize(TerritorizationType.CODING)

        assert result is True
        # Check that nodes without roles got assigned
        assert len(manager._history) > 0

    @pytest.mark.asyncio
    async def test_territorialize_capture(self, manager, mock_topology_with_nodes):
        """Test capture territorialization."""
        mock_topology_with_nodes.get_escaped_agents = MagicMock(return_value=["agent_1"])
        mock_topology_with_nodes.capture = MagicMock(return_value=True)

        result = await manager.territorialize(TerritorizationType.CAPTURE)

        assert result is True

    @pytest.mark.asyncio
    async def test_deterritorialize_decoding(self, manager, mock_topology_with_nodes):
        """Test decoding deterritorialization."""
        result = await manager.deterritorialize(DeterritorializationType.DECODING)

        assert result is True
        assert len(manager._history) > 0

    @pytest.mark.asyncio
    async def test_deterritorialize_nomadization(self, manager, mock_topology_with_nodes):
        """Test nomadization deterritorialization."""
        mock_topology_with_nodes.flux_cycle = AsyncMock()

        result = await manager.deterritorialize(DeterritorializationType.NOMADIZATION)

        assert result is True
        mock_topology_with_nodes.flux_cycle.assert_called_once()

    def test_record_role_change(self, manager):
        """Test recording role changes."""
        manager.record_role_change("agent_1")
        assert len(manager._role_changes) == 1

    def test_record_defection(self, manager):
        """Test recording defections."""
        manager.record_defection("agent_1")
        assert len(manager._defections) == 1

    def test_record_task_result(self, manager):
        """Test recording task results."""
        manager.record_task_result(True)
        manager.record_task_result(False)
        assert len(manager._task_results) == 2

    def test_get_history(self, manager):
        """Test getting event history."""
        assert manager.get_history() == []

    @pytest.mark.asyncio
    async def test_get_history_after_transitions(self, manager):
        """Test history after transitions."""
        await manager.territorialize(TerritorizationType.CODING)

        history = manager.get_history()
        assert len(history) >= 1

    def test_register_topology(self, manager):
        """Test registering a topology."""
        new_topo = MagicMock()
        manager.register_topology("new_topo", new_topo)
        assert "new_topo" in manager._topologies

    def test_unregister_topology(self, manager):
        """Test unregistering a topology."""
        manager.unregister_topology("main")
        assert "main" not in manager._topologies

    def test_to_dict(self, manager):
        """Test serialization."""
        data = manager.to_dict()

        assert "assembly_id" in data
        assert "state" in data
        assert "metrics" in data
        assert "history" in data

    @pytest.mark.asyncio
    async def test_start_stop(self, manager):
        """Test starting and stopping the manager."""
        await manager.start()
        assert manager._running is True

        await manager.stop()
        assert manager._running is False


class TestAssemblyIntegration:
    """Integration tests for assembly dynamics."""

    @pytest.fixture
    def full_manager(self):
        """Create a fully configured manager."""
        manager = AssemblyManager(
            assembly_id="integration_test",
            evaluation_interval=0.1,
        )

        # Create mock topology
        topology = MagicMock()
        nodes = {}
        for i in range(20):
            node = MagicMock()
            node.agent_id = f"agent_{i}"
            node.role = "worker"
            node.neighbors = [f"agent_{(i+1) % 20}"]
            node.child_ids = []
            nodes[f"agent_{i}"] = node

        topology.nodes = nodes
        manager.register_topology("main", topology)

        return manager

    @pytest.mark.asyncio
    async def test_stability_evaluation_loop(self, full_manager):
        """Test the evaluation loop runs correctly."""
        await full_manager.start()
        await asyncio.sleep(0.25)
        await full_manager.stop()

        # Should have evaluated at least once
        assert full_manager.metrics is not None

    @pytest.mark.asyncio
    async def test_complete_territorialize_deterritorialize_cycle(self, full_manager):
        """Test a complete cycle of territorialization and deterritorialization."""
        # Start unstable
        full_manager._state = AssemblyState.UNSTABLE
        full_manager._metrics = StabilityMetrics(
            defection_rate=0.3,
            task_completion_rate=0.2,
        )

        # Territorialize
        result = await full_manager.territorialize(TerritorizationType.CODING)
        assert result is True

        # Now make it crystallized
        full_manager._state = AssemblyState.CRYSTALLIZED
        full_manager._metrics = StabilityMetrics(
            connection_density=0.95,
            coordination_overhead=0.8,
        )

        # Deterritorialize
        result = await full_manager.deterritorialize(DeterritorializationType.DECODING)
        assert result is True

        # Should have two events in history
        history = full_manager.get_history()
        assert len(history) == 2
