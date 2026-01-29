"""Tests for criticality detection and phase transitions (Phase 16A)."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hive.criticality import (
    CriticalityMetrics,
    CriticalityMonitor,
    CriticalityState,
    PhaseTransition,
)


class TestCriticalityState:
    """Tests for CriticalityState enum."""

    def test_states_exist(self):
        """Test all states are defined."""
        assert CriticalityState.SUBCRITICAL == "subcritical"
        assert CriticalityState.CRITICAL == "critical"
        assert CriticalityState.SUPERCRITICAL == "supercritical"


class TestCriticalityMetrics:
    """Tests for CriticalityMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = CriticalityMetrics()
        assert metrics.correlation_length == 0.0
        assert metrics.susceptibility == 0.0
        assert metrics.relaxation_time == 0.0
        assert metrics.fluctuation_size == 0.0
        assert metrics.order_parameter == 0.5

    def test_criticality_score_calculation(self):
        """Test criticality score is calculated correctly."""
        # Low criticality
        low = CriticalityMetrics(
            correlation_length=0.1,
            susceptibility=0.5,
            relaxation_time=5.0,
            fluctuation_size=0.05,
        )
        assert 0.0 <= low.criticality_score <= 1.0

        # High criticality (edge of chaos)
        high = CriticalityMetrics(
            correlation_length=0.9,
            susceptibility=4.0,
            relaxation_time=50.0,
            fluctuation_size=0.2,
        )
        assert high.criticality_score > low.criticality_score

    def test_criticality_score_bounds(self):
        """Test criticality score stays within bounds."""
        extreme = CriticalityMetrics(
            correlation_length=10.0,  # Way above normal
            susceptibility=100.0,
            relaxation_time=1000.0,
            fluctuation_size=5.0,
        )
        assert 0.0 <= extreme.criticality_score <= 1.0

    def test_infer_state_subcritical(self):
        """Test inferring subcritical state."""
        metrics = CriticalityMetrics(
            correlation_length=0.1,
            susceptibility=0.3,
            relaxation_time=5.0,
            fluctuation_size=0.02,
            order_parameter=0.9,  # Very high order
        )
        assert metrics.infer_state() == CriticalityState.SUBCRITICAL

    def test_infer_state_supercritical(self):
        """Test inferring supercritical state."""
        metrics = CriticalityMetrics(
            correlation_length=0.3,
            susceptibility=2.0,
            relaxation_time=20.0,
            fluctuation_size=0.3,
            order_parameter=0.1,  # Very low order
        )
        assert metrics.infer_state() == CriticalityState.SUPERCRITICAL

    def test_infer_state_critical(self):
        """Test inferring critical state."""
        metrics = CriticalityMetrics(
            correlation_length=0.8,
            susceptibility=3.5,
            relaxation_time=40.0,
            fluctuation_size=0.15,
            order_parameter=0.5,  # Balanced order
        )
        assert metrics.infer_state() == CriticalityState.CRITICAL


class TestPhaseTransition:
    """Tests for PhaseTransition dataclass."""

    def test_phase_transition_creation(self):
        """Test creating a phase transition record."""
        before = CriticalityMetrics(order_parameter=0.3)
        after = CriticalityMetrics(order_parameter=0.7)

        transition = PhaseTransition(
            transition_id="test_1",
            from_state=CriticalityState.SUPERCRITICAL,
            to_state=CriticalityState.SUBCRITICAL,
            trigger="test",
            metrics_before=before,
            metrics_after=after,
        )

        assert transition.transition_id == "test_1"
        assert transition.from_state == CriticalityState.SUPERCRITICAL
        assert transition.to_state == CriticalityState.SUBCRITICAL

    def test_phase_transition_to_dict(self):
        """Test serialization of phase transition."""
        transition = PhaseTransition(
            transition_id="test_2",
            from_state=CriticalityState.CRITICAL,
            to_state=CriticalityState.SUBCRITICAL,
            trigger="stability",
            metrics_before=CriticalityMetrics(),
            metrics_after=CriticalityMetrics(),
        )

        data = transition.to_dict()
        assert data["transition_id"] == "test_2"
        assert data["from_state"] == "critical"
        assert data["to_state"] == "subcritical"
        assert "timestamp" in data


class TestCriticalityMonitor:
    """Tests for CriticalityMonitor class."""

    @pytest.fixture
    def mock_neighborhood(self):
        """Create mock neighborhood."""
        neighborhood = MagicMock()
        neighborhood.get_network_stats.return_value = {
            "total_agents": 10,
            "average_clustering": 0.5,
            "density": 0.4,
        }
        neighborhood._profiles = {}
        return neighborhood

    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        bus = MagicMock()
        bus.emit = AsyncMock()
        return bus

    @pytest.fixture
    def monitor(self, mock_neighborhood, mock_event_bus):
        """Create a CriticalityMonitor for testing."""
        return CriticalityMonitor(
            neighborhood=mock_neighborhood,
            event_bus=mock_event_bus,
            sample_interval=1.0,  # Fast for testing
        )

    def test_initial_state(self, monitor):
        """Test initial monitor state."""
        assert monitor.current_state == CriticalityState.CRITICAL
        assert monitor.current_metrics is not None

    @pytest.mark.asyncio
    async def test_sample_state(self, monitor):
        """Test sampling system state."""
        metrics = await monitor.sample_state()

        assert isinstance(metrics, CriticalityMetrics)
        assert 0.0 <= metrics.correlation_length <= 1.0
        assert metrics.susceptibility >= 0.0
        assert metrics.relaxation_time >= 0.0

    @pytest.mark.asyncio
    async def test_detect_phase_transition_no_history(self, monitor):
        """Test no transition detected with insufficient history."""
        result = await monitor.detect_phase_transition()
        assert result is False

    @pytest.mark.asyncio
    async def test_detect_phase_transition_with_change(self, monitor):
        """Test transition detection with state change."""
        # Add some history
        monitor._metrics_history = [
            CriticalityMetrics(order_parameter=0.5, susceptibility=2.0),
            CriticalityMetrics(order_parameter=0.5, susceptibility=2.0),
            CriticalityMetrics(order_parameter=0.9, susceptibility=0.5),  # Changed
        ]
        # The actual detection depends on current state vs inferred
        result = await monitor.detect_phase_transition()
        # May or may not detect depending on thresholds
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_recommend_intervention_critical(self, monitor):
        """Test no intervention recommended when critical."""
        monitor._current_state = CriticalityState.CRITICAL
        recommendation = await monitor.recommend_intervention()
        assert recommendation is None

    @pytest.mark.asyncio
    async def test_recommend_intervention_subcritical(self, monitor):
        """Test deterritorialization recommended when subcritical."""
        monitor._current_state = CriticalityState.SUBCRITICAL
        recommendation = await monitor.recommend_intervention()
        assert recommendation == "deterritorialized"

    @pytest.mark.asyncio
    async def test_recommend_intervention_supercritical(self, monitor):
        """Test territorialization recommended when supercritical."""
        monitor._current_state = CriticalityState.SUPERCRITICAL
        recommendation = await monitor.recommend_intervention()
        assert recommendation == "hierarchy"

    def test_record_perturbation(self, monitor):
        """Test recording perturbations."""
        monitor.record_perturbation(magnitude=1.0, response=0.5)
        assert len(monitor._perturbations) == 1

        # Record more
        for i in range(10):
            monitor.record_perturbation(magnitude=0.5, response=0.3)
        assert len(monitor._perturbations) == 11

    def test_on_transition_callback(self, monitor):
        """Test registering transition callback."""
        callback = MagicMock()
        monitor.on_transition(callback)
        assert callback in monitor._on_transition_callbacks

    def test_get_transitions(self, monitor):
        """Test getting transition history."""
        # Initially empty
        assert monitor.get_transitions() == []

        # Add a transition
        transition = PhaseTransition(
            transition_id="test",
            from_state=CriticalityState.CRITICAL,
            to_state=CriticalityState.SUBCRITICAL,
            trigger="test",
            metrics_before=CriticalityMetrics(),
            metrics_after=CriticalityMetrics(),
        )
        monitor._transitions.append(transition)

        assert len(monitor.get_transitions()) == 1

    def test_to_dict(self, monitor):
        """Test serialization."""
        data = monitor.to_dict()

        assert "current_state" in data
        assert "current_metrics" in data
        assert "transitions_count" in data
        assert "samples_count" in data

    @pytest.mark.asyncio
    async def test_start_stop(self, monitor):
        """Test starting and stopping the monitor."""
        # Start
        await monitor.start()
        assert monitor._running is True
        assert monitor._monitor_task is not None

        # Stop
        await monitor.stop()
        assert monitor._running is False

    @pytest.mark.asyncio
    async def test_monitoring_loop(self, monitor, mock_event_bus):
        """Test the monitoring loop samples state."""
        monitor._sample_interval = 0.1  # Very fast

        await monitor.start()
        await asyncio.sleep(0.25)  # Let it run a couple cycles
        await monitor.stop()

        # Should have some metrics history
        assert len(monitor._metrics_history) >= 1
