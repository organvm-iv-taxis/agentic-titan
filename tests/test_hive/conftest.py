"""Fixtures and configuration for hive tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    bus = MagicMock()
    bus.emit = AsyncMock()
    bus.subscribe = MagicMock()
    bus.unsubscribe = MagicMock()
    return bus


@pytest.fixture
def mock_neighborhood():
    """Create a mock TopologicalNeighborhood."""
    neighborhood = MagicMock()
    neighborhood.get_network_stats.return_value = {
        "total_agents": 10,
        "average_clustering": 0.5,
        "density": 0.4,
        "neighbor_count": 7,
    }
    neighborhood._profiles = {}
    neighborhood.neighbor_count = 7
    return neighborhood


@pytest.fixture
def mock_pheromone_field():
    """Create a mock PheromoneField."""
    field = MagicMock()
    field.deposit = AsyncMock()
    field.sense = AsyncMock(return_value=[])
    field.sense_gradient = AsyncMock()
    field.follow_strongest = AsyncMock(return_value=None)
    return field
