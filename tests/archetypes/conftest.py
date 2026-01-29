"""Shared fixtures for archetype tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


@dataclass
class MockLLMResponse:
    """Mock LLM response for testing."""

    content: str = "Mock LLM response"
    model: str = "mock-model"
    usage: dict[str, int] | None = None
    finish_reason: str = "stop"

    def __post_init__(self):
        if self.usage is None:
            self.usage = {"prompt_tokens": 100, "completion_tokens": 50}


@pytest.fixture
def mock_llm_router():
    """Create a mock LLM router for testing."""
    router = AsyncMock()
    router.initialize = AsyncMock()
    router.complete = AsyncMock(return_value=MockLLMResponse())

    # Make it return different responses based on content
    def smart_complete(messages, **kwargs):
        system = kwargs.get("system", "")
        content = messages[0].content if messages else ""

        # Generate appropriate mock responses
        if "plan" in content.lower():
            return MockLLMResponse(content="1. Analyze requirements\n2. Design solution\n3. Implement")
        elif "code" in content.lower() or "write" in content.lower():
            return MockLLMResponse(content="```python\ndef example():\n    return 'Hello, World!'\n```")
        elif "test" in content.lower():
            return MockLLMResponse(content="```python\ndef test_example():\n    assert example() == 'Hello, World!'\n```")
        elif "review" in content.lower():
            return MockLLMResponse(content="No issues found. Code looks good.")
        elif "research" in content.lower() or "question" in content.lower():
            return MockLLMResponse(content="Research finding: The topic is complex but manageable.")
        elif "synthesize" in content.lower() or "summary" in content.lower():
            return MockLLMResponse(content="Summary: All findings have been analyzed and synthesized.")
        elif "decompose" in content.lower() or "subtask" in content.lower():
            return MockLLMResponse(content="SUBTASK: Research requirements\nAGENT: researcher\nDEPENDS: none\n---\nSUBTASK: Write code\nAGENT: coder\nDEPENDS: st-0")
        elif "topology" in content.lower():
            return MockLLMResponse(content="Recommended topology: pipeline")
        elif "aggregate" in content.lower():
            return MockLLMResponse(content="Final aggregated result from all subtasks.")
        else:
            return MockLLMResponse(content=f"Mock response for: {content[:50]}...")

    router.complete.side_effect = smart_complete
    return router


@pytest.fixture
def mock_hive_mind():
    """Create a mock HiveMind for testing."""
    hive = AsyncMock()
    hive.remember = AsyncMock(return_value="memory-id-123")
    hive.recall = AsyncMock(return_value=[])
    hive.broadcast = AsyncMock()
    hive.send = AsyncMock()
    hive.subscribe = AsyncMock()
    hive.set = AsyncMock()
    hive.get = AsyncMock(return_value=None)
    hive.set_topology = AsyncMock()
    return hive


@pytest.fixture
def mock_topology_engine():
    """Create a mock TopologyEngine for testing."""
    engine = MagicMock()
    engine.current_topology = None
    engine.suggest_topology = MagicMock(return_value={"recommended": "pipeline"})
    engine.select_topology = MagicMock(return_value="pipeline")
    engine.create_topology = MagicMock()
    return engine


@pytest.fixture
def mock_audit_logger():
    """Create a mock AuditLogger for testing."""
    logger = AsyncMock()
    logger.log_agent_started = AsyncMock()
    logger.log_agent_completed = AsyncMock()
    logger.log_agent_failed = AsyncMock()
    logger.log_decision = AsyncMock()
    logger.log_event = AsyncMock()
    return logger


@pytest.fixture
def mock_pheromone_field():
    """Create a mock PheromoneField for testing."""
    field = AsyncMock()
    field.deposit = AsyncMock()
    field.sense = AsyncMock(return_value=[])
    field.sense_radius = AsyncMock(return_value={})
    field.sense_gradient = AsyncMock()
    field.follow_strongest = AsyncMock(return_value=None)
    field.start = AsyncMock()
    field.stop = AsyncMock()
    return field


@pytest.fixture
def mock_event_bus():
    """Create a mock EventBus for testing."""
    bus = AsyncMock()
    bus.emit = AsyncMock()
    bus.subscribe = AsyncMock()
    bus.unsubscribe = AsyncMock()
    return bus


@pytest.fixture
def agent_factory(mock_llm_router, mock_hive_mind, mock_audit_logger):
    """Factory to create agents with common mocks injected."""

    def _create_agent(agent_class, **kwargs):
        # Patch the router before creating
        import adapters.router

        original_get_router = adapters.router.get_router

        def patched_get_router():
            return mock_llm_router

        adapters.router.get_router = patched_get_router

        try:
            kwargs.setdefault("hive_mind", mock_hive_mind)
            kwargs.setdefault("audit_logger", mock_audit_logger)
            kwargs.setdefault("max_turns", 5)
            kwargs.setdefault("timeout_ms", 10000)
            agent = agent_class(**kwargs)
            agent._router = mock_llm_router
            return agent
        finally:
            adapters.router.get_router = original_get_router

    return _create_agent


@pytest.fixture
def patch_router(mock_llm_router):
    """Context manager to patch the router globally."""
    import adapters.router

    original = adapters.router.get_router

    def patched():
        return mock_llm_router

    adapters.router.get_router = patched
    yield mock_llm_router
    adapters.router.get_router = original
