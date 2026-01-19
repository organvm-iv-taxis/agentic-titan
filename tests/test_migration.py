"""
Tests for agent migration system.
"""

import pytest

from hive.migration.state import AgentState, StateSnapshot
from hive.migration.runtime import (
    RuntimeType,
    RuntimeSelector,
    RuntimeConfig,
    RuntimeCapabilities,
    AgentRequirements,
)
from hive.migration.manager import (
    MigrationManager,
    MigrationRequest,
    MigrationStatus,
)


# ============================================================================
# State Tests
# ============================================================================


class TestStateSnapshot:
    """Test StateSnapshot."""

    def test_create_snapshot(self) -> None:
        """Test creating a snapshot."""
        from datetime import datetime

        snap = StateSnapshot(
            id="snap_test",
            timestamp=datetime.now(),
            agent_id="agent_1",
            agent_type="researcher",
            task="Research topic",
            context={"key": "value"},
            memory=[{"entry": "1"}],
            turn_number=5,
            tool_calls_made=3,
            llm_calls_made=2,
            status="running",
            last_action="search",
            last_result="found results",
        )

        assert snap.id == "snap_test"
        assert snap.agent_id == "agent_1"
        assert snap.checksum != ""

    def test_verify_integrity(self) -> None:
        """Test integrity verification."""
        from datetime import datetime

        snap = StateSnapshot(
            id="snap_1",
            timestamp=datetime.now(),
            agent_id="a1",
            agent_type="coder",
            task="Code task",
            context={},
            memory=[],
            turn_number=0,
            tool_calls_made=0,
            llm_calls_made=0,
            status="pending",
            last_action="",
            last_result=None,
        )

        # Should pass
        assert snap.verify() is True

        # Tamper with state
        snap.task = "Modified task"

        # Should fail
        assert snap.verify() is False

    def test_serialization(self) -> None:
        """Test snapshot serialization."""
        from datetime import datetime

        snap = StateSnapshot(
            id="snap_ser",
            timestamp=datetime.now(),
            agent_id="a1",
            agent_type="reviewer",
            task="Review code",
            context={"file": "main.py"},
            memory=[],
            turn_number=1,
            tool_calls_made=0,
            llm_calls_made=0,
            status="running",
            last_action="read_file",
            last_result="content",
        )

        # To dict and back
        d = snap.to_dict()
        restored = StateSnapshot.from_dict(d)

        assert restored.id == snap.id
        assert restored.agent_id == snap.agent_id
        assert restored.task == snap.task
        assert restored.context == snap.context


class TestAgentState:
    """Test AgentState."""

    def test_create_state(self) -> None:
        """Test creating agent state."""
        state = AgentState(
            agent_id="agent_1",
            agent_type="researcher",
            task="Research AI",
        )

        assert state.agent_id == "agent_1"
        assert state.status == "pending"
        assert state.turn_number == 0

    def test_snapshot_and_restore(self) -> None:
        """Test snapshot creation and restoration."""
        state = AgentState(
            agent_id="a1",
            agent_type="coder",
            task="Write code",
        )

        # Modify state
        state.turn_number = 5
        state.tool_calls_made = 3
        state.status = "running"

        # Take snapshot
        snap = state.snapshot()

        # Modify more
        state.turn_number = 10
        state.status = "completed"

        # Restore
        success = state.restore(snap)

        assert success is True
        assert state.turn_number == 5
        assert state.status == "running"

    def test_progress_tracking(self) -> None:
        """Test progress tracking."""
        state = AgentState(
            agent_id="a1",
            agent_type="simple",
            task="Task",
        )

        state.update_progress("action1", "result1", increment_turn=True)
        state.record_tool_call()
        state.record_llm_call()

        assert state.turn_number == 1
        assert state.tool_calls_made == 1
        assert state.llm_calls_made == 1
        assert state.last_action == "action1"

    def test_json_serialization(self) -> None:
        """Test JSON serialization."""
        state = AgentState(
            agent_id="a1",
            agent_type="researcher",
            task="Research",
            context={"key": "value"},
        )
        state.add_memory({"type": "observation", "content": "Found data"})

        # To JSON and back
        json_str = state.to_json()
        restored = AgentState.from_json(json_str)

        assert restored.agent_id == state.agent_id
        assert restored.context == state.context
        assert len(restored.memory) == 1


# ============================================================================
# Runtime Selector Tests
# ============================================================================


class TestRuntimeSelector:
    """Test RuntimeSelector."""

    @pytest.fixture
    def selector(self) -> RuntimeSelector:
        return RuntimeSelector()

    def test_default_runtimes(self, selector: RuntimeSelector) -> None:
        """Test default runtimes are registered."""
        runtimes = selector.list_runtimes()

        names = {r.name for r in runtimes}
        assert "local" in names
        assert "k3s" in names
        assert "openfaas" in names

    def test_select_local_for_gpu(self, selector: RuntimeSelector) -> None:
        """Test GPU requirement selects local."""
        requirements = AgentRequirements(
            needs_gpu=True,
        )

        selected = selector.select(requirements)

        assert selected.name == "local"

    def test_select_for_long_running(self, selector: RuntimeSelector) -> None:
        """Test long-running task selection."""
        requirements = AgentRequirements(
            needs_long_running=True,
            expected_duration_seconds=1000,
        )

        selected = selector.select(requirements)

        # Should not be serverless (5 min limit)
        assert selected.type != RuntimeType.SERVERLESS

    def test_select_for_low_cost(self, selector: RuntimeSelector) -> None:
        """Test cost optimization."""
        requirements = AgentRequirements(
            prefer_low_cost=True,
        )

        selected = selector.select(requirements)

        # Local is free
        assert selected.name == "local"

    def test_select_for_isolation(self, selector: RuntimeSelector) -> None:
        """Test isolation preference."""
        requirements = AgentRequirements(
            prefer_isolation=True,
        )

        selected = selector.select(requirements)

        # Should not be local (no isolation)
        # Both container and serverless provide isolation
        assert selected.type in [RuntimeType.CONTAINER, RuntimeType.SERVERLESS]

    def test_select_for_burst(self, selector: RuntimeSelector) -> None:
        """Test burst traffic selection."""
        # Small burst
        small = selector.select_for_burst(3)
        assert small.name == "local"

        # Large burst
        large = selector.select_for_burst(50)
        assert large.type == RuntimeType.SERVERLESS

    def test_register_custom_runtime(self, selector: RuntimeSelector) -> None:
        """Test registering custom runtime."""
        selector.register_runtime(
            RuntimeConfig(
                type=RuntimeType.CONTAINER,
                name="custom-k8s",
                endpoint="http://custom:6443",
            ),
            RuntimeCapabilities(
                supports_gpu=True,
                max_memory_mb=16384,
            ),
        )

        runtime = selector.get_runtime("custom-k8s")

        assert runtime is not None
        assert runtime.name == "custom-k8s"


# ============================================================================
# Migration Manager Tests
# ============================================================================


class TestMigrationManager:
    """Test MigrationManager."""

    @pytest.fixture
    def manager(self) -> MigrationManager:
        return MigrationManager()

    @pytest.fixture
    def sample_state(self) -> AgentState:
        return AgentState(
            agent_id="agent_test",
            agent_type="researcher",
            task="Research task",
        )

    @pytest.mark.asyncio
    async def test_basic_migration(
        self,
        manager: MigrationManager,
        sample_state: AgentState,
    ) -> None:
        """Test basic migration."""
        request = MigrationRequest(
            agent_id="agent_test",
            source_runtime="local",
            target_runtime="k3s",
        )

        result = await manager.migrate(request, sample_state)

        assert result.agent_id == "agent_test"
        assert result.source_runtime == "local"
        assert result.target_runtime == "k3s"
        assert result.status == MigrationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_auto_select_target(
        self,
        manager: MigrationManager,
        sample_state: AgentState,
    ) -> None:
        """Test auto-selecting target runtime."""
        request = MigrationRequest(
            agent_id="agent_test",
            source_runtime="local",
            target_runtime=None,  # Auto-select
        )

        result = await manager.migrate(request, sample_state)

        assert result.target_runtime != ""
        assert result.target_runtime != "local"

    @pytest.mark.asyncio
    async def test_migration_with_callbacks(
        self,
        manager: MigrationManager,
        sample_state: AgentState,
    ) -> None:
        """Test migration with runtime callbacks."""
        paused = False
        spawned_id = ""

        async def mock_pause(agent_id: str) -> bool:
            nonlocal paused
            paused = True
            return True

        async def mock_spawn(state: AgentState) -> str:
            nonlocal spawned_id
            spawned_id = f"new_{state.agent_id}"
            return spawned_id

        async def mock_health(instance_id: str) -> bool:
            return True

        manager.register_runtime_callbacks(
            "local",
            pause=mock_pause,
        )
        manager.register_runtime_callbacks(
            "k3s",
            spawn=mock_spawn,
            health=mock_health,
        )

        request = MigrationRequest(
            agent_id="agent_test",
            source_runtime="local",
            target_runtime="k3s",
        )

        result = await manager.migrate(request, sample_state)

        assert paused is True
        assert spawned_id == "new_agent_test"
        assert result.status == MigrationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_migration_failure_rollback(
        self,
        manager: MigrationManager,
        sample_state: AgentState,
    ) -> None:
        """Test rollback on migration failure."""
        resumed = False

        async def mock_pause(agent_id: str) -> bool:
            return True

        async def mock_resume(agent_id: str) -> bool:
            nonlocal resumed
            resumed = True
            return True

        async def mock_spawn(state: AgentState) -> str:
            raise RuntimeError("Spawn failed")

        manager.register_runtime_callbacks(
            "local",
            pause=mock_pause,
            resume=mock_resume,
        )
        manager.register_runtime_callbacks(
            "k3s",
            spawn=mock_spawn,
        )

        request = MigrationRequest(
            agent_id="agent_test",
            source_runtime="local",
            target_runtime="k3s",
            rollback_on_failure=True,
        )

        result = await manager.migrate(request, sample_state)

        assert result.status == MigrationStatus.ROLLED_BACK
        assert resumed is True

    def test_history(self, manager: MigrationManager) -> None:
        """Test migration history."""
        from hive.migration.manager import MigrationResult
        from datetime import datetime

        # Add mock history
        manager._history.append(MigrationResult(
            id="mig_1",
            agent_id="a1",
            status=MigrationStatus.COMPLETED,
            source_runtime="local",
            target_runtime="k3s",
            started_at=datetime.now(),
        ))

        history = manager.get_history()

        assert len(history) == 1
        assert history[0].agent_id == "a1"

    @pytest.mark.asyncio
    async def test_auto_migrate(
        self,
        manager: MigrationManager,
    ) -> None:
        """Test automatic migration."""
        states = {
            "agent_1": AgentState(
                agent_id="agent_1",
                agent_type="researcher",
                task="Task 1",
                context={"runtime": "local"},
            ),
        }

        # Agent is in local but doesn't need it
        states["agent_1"].turn_number = 1  # Short-running

        results = await manager.auto_migrate(states)

        # Should have considered migration
        assert isinstance(results, list)
