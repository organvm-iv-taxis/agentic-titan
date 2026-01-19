"""
Migration Manager - Coordinate agent migrations between runtimes.

Handles:
- Pausing agents for migration
- State transfer
- Runtime spawning
- Health verification
- Rollback on failure
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable

from hive.migration.state import AgentState, StateSnapshot
from hive.migration.runtime import RuntimeType, RuntimeSelector, RuntimeConfig

logger = logging.getLogger("titan.migration.manager")


class MigrationStatus(str, Enum):
    """Migration status."""

    PENDING = "pending"
    PREPARING = "preparing"  # Taking snapshot
    TRANSFERRING = "transferring"  # Moving state
    SPAWNING = "spawning"  # Starting in new runtime
    VERIFYING = "verifying"  # Health check
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class MigrationRequest:
    """Request to migrate an agent."""

    agent_id: str
    source_runtime: str
    target_runtime: str | None = None  # None = auto-select

    # Options
    force: bool = False  # Migrate even if not recommended
    rollback_on_failure: bool = True
    timeout_seconds: int = 300

    # Metadata
    reason: str = ""


@dataclass
class MigrationResult:
    """Result of a migration attempt."""

    id: str
    agent_id: str
    status: MigrationStatus

    # Runtimes
    source_runtime: str
    target_runtime: str

    # Timing
    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float = 0.0

    # State
    snapshot_id: str = ""
    new_instance_id: str = ""

    # Details
    error: str | None = None
    steps_completed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "source_runtime": self.source_runtime,
            "target_runtime": self.target_runtime,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "snapshot_id": self.snapshot_id,
            "new_instance_id": self.new_instance_id,
            "error": self.error,
            "steps_completed": self.steps_completed,
        }


class MigrationManager:
    """
    Manages agent migrations between runtimes.

    Migration steps:
    1. Pause agent in source runtime
    2. Take state snapshot
    3. Transfer state to target
    4. Spawn agent in target runtime
    5. Verify health
    6. Cleanup source (or rollback on failure)
    """

    def __init__(
        self,
        runtime_selector: RuntimeSelector | None = None,
    ) -> None:
        self._selector = runtime_selector or RuntimeSelector()
        self._active_migrations: dict[str, MigrationResult] = {}
        self._history: list[MigrationResult] = []

        # Callbacks for runtime operations
        self._pause_callbacks: dict[str, Callable[[str], Awaitable[bool]]] = {}
        self._resume_callbacks: dict[str, Callable[[str], Awaitable[bool]]] = {}
        self._spawn_callbacks: dict[str, Callable[[AgentState], Awaitable[str]]] = {}
        self._health_callbacks: dict[str, Callable[[str], Awaitable[bool]]] = {}
        self._cleanup_callbacks: dict[str, Callable[[str], Awaitable[None]]] = {}

    def register_runtime_callbacks(
        self,
        runtime: str,
        *,
        pause: Callable[[str], Awaitable[bool]] | None = None,
        resume: Callable[[str], Awaitable[bool]] | None = None,
        spawn: Callable[[AgentState], Awaitable[str]] | None = None,
        health: Callable[[str], Awaitable[bool]] | None = None,
        cleanup: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        """Register callbacks for a runtime."""
        if pause:
            self._pause_callbacks[runtime] = pause
        if resume:
            self._resume_callbacks[runtime] = resume
        if spawn:
            self._spawn_callbacks[runtime] = spawn
        if health:
            self._health_callbacks[runtime] = health
        if cleanup:
            self._cleanup_callbacks[runtime] = cleanup

        logger.info(f"Registered callbacks for runtime: {runtime}")

    async def migrate(
        self,
        request: MigrationRequest,
        agent_state: AgentState,
    ) -> MigrationResult:
        """
        Migrate an agent to a new runtime.

        Args:
            request: Migration request
            agent_state: Current agent state

        Returns:
            MigrationResult with outcome
        """
        migration_id = f"mig_{uuid.uuid4().hex[:8]}"

        result = MigrationResult(
            id=migration_id,
            agent_id=request.agent_id,
            status=MigrationStatus.PENDING,
            source_runtime=request.source_runtime,
            target_runtime=request.target_runtime or "",
            started_at=datetime.now(),
        )

        self._active_migrations[migration_id] = result

        try:
            # Select target runtime if not specified
            if not request.target_runtime:
                from hive.migration.runtime import AgentRequirements

                requirements = self._infer_requirements(agent_state)
                target = self._selector.select(
                    requirements,
                    exclude=[request.source_runtime],
                )
                result.target_runtime = target.name
            else:
                result.target_runtime = request.target_runtime

            logger.info(
                f"Starting migration {migration_id}: "
                f"{request.source_runtime} -> {result.target_runtime}"
            )

            # Step 1: Prepare (pause and snapshot)
            result.status = MigrationStatus.PREPARING
            snapshot = await self._prepare_migration(
                request, agent_state, result
            )

            # Step 2: Transfer state
            result.status = MigrationStatus.TRANSFERRING
            await self._transfer_state(snapshot, result)

            # Step 3: Spawn in target
            result.status = MigrationStatus.SPAWNING
            new_instance = await self._spawn_in_target(
                agent_state, result
            )

            # Step 4: Verify health
            result.status = MigrationStatus.VERIFYING
            healthy = await self._verify_health(
                new_instance, result
            )

            if not healthy:
                raise RuntimeError("Health check failed in target runtime")

            # Step 5: Cleanup source
            await self._cleanup_source(request, result)

            # Success
            result.status = MigrationStatus.COMPLETED
            result.completed_at = datetime.now()
            result.duration_seconds = (
                result.completed_at - result.started_at
            ).total_seconds()

            logger.info(
                f"Migration {migration_id} completed in "
                f"{result.duration_seconds:.1f}s"
            )

        except Exception as e:
            logger.error(f"Migration {migration_id} failed: {e}")
            result.error = str(e)
            result.status = MigrationStatus.FAILED

            # Attempt rollback
            if request.rollback_on_failure:
                await self._rollback(request, agent_state, result)

        finally:
            self._active_migrations.pop(migration_id, None)
            self._history.append(result)

        return result

    async def _prepare_migration(
        self,
        request: MigrationRequest,
        state: AgentState,
        result: MigrationResult,
    ) -> StateSnapshot:
        """Pause agent and take snapshot."""
        # Pause in source runtime
        pause_cb = self._pause_callbacks.get(request.source_runtime)
        if pause_cb:
            paused = await pause_cb(request.agent_id)
            if not paused:
                raise RuntimeError("Failed to pause agent")
            result.steps_completed.append("paused")

        # Take snapshot
        snapshot = state.snapshot()
        result.snapshot_id = snapshot.id
        result.steps_completed.append("snapshot_taken")

        logger.debug(f"Prepared migration: snapshot {snapshot.id}")
        return snapshot

    async def _transfer_state(
        self,
        snapshot: StateSnapshot,
        result: MigrationResult,
    ) -> None:
        """Transfer state to target runtime."""
        # In practice, this would serialize and send to target
        # For now, state is in-memory

        # Verify snapshot integrity
        if not snapshot.verify():
            raise RuntimeError("Snapshot integrity check failed")

        result.steps_completed.append("state_transferred")
        logger.debug("State transferred to target runtime")

    async def _spawn_in_target(
        self,
        state: AgentState,
        result: MigrationResult,
    ) -> str:
        """Spawn agent in target runtime."""
        spawn_cb = self._spawn_callbacks.get(result.target_runtime)
        if spawn_cb:
            instance_id = await spawn_cb(state)
            result.new_instance_id = instance_id
            result.steps_completed.append("spawned_in_target")
            return instance_id

        # Default: mock spawn
        instance_id = f"inst_{uuid.uuid4().hex[:8]}"
        result.new_instance_id = instance_id
        result.steps_completed.append("spawned_in_target")

        logger.debug(f"Spawned in target: {instance_id}")
        return instance_id

    async def _verify_health(
        self,
        instance_id: str,
        result: MigrationResult,
    ) -> bool:
        """Verify agent health in target runtime."""
        health_cb = self._health_callbacks.get(result.target_runtime)
        if health_cb:
            # Retry with backoff
            for attempt in range(3):
                try:
                    if await health_cb(instance_id):
                        result.steps_completed.append("health_verified")
                        return True
                except Exception as e:
                    logger.warning(f"Health check attempt {attempt + 1} failed: {e}")

                await asyncio.sleep(2 ** attempt)

            return False

        # Default: assume healthy
        result.steps_completed.append("health_verified")
        return True

    async def _cleanup_source(
        self,
        request: MigrationRequest,
        result: MigrationResult,
    ) -> None:
        """Cleanup agent in source runtime."""
        cleanup_cb = self._cleanup_callbacks.get(request.source_runtime)
        if cleanup_cb:
            await cleanup_cb(request.agent_id)

        result.steps_completed.append("source_cleaned")
        logger.debug("Source runtime cleaned up")

    async def _rollback(
        self,
        request: MigrationRequest,
        state: AgentState,
        result: MigrationResult,
    ) -> None:
        """Rollback a failed migration."""
        logger.info(f"Rolling back migration {result.id}")

        try:
            # Resume in source if paused
            resume_cb = self._resume_callbacks.get(request.source_runtime)
            if resume_cb and "paused" in result.steps_completed:
                await resume_cb(request.agent_id)

            # Cleanup target if spawned
            if result.new_instance_id:
                cleanup_cb = self._cleanup_callbacks.get(result.target_runtime)
                if cleanup_cb:
                    await cleanup_cb(result.new_instance_id)

            result.status = MigrationStatus.ROLLED_BACK
            result.steps_completed.append("rolled_back")

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            result.error = f"Rollback failed: {e}"

    def _infer_requirements(self, state: AgentState) -> Any:
        """Infer agent requirements from state."""
        from hive.migration.runtime import AgentRequirements

        return AgentRequirements(
            needs_gpu=state.context.get("needs_gpu", False),
            needs_long_running=state.turn_number > 10,
            min_memory_mb=256,
            expected_duration_seconds=state.context.get(
                "expected_duration", 60
            ),
        )

    def get_active_migrations(self) -> list[MigrationResult]:
        """Get currently active migrations."""
        return list(self._active_migrations.values())

    def get_history(self, limit: int = 10) -> list[MigrationResult]:
        """Get migration history."""
        return self._history[-limit:]

    async def auto_migrate(
        self,
        agent_states: dict[str, AgentState],
    ) -> list[MigrationResult]:
        """
        Automatically migrate agents to optimal runtimes.

        Analyzes current placement and suggests/executes migrations.
        """
        results: list[MigrationResult] = []

        for agent_id, state in agent_states.items():
            current_runtime = state.context.get("runtime", "local")
            requirements = self._infer_requirements(state)

            optimal = self._selector.select(requirements)

            if optimal.name != current_runtime:
                logger.info(
                    f"Auto-migrating {agent_id}: "
                    f"{current_runtime} -> {optimal.name}"
                )

                request = MigrationRequest(
                    agent_id=agent_id,
                    source_runtime=current_runtime,
                    target_runtime=optimal.name,
                    reason="auto-optimization",
                )

                result = await self.migrate(request, state)
                results.append(result)

        return results
