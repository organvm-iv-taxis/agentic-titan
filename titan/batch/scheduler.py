"""
Titan Batch - Load-Aware Scheduler

Implements intelligent scheduling for batch sessions based on
system load, worker availability, and runtime characteristics.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from titan.batch.models import BatchJob, QueuedSession

logger = logging.getLogger("titan.batch.scheduler")


# =============================================================================
# System Load Detection
# =============================================================================

class LoadLevel(str, Enum):
    """System load level classification."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemLoad:
    """Current system resource utilization."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    load_average_1m: float = 0.0
    load_average_5m: float = 0.0
    load_average_15m: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def level(self) -> LoadLevel:
        """Classify current load level."""
        if self.cpu_percent > 90 or self.memory_percent > 90:
            return LoadLevel.CRITICAL
        if self.cpu_percent > 80 or self.memory_percent > 85:
            return LoadLevel.HIGH
        if self.cpu_percent > 50 or self.memory_percent > 60:
            return LoadLevel.MODERATE
        return LoadLevel.LOW

    @property
    def should_offload(self) -> bool:
        """Whether tasks should be offloaded to remote workers."""
        return self.level in (LoadLevel.HIGH, LoadLevel.CRITICAL)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_percent": round(self.memory_percent, 1),
            "disk_percent": round(self.disk_percent, 1),
            "load_average_1m": round(self.load_average_1m, 2),
            "load_average_5m": round(self.load_average_5m, 2),
            "load_average_15m": round(self.load_average_15m, 2),
            "level": self.level.value,
            "should_offload": self.should_offload,
            "timestamp": self.timestamp.isoformat(),
        }


def get_system_load() -> SystemLoad:
    """
    Get current system resource utilization.

    Uses psutil if available, falls back to os.getloadavg().
    """
    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        load_avg = os.getloadavg()

        return SystemLoad(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            load_average_1m=load_avg[0],
            load_average_5m=load_avg[1],
            load_average_15m=load_avg[2],
        )

    except ImportError:
        # psutil not installed
        load_avg = os.getloadavg()
        # Estimate CPU from load average (rough approximation)
        cpu_count = os.cpu_count() or 1
        cpu_estimate = min(100, (load_avg[0] / cpu_count) * 100)

        return SystemLoad(
            cpu_percent=cpu_estimate,
            memory_percent=50.0,  # Unknown
            disk_percent=50.0,  # Unknown
            load_average_1m=load_avg[0],
            load_average_5m=load_avg[1],
            load_average_15m=load_avg[2],
        )


# =============================================================================
# Worker Status
# =============================================================================

@dataclass
class WorkerStatus:
    """Status of a Celery worker."""

    worker_id: str
    hostname: str
    runtime_type: str = "local"
    concurrency: int = 1
    active_tasks: int = 0
    available_slots: int = 1
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    is_healthy: bool = True

    @property
    def utilization(self) -> float:
        """Worker utilization percentage."""
        if self.concurrency == 0:
            return 100.0
        return (self.active_tasks / self.concurrency) * 100

    @property
    def is_available(self) -> bool:
        """Whether worker can accept new tasks."""
        return self.is_healthy and self.available_slots > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "runtime_type": self.runtime_type,
            "concurrency": self.concurrency,
            "active_tasks": self.active_tasks,
            "available_slots": self.available_slots,
            "utilization": round(self.utilization, 1),
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_percent": round(self.memory_percent, 1),
            "is_healthy": self.is_healthy,
            "is_available": self.is_available,
            "last_heartbeat": self.last_heartbeat.isoformat(),
        }


# =============================================================================
# Scheduling Strategy
# =============================================================================

class SchedulingStrategy(str, Enum):
    """Task scheduling strategy."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PREFER_LOCAL = "prefer_local"
    PREFER_REMOTE = "prefer_remote"
    LOAD_AWARE = "load_aware"


@dataclass
class SchedulingDecision:
    """Result of a scheduling decision."""

    worker_id: str | None
    runtime_type: str
    queue: str
    reasoning: list[str]
    system_load: SystemLoad
    selected_worker: WorkerStatus | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "worker_id": self.worker_id,
            "runtime_type": self.runtime_type,
            "queue": self.queue,
            "reasoning": self.reasoning,
            "system_load": self.system_load.to_dict(),
            "selected_worker": self.selected_worker.to_dict() if self.selected_worker else None,
        }


# =============================================================================
# Batch Scheduler
# =============================================================================

class BatchScheduler:
    """
    Load-aware scheduler for batch processing.

    Features:
    - System load monitoring
    - Worker health tracking
    - Intelligent task routing
    - Automatic offloading under high load
    """

    def __init__(
        self,
        strategy: SchedulingStrategy = SchedulingStrategy.LOAD_AWARE,
        local_threshold_cpu: float = 80.0,
        local_threshold_memory: float = 85.0,
        worker_health_timeout: float = 60.0,
    ) -> None:
        """
        Initialize the scheduler.

        Args:
            strategy: Default scheduling strategy
            local_threshold_cpu: CPU % above which to prefer remote
            local_threshold_memory: Memory % above which to prefer remote
            worker_health_timeout: Seconds before worker considered unhealthy
        """
        self.strategy = strategy
        self.local_threshold_cpu = local_threshold_cpu
        self.local_threshold_memory = local_threshold_memory
        self.worker_health_timeout = worker_health_timeout

        self._workers: dict[str, WorkerStatus] = {}
        self._round_robin_index = 0

        logger.info(f"Batch scheduler initialized with strategy: {strategy.value}")

    # =========================================================================
    # Worker Management
    # =========================================================================

    def register_worker(self, status: WorkerStatus) -> None:
        """Register or update a worker."""
        self._workers[status.worker_id] = status
        logger.debug(f"Registered worker: {status.worker_id}")

    def deregister_worker(self, worker_id: str) -> None:
        """Deregister a worker."""
        if worker_id in self._workers:
            del self._workers[worker_id]
            logger.debug(f"Deregistered worker: {worker_id}")

    def update_worker_status(
        self,
        worker_id: str,
        active_tasks: int | None = None,
        cpu_percent: float | None = None,
        memory_percent: float | None = None,
    ) -> None:
        """Update worker status metrics."""
        if worker_id not in self._workers:
            return

        worker = self._workers[worker_id]
        if active_tasks is not None:
            worker.active_tasks = active_tasks
            worker.available_slots = max(0, worker.concurrency - active_tasks)
        if cpu_percent is not None:
            worker.cpu_percent = cpu_percent
        if memory_percent is not None:
            worker.memory_percent = memory_percent
        worker.last_heartbeat = datetime.now()

    def get_available_workers(self) -> list[WorkerStatus]:
        """Get list of available workers."""
        now = datetime.now()
        available = []

        for worker in self._workers.values():
            # Check health timeout
            elapsed = (now - worker.last_heartbeat).total_seconds()
            worker.is_healthy = elapsed < self.worker_health_timeout

            if worker.is_available:
                available.append(worker)

        return available

    async def refresh_workers_from_celery(self) -> list[WorkerStatus]:
        """
        Refresh worker list from Celery.

        Returns updated list of workers.
        """
        try:
            from titan.batch.celery_app import celery_app

            inspect = celery_app.control.inspect()
            stats = inspect.stats() or {}
            active = inspect.active() or {}

            for worker_name, worker_stats in stats.items():
                pool = worker_stats.get("pool", {})
                concurrency = pool.get("max-concurrency", 1)
                active_tasks = len(active.get(worker_name, []))

                status = WorkerStatus(
                    worker_id=worker_name,
                    hostname=worker_stats.get("hostname", worker_name),
                    runtime_type=worker_stats.get("runtime_type", "local"),
                    concurrency=concurrency,
                    active_tasks=active_tasks,
                    available_slots=max(0, concurrency - active_tasks),
                )
                self.register_worker(status)

            return self.get_available_workers()

        except Exception as e:
            logger.warning(f"Failed to refresh workers from Celery: {e}")
            return []

    # =========================================================================
    # Scheduling
    # =========================================================================

    def schedule(
        self,
        session: "QueuedSession",
        batch: "BatchJob",
        strategy: SchedulingStrategy | None = None,
    ) -> SchedulingDecision:
        """
        Determine where to run a session.

        Args:
            session: Session to schedule
            batch: Parent batch job
            strategy: Override default strategy

        Returns:
            SchedulingDecision with routing information
        """
        strategy = strategy or self.strategy
        system_load = get_system_load()
        reasoning = []

        # Get available workers
        workers = self.get_available_workers()
        remote_workers = [w for w in workers if w.runtime_type != "local"]
        local_workers = [w for w in workers if w.runtime_type == "local"]

        reasoning.append(f"System load: {system_load.level.value}")
        reasoning.append(f"Available workers: {len(workers)} ({len(local_workers)} local, {len(remote_workers)} remote)")

        # Apply strategy
        if strategy == SchedulingStrategy.LOAD_AWARE:
            return self._schedule_load_aware(
                system_load, workers, local_workers, remote_workers, reasoning
            )
        elif strategy == SchedulingStrategy.PREFER_LOCAL:
            return self._schedule_prefer_local(
                system_load, local_workers, remote_workers, reasoning
            )
        elif strategy == SchedulingStrategy.PREFER_REMOTE:
            return self._schedule_prefer_remote(
                system_load, remote_workers, local_workers, reasoning
            )
        elif strategy == SchedulingStrategy.LEAST_LOADED:
            return self._schedule_least_loaded(
                system_load, workers, reasoning
            )
        else:  # ROUND_ROBIN
            return self._schedule_round_robin(
                system_load, workers, reasoning
            )

    def _schedule_load_aware(
        self,
        system_load: SystemLoad,
        workers: list[WorkerStatus],
        local_workers: list[WorkerStatus],
        remote_workers: list[WorkerStatus],
        reasoning: list[str],
    ) -> SchedulingDecision:
        """Load-aware scheduling: prefer remote under high load."""
        if system_load.should_offload:
            reasoning.append("High system load detected, preferring remote workers")
            if remote_workers:
                worker = min(remote_workers, key=lambda w: w.utilization)
                reasoning.append(f"Selected remote worker: {worker.worker_id} ({worker.utilization:.1f}% utilized)")
                return SchedulingDecision(
                    worker_id=worker.worker_id,
                    runtime_type=worker.runtime_type,
                    queue="titan.batch.inquiry",
                    reasoning=reasoning,
                    system_load=system_load,
                    selected_worker=worker,
                )
            reasoning.append("No remote workers available, falling back to local")

        # Use local if load is acceptable
        if local_workers:
            worker = min(local_workers, key=lambda w: w.utilization)
            reasoning.append(f"Using local worker: {worker.worker_id}")
            return SchedulingDecision(
                worker_id=worker.worker_id,
                runtime_type="local",
                queue="titan.batch.inquiry",
                reasoning=reasoning,
                system_load=system_load,
                selected_worker=worker,
            )

        # No workers available
        reasoning.append("No workers available, using default queue")
        return SchedulingDecision(
            worker_id=None,
            runtime_type="auto",
            queue="titan.batch.inquiry",
            reasoning=reasoning,
            system_load=system_load,
        )

    def _schedule_prefer_local(
        self,
        system_load: SystemLoad,
        local_workers: list[WorkerStatus],
        remote_workers: list[WorkerStatus],
        reasoning: list[str],
    ) -> SchedulingDecision:
        """Prefer local workers."""
        if local_workers:
            worker = min(local_workers, key=lambda w: w.utilization)
            reasoning.append(f"Selected local worker: {worker.worker_id}")
            return SchedulingDecision(
                worker_id=worker.worker_id,
                runtime_type="local",
                queue="titan.batch.inquiry",
                reasoning=reasoning,
                system_load=system_load,
                selected_worker=worker,
            )

        if remote_workers:
            worker = min(remote_workers, key=lambda w: w.utilization)
            reasoning.append(f"No local workers, using remote: {worker.worker_id}")
            return SchedulingDecision(
                worker_id=worker.worker_id,
                runtime_type=worker.runtime_type,
                queue="titan.batch.inquiry",
                reasoning=reasoning,
                system_load=system_load,
                selected_worker=worker,
            )

        reasoning.append("No workers available")
        return SchedulingDecision(
            worker_id=None,
            runtime_type="auto",
            queue="titan.batch.inquiry",
            reasoning=reasoning,
            system_load=system_load,
        )

    def _schedule_prefer_remote(
        self,
        system_load: SystemLoad,
        remote_workers: list[WorkerStatus],
        local_workers: list[WorkerStatus],
        reasoning: list[str],
    ) -> SchedulingDecision:
        """Prefer remote workers."""
        if remote_workers:
            worker = min(remote_workers, key=lambda w: w.utilization)
            reasoning.append(f"Selected remote worker: {worker.worker_id}")
            return SchedulingDecision(
                worker_id=worker.worker_id,
                runtime_type=worker.runtime_type,
                queue="titan.batch.inquiry",
                reasoning=reasoning,
                system_load=system_load,
                selected_worker=worker,
            )

        if local_workers:
            worker = min(local_workers, key=lambda w: w.utilization)
            reasoning.append(f"No remote workers, using local: {worker.worker_id}")
            return SchedulingDecision(
                worker_id=worker.worker_id,
                runtime_type="local",
                queue="titan.batch.inquiry",
                reasoning=reasoning,
                system_load=system_load,
                selected_worker=worker,
            )

        reasoning.append("No workers available")
        return SchedulingDecision(
            worker_id=None,
            runtime_type="auto",
            queue="titan.batch.inquiry",
            reasoning=reasoning,
            system_load=system_load,
        )

    def _schedule_least_loaded(
        self,
        system_load: SystemLoad,
        workers: list[WorkerStatus],
        reasoning: list[str],
    ) -> SchedulingDecision:
        """Schedule to least loaded worker."""
        if workers:
            worker = min(workers, key=lambda w: w.utilization)
            reasoning.append(f"Least loaded worker: {worker.worker_id} ({worker.utilization:.1f}%)")
            return SchedulingDecision(
                worker_id=worker.worker_id,
                runtime_type=worker.runtime_type,
                queue="titan.batch.inquiry",
                reasoning=reasoning,
                system_load=system_load,
                selected_worker=worker,
            )

        reasoning.append("No workers available")
        return SchedulingDecision(
            worker_id=None,
            runtime_type="auto",
            queue="titan.batch.inquiry",
            reasoning=reasoning,
            system_load=system_load,
        )

    def _schedule_round_robin(
        self,
        system_load: SystemLoad,
        workers: list[WorkerStatus],
        reasoning: list[str],
    ) -> SchedulingDecision:
        """Round-robin scheduling."""
        if workers:
            self._round_robin_index = (self._round_robin_index + 1) % len(workers)
            worker = workers[self._round_robin_index]
            reasoning.append(f"Round robin selected: {worker.worker_id}")
            return SchedulingDecision(
                worker_id=worker.worker_id,
                runtime_type=worker.runtime_type,
                queue="titan.batch.inquiry",
                reasoning=reasoning,
                system_load=system_load,
                selected_worker=worker,
            )

        reasoning.append("No workers available")
        return SchedulingDecision(
            worker_id=None,
            runtime_type="auto",
            queue="titan.batch.inquiry",
            reasoning=reasoning,
            system_load=system_load,
        )


# =============================================================================
# Factory Functions
# =============================================================================

_default_scheduler: BatchScheduler | None = None


def get_batch_scheduler() -> BatchScheduler:
    """Get the default batch scheduler instance."""
    global _default_scheduler
    if _default_scheduler is None:
        _default_scheduler = BatchScheduler()
    return _default_scheduler
