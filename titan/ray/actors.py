"""Ray actors for Titan orchestration.

Provides Ray actors for managing distributed state and coordination.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from titan.ray import RAY_AVAILABLE

if RAY_AVAILABLE:
    import ray

if TYPE_CHECKING:
    from titan.batch.orchestrator import BatchOrchestrator

logger = logging.getLogger("titan.ray.actors")


if RAY_AVAILABLE:
    @ray.remote
    class StateManagerActor:
        """Ray actor for managing distributed state.

        Provides a single source of truth for batch and session state
        across multiple Ray workers.
        """

        def __init__(self):
            """Initialize the state manager."""
            self._batches: dict[str, dict[str, Any]] = {}
            self._sessions: dict[str, dict[str, Any]] = {}
            self._metrics: dict[str, float] = {}
            logger.info("StateManagerActor initialized")

        def set_batch(self, batch_id: str, batch_data: dict[str, Any]) -> None:
            """Store batch state.

            Args:
                batch_id: Batch job ID
                batch_data: Batch data to store
            """
            self._batches[batch_id] = batch_data

        def get_batch(self, batch_id: str) -> dict[str, Any] | None:
            """Get batch state.

            Args:
                batch_id: Batch job ID

            Returns:
                Batch data or None
            """
            return self._batches.get(batch_id)

        def list_batches(self) -> list[dict[str, Any]]:
            """List all batches.

            Returns:
                List of batch data
            """
            return list(self._batches.values())

        def delete_batch(self, batch_id: str) -> bool:
            """Delete batch state.

            Args:
                batch_id: Batch job ID

            Returns:
                True if deleted, False if not found
            """
            if batch_id in self._batches:
                del self._batches[batch_id]
                return True
            return False

        def set_session(self, session_id: str, session_data: dict[str, Any]) -> None:
            """Store session state.

            Args:
                session_id: Session ID
                session_data: Session data to store
            """
            self._sessions[session_id] = session_data

        def get_session(self, session_id: str) -> dict[str, Any] | None:
            """Get session state.

            Args:
                session_id: Session ID

            Returns:
                Session data or None
            """
            return self._sessions.get(session_id)

        def delete_session(self, session_id: str) -> bool:
            """Delete session state.

            Args:
                session_id: Session ID

            Returns:
                True if deleted, False if not found
            """
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

        def increment_metric(self, metric_name: str, value: float = 1.0) -> float:
            """Increment a metric counter.

            Args:
                metric_name: Name of the metric
                value: Value to add

            Returns:
                New metric value
            """
            self._metrics[metric_name] = self._metrics.get(metric_name, 0.0) + value
            return self._metrics[metric_name]

        def get_metric(self, metric_name: str) -> float:
            """Get metric value.

            Args:
                metric_name: Name of the metric

            Returns:
                Metric value or 0
            """
            return self._metrics.get(metric_name, 0.0)

        def get_all_metrics(self) -> dict[str, float]:
            """Get all metrics.

            Returns:
                Dictionary of all metrics
            """
            return dict(self._metrics)


    @ray.remote
    class WorkerPoolActor:
        """Ray actor for managing a pool of inquiry workers.

        Coordinates work distribution across multiple worker actors.
        """

        def __init__(self, num_workers: int = 4):
            """Initialize the worker pool.

            Args:
                num_workers: Number of worker actors to create
            """
            self._workers: list[ray.ObjectRef] = []
            self._num_workers = num_workers
            self._next_worker = 0
            self._active_tasks: dict[str, ray.ObjectRef] = {}
            logger.info(f"WorkerPoolActor initialized with {num_workers} workers")

        async def initialize(self) -> None:
            """Initialize worker actors."""
            from titan.ray.actors import InquiryWorkerActor
            for i in range(self._num_workers):
                worker = InquiryWorkerActor.remote()
                self._workers.append(worker)
            logger.info(f"Initialized {len(self._workers)} worker actors")

        def submit_stage(
            self,
            session_id: str,
            stage_name: str,
        ) -> ray.ObjectRef:
            """Submit a stage for execution.

            Args:
                session_id: Session ID
                stage_name: Stage to run

            Returns:
                Ray object reference for the result
            """
            # Round-robin worker selection
            worker = self._workers[self._next_worker]
            self._next_worker = (self._next_worker + 1) % len(self._workers)

            # Submit task
            task_ref = worker.run_stage.remote(session_id, stage_name)
            task_id = f"{session_id}:{stage_name}"
            self._active_tasks[task_id] = task_ref

            return task_ref

        def get_active_task_count(self) -> int:
            """Get number of active tasks.

            Returns:
                Number of active tasks
            """
            return len(self._active_tasks)

        def get_worker_count(self) -> int:
            """Get number of workers.

            Returns:
                Number of workers
            """
            return len(self._workers)


    @ray.remote
    class InquiryWorkerActor:
        """Ray actor for executing inquiry stages.

        Runs inquiry stages in isolation with its own engine instance.
        """

        def __init__(self):
            """Initialize the inquiry worker."""
            from titan.workflows.inquiry_engine import InquiryEngine
            self.engine = InquiryEngine()
            self._tasks_completed = 0
            logger.info("InquiryWorkerActor initialized")

        async def run_stage(
            self,
            session_id: str,
            stage_name: str,
        ) -> dict[str, Any]:
            """Run an inquiry stage.

            Args:
                session_id: Session ID
                stage_name: Stage to run

            Returns:
                Stage result as dictionary
            """
            try:
                result = await self.engine.run_stage(session_id, stage_name)
                self._tasks_completed += 1
                return result.to_dict()
            except Exception as e:
                logger.exception(f"Stage execution failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "stage_name": stage_name,
                }

        def get_tasks_completed(self) -> int:
            """Get number of completed tasks.

            Returns:
                Number of completed tasks
            """
            return self._tasks_completed


# Factory functions for creating actors
def create_state_manager() -> Any:
    """Create a StateManagerActor.

    Returns:
        StateManagerActor ray object reference
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray is not installed")
    return StateManagerActor.remote()


def create_worker_pool(num_workers: int = 4) -> Any:
    """Create a WorkerPoolActor.

    Args:
        num_workers: Number of workers in the pool

    Returns:
        WorkerPoolActor ray object reference
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray is not installed")
    return WorkerPoolActor.remote(num_workers)
