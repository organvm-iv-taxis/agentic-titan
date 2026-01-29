"""Ray Serve deployments for Titan.

Provides Ray Serve deployments for inquiry and batch processing
as an alternative to Celery for distributed compute.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from titan.ray import RAY_AVAILABLE

if RAY_AVAILABLE:
    import ray
    from ray import serve

from titan.ray.config import get_ray_config

if TYPE_CHECKING:
    from titan.workflows.inquiry_engine import InquiryEngine, StageResult
    from titan.batch.orchestrator import BatchOrchestrator, BatchJob

logger = logging.getLogger("titan.ray.serve")


if RAY_AVAILABLE:
    @serve.deployment(
        name="inquiry-deployment",
        route_prefix="/ray/inquiry",
    )
    class InquiryDeployment:
        """Ray Serve deployment for inquiry execution.

        Handles inquiry stage execution with automatic scaling
        and fault tolerance.
        """

        def __init__(self):
            """Initialize the inquiry deployment."""
            from titan.workflows.inquiry_engine import get_inquiry_engine
            self.engine = get_inquiry_engine()
            logger.info("InquiryDeployment initialized")

        async def run_stage(
            self,
            session_id: str,
            stage_name: str,
        ) -> dict[str, Any]:
            """Run a single inquiry stage.

            Args:
                session_id: Inquiry session ID
                stage_name: Name of stage to run

            Returns:
                Stage result as dictionary
            """
            try:
                result = await self.engine.run_stage(session_id, stage_name)
                return result.to_dict()
            except Exception as e:
                logger.exception(f"Stage execution failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "stage_name": stage_name,
                }

        async def run_full_workflow(
            self,
            session_id: str,
        ) -> dict[str, Any]:
            """Run the full inquiry workflow.

            Args:
                session_id: Inquiry session ID

            Returns:
                Workflow result as dictionary
            """
            try:
                results = await self.engine.run_full_workflow(session_id)
                return {
                    "success": True,
                    "results": [r.to_dict() for r in results],
                    "total_stages": len(results),
                }
            except Exception as e:
                logger.exception(f"Workflow execution failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                }

        async def get_session(self, session_id: str) -> dict[str, Any] | None:
            """Get session status.

            Args:
                session_id: Inquiry session ID

            Returns:
                Session data or None if not found
            """
            session = self.engine.get_session(session_id)
            return session.to_dict() if session else None

        async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
            """Handle HTTP requests."""
            action = request.get("action", "run_stage")

            if action == "run_stage":
                return await self.run_stage(
                    request["session_id"],
                    request["stage_name"],
                )
            elif action == "run_workflow":
                return await self.run_full_workflow(request["session_id"])
            elif action == "get_session":
                return await self.get_session(request["session_id"]) or {}
            else:
                return {"error": f"Unknown action: {action}"}


    @serve.deployment(
        name="batch-deployment",
        route_prefix="/ray/batch",
    )
    class BatchDeployment:
        """Ray Serve deployment for batch processing.

        Handles batch job orchestration with automatic scaling.
        """

        def __init__(self):
            """Initialize the batch deployment."""
            from titan.batch.orchestrator import get_batch_orchestrator
            self.orchestrator = get_batch_orchestrator()
            self.orchestrator.enable_celery = False  # Disable Celery when using Ray
            logger.info("BatchDeployment initialized")

        async def start_batch(self, batch_id: str) -> dict[str, Any]:
            """Start a batch job.

            Args:
                batch_id: Batch job ID

            Returns:
                Batch status as dictionary
            """
            try:
                batch = await self.orchestrator.start_batch(batch_id)
                return batch.to_dict()
            except Exception as e:
                logger.exception(f"Batch start failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                }

        async def get_batch(self, batch_id: str) -> dict[str, Any] | None:
            """Get batch status.

            Args:
                batch_id: Batch job ID

            Returns:
                Batch data or None if not found
            """
            batch = self.orchestrator.get_batch(batch_id)
            return batch.to_dict() if batch else None

        async def cancel_batch(self, batch_id: str) -> dict[str, Any]:
            """Cancel a batch job.

            Args:
                batch_id: Batch job ID

            Returns:
                Cancellation result
            """
            try:
                batch = await self.orchestrator.cancel_batch(batch_id)
                return batch.to_dict()
            except Exception as e:
                logger.exception(f"Batch cancellation failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                }

        async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
            """Handle HTTP requests."""
            action = request.get("action", "get_batch")

            if action == "start_batch":
                return await self.start_batch(request["batch_id"])
            elif action == "get_batch":
                return await self.get_batch(request["batch_id"]) or {}
            elif action == "cancel_batch":
                return await self.cancel_batch(request["batch_id"])
            else:
                return {"error": f"Unknown action: {action}"}


class RayBackend:
    """High-level interface for Ray Serve deployments.

    Provides a unified interface for running inquiries and batches
    via Ray Serve, handling deployment initialization and cleanup.
    """

    def __init__(self):
        """Initialize the Ray backend."""
        if not RAY_AVAILABLE:
            raise ImportError("Ray is not installed")

        self._initialized = False
        self._inquiry_handle = None
        self._batch_handle = None

    async def initialize(self) -> None:
        """Initialize Ray and deploy services."""
        if self._initialized:
            return

        config = get_ray_config()

        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(
                address=config.address,
                namespace=config.namespace,
            )

        # Get deployment configuration
        deployment_config = config.to_deployment_config()

        # Apply configuration to deployments
        InquiryDeployment.options(**deployment_config)
        BatchDeployment.options(**deployment_config)

        # Deploy
        serve.run(InquiryDeployment.bind(), name="inquiry", route_prefix="/ray/inquiry")
        serve.run(BatchDeployment.bind(), name="batch", route_prefix="/ray/batch")

        # Get handles
        self._inquiry_handle = serve.get_deployment_handle("inquiry")
        self._batch_handle = serve.get_deployment_handle("batch")

        self._initialized = True
        logger.info("Ray backend initialized")

    async def shutdown(self) -> None:
        """Shutdown Ray Serve deployments."""
        if not self._initialized:
            return

        serve.shutdown()
        self._initialized = False
        self._inquiry_handle = None
        self._batch_handle = None
        logger.info("Ray backend shutdown")

    async def run_inquiry_stage(
        self,
        session_id: str,
        stage_name: str,
    ) -> dict[str, Any]:
        """Run an inquiry stage via Ray Serve.

        Args:
            session_id: Inquiry session ID
            stage_name: Stage to run

        Returns:
            Stage result
        """
        if not self._initialized:
            await self.initialize()

        return await self._inquiry_handle.run_stage.remote(session_id, stage_name)

    async def run_inquiry_workflow(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """Run a full inquiry workflow via Ray Serve.

        Args:
            session_id: Inquiry session ID

        Returns:
            Workflow results
        """
        if not self._initialized:
            await self.initialize()

        return await self._inquiry_handle.run_full_workflow.remote(session_id)

    async def start_batch(self, batch_id: str) -> dict[str, Any]:
        """Start a batch job via Ray Serve.

        Args:
            batch_id: Batch job ID

        Returns:
            Batch status
        """
        if not self._initialized:
            await self.initialize()

        return await self._batch_handle.start_batch.remote(batch_id)

    async def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        """Get batch status via Ray Serve.

        Args:
            batch_id: Batch job ID

        Returns:
            Batch status or None
        """
        if not self._initialized:
            await self.initialize()

        return await self._batch_handle.get_batch.remote(batch_id)
