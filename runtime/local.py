"""
Local Runtime - Direct Python process execution.

This is the simplest and fastest runtime, executing agents directly
in the current Python process. Best for:
- Development and testing
- Low-latency requirements
- GPU access (local GPU)
- Small-scale deployments
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import datetime
from typing import Any

from runtime.base import (
    Runtime,
    RuntimeType,
    RuntimeConfig,
    AgentProcess,
    ProcessState,
)

logger = logging.getLogger("titan.runtime.local")


class LocalRuntime(Runtime):
    """
    Local runtime executing agents as Python coroutines.

    Features:
    - Direct execution (no container overhead)
    - Full access to local resources
    - In-process logging and monitoring
    - Supports async agents natively
    """

    type = RuntimeType.LOCAL

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        super().__init__(config or RuntimeConfig(type=RuntimeType.LOCAL))
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._logs: dict[str, list[str]] = {}

    async def initialize(self) -> None:
        """Initialize local runtime (minimal setup needed)."""
        logger.info("Local runtime initialized")
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown local runtime, cancelling all tasks."""
        logger.info("Shutting down local runtime...")

        # Cancel all running tasks
        for process_id, task in list(self._tasks.items()):
            if not task.done():
                logger.debug(f"Cancelling task {process_id}")
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=self.config.shutdown_timeout)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        self._tasks.clear()
        self._processes.clear()
        self._initialized = False
        logger.info("Local runtime shutdown complete")

    async def spawn(
        self,
        agent_id: str,
        agent_spec: dict[str, Any],
        prompt: str | None = None,
    ) -> AgentProcess:
        """
        Spawn an agent as a local Python task.

        Args:
            agent_id: Unique identifier for the agent
            agent_spec: Agent specification dict
            prompt: Optional initial prompt

        Returns:
            AgentProcess with task info
        """
        # Create process record
        process = AgentProcess(
            agent_id=agent_id,
            runtime_type=RuntimeType.LOCAL,
            state=ProcessState.STARTING,
            metadata={"spec": agent_spec, "prompt": prompt},
        )
        self._register_process(process)
        self._logs[process.process_id] = []

        # Create and start the execution task
        task = asyncio.create_task(
            self._execute_agent(process, agent_spec, prompt),
            name=f"agent-{agent_id}-{process.process_id}",
        )
        self._tasks[process.process_id] = task

        logger.info(f"Spawned local agent {agent_id} as process {process.process_id}")
        return process

    async def _execute_agent(
        self,
        process: AgentProcess,
        agent_spec: dict[str, Any],
        prompt: str | None,
    ) -> None:
        """
        Execute an agent in this process.

        Args:
            process: The process record to update
            agent_spec: Agent specification
            prompt: Optional prompt
        """
        self._log(process.process_id, f"Starting agent execution")
        process.mark_started()

        try:
            # Import agent classes
            from agents.archetypes import (
                ResearcherAgent,
                CoderAgent,
                ReviewerAgent,
                OrchestratorAgent,
            )
            from hive import HiveMind

            # Map spec names to agent classes
            agent_map = {
                "researcher": ResearcherAgent,
                "coder": CoderAgent,
                "reviewer": ReviewerAgent,
                "orchestrator": OrchestratorAgent,
            }

            # Get agent class
            agent_name = agent_spec.get("metadata", {}).get("name", "").lower()
            agent_class = agent_map.get(agent_name)

            if not agent_class:
                raise ValueError(f"Unknown agent type: {agent_name}")

            self._log(process.process_id, f"Creating {agent_name} agent")

            # Initialize Hive Mind
            hive = HiveMind()
            await hive.initialize()

            try:
                # Create agent with appropriate kwargs
                kwargs: dict[str, Any] = {"hive_mind": hive}
                if agent_name == "researcher":
                    kwargs["topic"] = prompt
                elif agent_name == "coder":
                    kwargs["task_description"] = prompt
                elif agent_name == "reviewer":
                    kwargs["content"] = prompt
                elif agent_name == "orchestrator":
                    kwargs["task"] = prompt

                agent = agent_class(**kwargs)

                # Run agent
                self._log(process.process_id, f"Executing agent")
                result = await asyncio.wait_for(
                    agent.run(prompt),
                    timeout=self.config.execution_timeout,
                )

                self._log(process.process_id, f"Agent completed: {result.state}")
                process.mark_completed(result)

            finally:
                await hive.shutdown()

        except asyncio.CancelledError:
            self._log(process.process_id, "Agent cancelled")
            process.state = ProcessState.CANCELLED
            process.completed_at = datetime.now()
            raise

        except asyncio.TimeoutError:
            self._log(process.process_id, "Agent timed out")
            process.mark_failed("Execution timeout", exit_code=124)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            self._log(process.process_id, f"Agent failed: {error_msg}")
            self._log(process.process_id, traceback.format_exc())
            process.mark_failed(error_msg)

    async def stop(self, process_id: str, force: bool = False) -> bool:
        """
        Stop a running local process.

        Args:
            process_id: ID of the process
            force: If True, cancel immediately

        Returns:
            True if stopped
        """
        task = self._tasks.get(process_id)
        if not task:
            return False

        if task.done():
            return True

        self._log(process_id, f"Stopping process (force={force})")
        task.cancel()

        try:
            timeout = 0 if force else self.config.shutdown_timeout
            await asyncio.wait_for(task, timeout=timeout)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        process = self._processes.get(process_id)
        if process and process.state == ProcessState.RUNNING:
            process.state = ProcessState.CANCELLED
            process.completed_at = datetime.now()

        return True

    async def get_status(self, process_id: str) -> AgentProcess | None:
        """Get current status of a process."""
        process = self._processes.get(process_id)
        if not process:
            return None

        # Update state based on task
        task = self._tasks.get(process_id)
        if task:
            if task.done():
                # Ensure final state is captured
                if process.state == ProcessState.RUNNING:
                    try:
                        result = task.result()
                        if process.state == ProcessState.RUNNING:
                            process.mark_completed(result)
                    except asyncio.CancelledError:
                        process.state = ProcessState.CANCELLED
                        process.completed_at = datetime.now()
                    except Exception as e:
                        process.mark_failed(str(e))

        return process

    async def get_logs(self, process_id: str, tail: int = 100) -> list[str]:
        """Get logs from a process."""
        logs = self._logs.get(process_id, [])
        return logs[-tail:]

    def _log(self, process_id: str, message: str) -> None:
        """Add a log entry for a process."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{timestamp}] {message}"
        if process_id in self._logs:
            self._logs[process_id].append(entry)
        logger.debug(f"[{process_id}] {message}")

    async def health_check(self) -> dict[str, Any]:
        """Check local runtime health."""
        base = await super().health_check()
        base.update({
            "active_tasks": len([t for t in self._tasks.values() if not t.done()]),
            "total_processes": len(self._processes),
        })
        return base
