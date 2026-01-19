"""
Base Runtime Interface - Abstract runtime for agent execution.

Defines the contract for all runtime implementations:
- Local (Python process)
- Container (Docker/K3s)
- Serverless (OpenFaaS)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine
from uuid import uuid4

logger = logging.getLogger("titan.runtime")


class RuntimeType(str, Enum):
    """Types of runtimes available."""

    LOCAL = "local"          # Direct Python process
    DOCKER = "docker"        # Docker container
    K3S = "k3s"              # Kubernetes (K3s)
    OPENFAAS = "openfaas"    # OpenFaaS serverless


class ProcessState(str, Enum):
    """State of an agent process."""

    PENDING = "pending"        # Waiting to start
    STARTING = "starting"      # Initializing
    RUNNING = "running"        # Actively executing
    PAUSED = "paused"          # Temporarily stopped
    COMPLETED = "completed"    # Finished successfully
    FAILED = "failed"          # Terminated with error
    CANCELLED = "cancelled"    # User cancelled


@dataclass
class RuntimeConfig:
    """Configuration for a runtime."""

    type: RuntimeType
    name: str | None = None

    # Resource limits
    cpu_limit: float | None = None        # CPU cores (e.g., 0.5 = 500m)
    memory_limit: str | None = None       # Memory limit (e.g., "512Mi")
    gpu_required: bool = False

    # Networking
    network_mode: str = "bridge"
    expose_ports: list[int] = field(default_factory=list)

    # Volumes
    volumes: dict[str, str] = field(default_factory=dict)  # host:container

    # Environment
    environment: dict[str, str] = field(default_factory=dict)

    # Timeouts
    startup_timeout: float = 30.0         # Seconds to wait for startup
    execution_timeout: float = 300.0      # Max execution time
    shutdown_timeout: float = 10.0        # Time to wait for graceful shutdown

    # Docker-specific
    image: str | None = None
    dockerfile: str | None = None
    build_context: str | None = None

    # OpenFaaS-specific
    function_name: str | None = None
    gateway_url: str | None = None


@dataclass
class RuntimeConstraints:
    """Constraints that influence runtime selection."""

    # Hardware requirements
    requires_gpu: bool = False
    min_memory_mb: int = 256
    min_cpu_cores: float = 0.1

    # Scale requirements
    expected_instances: int = 1
    max_instances: int = 1
    auto_scale: bool = False

    # Execution requirements
    max_execution_time: float = 300.0     # Seconds
    needs_isolation: bool = False
    needs_persistence: bool = False

    # Cost optimization
    cost_sensitive: bool = False
    prefer_local: bool = True

    # Fault tolerance
    needs_retry: bool = False
    max_retries: int = 3


@dataclass
class AgentProcess:
    """Represents a running agent process."""

    process_id: str = field(default_factory=lambda: str(uuid4())[:8])
    agent_id: str = ""
    runtime_type: RuntimeType = RuntimeType.LOCAL
    state: ProcessState = ProcessState.PENDING

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Results
    result: Any = None
    error: str | None = None
    exit_code: int | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Container-specific
    container_id: str | None = None

    def mark_started(self) -> None:
        """Mark process as started."""
        self.state = ProcessState.RUNNING
        self.started_at = datetime.now()

    def mark_completed(self, result: Any = None) -> None:
        """Mark process as completed successfully."""
        self.state = ProcessState.COMPLETED
        self.completed_at = datetime.now()
        self.result = result
        self.exit_code = 0

    def mark_failed(self, error: str, exit_code: int = 1) -> None:
        """Mark process as failed."""
        self.state = ProcessState.FAILED
        self.completed_at = datetime.now()
        self.error = error
        self.exit_code = exit_code

    @property
    def duration_ms(self) -> int | None:
        """Get execution duration in milliseconds."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None


class Runtime(ABC):
    """
    Abstract base class for agent runtimes.

    A runtime is responsible for:
    - Starting agent processes
    - Monitoring process health
    - Collecting results
    - Cleaning up resources
    """

    type: RuntimeType

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig(type=self.type)
        self._processes: dict[str, AgentProcess] = {}
        self._initialized = False

    @property
    def name(self) -> str:
        """Get runtime name."""
        return self.config.name or self.type.value

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the runtime.

        Called once before first use. Should:
        - Validate configuration
        - Connect to services (Docker, K8s, etc.)
        - Prepare resources
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the runtime.

        Should:
        - Stop all running processes
        - Release resources
        - Disconnect from services
        """
        pass

    @abstractmethod
    async def spawn(
        self,
        agent_id: str,
        agent_spec: dict[str, Any],
        prompt: str | None = None,
    ) -> AgentProcess:
        """
        Spawn an agent in this runtime.

        Args:
            agent_id: Unique identifier for the agent
            agent_spec: Agent specification (from YAML)
            prompt: Optional initial prompt

        Returns:
            AgentProcess representing the running agent
        """
        pass

    @abstractmethod
    async def stop(self, process_id: str, force: bool = False) -> bool:
        """
        Stop a running process.

        Args:
            process_id: ID of the process to stop
            force: If True, force kill without graceful shutdown

        Returns:
            True if stopped successfully
        """
        pass

    @abstractmethod
    async def get_status(self, process_id: str) -> AgentProcess | None:
        """
        Get the current status of a process.

        Args:
            process_id: ID of the process

        Returns:
            AgentProcess with current state, or None if not found
        """
        pass

    @abstractmethod
    async def get_logs(
        self,
        process_id: str,
        tail: int = 100,
    ) -> list[str]:
        """
        Get logs from a process.

        Args:
            process_id: ID of the process
            tail: Number of lines to return

        Returns:
            List of log lines
        """
        pass

    async def health_check(self) -> dict[str, Any]:
        """
        Check runtime health.

        Returns:
            Dict with health status and details
        """
        return {
            "type": self.type.value,
            "name": self.name,
            "initialized": self._initialized,
            "active_processes": len([
                p for p in self._processes.values()
                if p.state == ProcessState.RUNNING
            ]),
        }

    def get_process(self, process_id: str) -> AgentProcess | None:
        """Get a process by ID."""
        return self._processes.get(process_id)

    def list_processes(
        self,
        state: ProcessState | None = None,
    ) -> list[AgentProcess]:
        """
        List all processes, optionally filtered by state.

        Args:
            state: Optional state filter

        Returns:
            List of matching processes
        """
        processes = list(self._processes.values())
        if state:
            processes = [p for p in processes if p.state == state]
        return processes

    async def wait_for_completion(
        self,
        process_id: str,
        timeout: float | None = None,
    ) -> AgentProcess | None:
        """
        Wait for a process to complete.

        Args:
            process_id: ID of the process
            timeout: Maximum time to wait (seconds)

        Returns:
            Completed AgentProcess, or None if timeout
        """
        timeout = timeout or self.config.execution_timeout
        start = asyncio.get_event_loop().time()

        while True:
            process = await self.get_status(process_id)
            if not process:
                return None

            if process.state in (
                ProcessState.COMPLETED,
                ProcessState.FAILED,
                ProcessState.CANCELLED,
            ):
                return process

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed >= timeout:
                logger.warning(f"Process {process_id} timed out after {elapsed:.1f}s")
                return process

            await asyncio.sleep(0.5)

    async def spawn_and_wait(
        self,
        agent_id: str,
        agent_spec: dict[str, Any],
        prompt: str | None = None,
        timeout: float | None = None,
    ) -> AgentProcess:
        """
        Spawn an agent and wait for completion.

        Convenience method combining spawn() and wait_for_completion().

        Args:
            agent_id: Unique identifier for the agent
            agent_spec: Agent specification
            prompt: Optional initial prompt
            timeout: Maximum execution time

        Returns:
            Completed AgentProcess
        """
        process = await self.spawn(agent_id, agent_spec, prompt)
        result = await self.wait_for_completion(process.process_id, timeout)
        return result or process

    def _register_process(self, process: AgentProcess) -> None:
        """Register a process for tracking."""
        self._processes[process.process_id] = process
        logger.debug(f"Registered process {process.process_id} for agent {process.agent_id}")

    def _unregister_process(self, process_id: str) -> AgentProcess | None:
        """Unregister a process."""
        return self._processes.pop(process_id, None)
