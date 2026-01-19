"""
Runtime Selection - Choose optimal execution environment.

Runtimes:
- Local: Direct Python process (development, GPU tasks)
- Container: Docker/K3s (production, isolation)
- Serverless: OpenFaaS (burst traffic, cost optimization)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("titan.migration.runtime")


class RuntimeType(str, Enum):
    """Available runtime environments."""

    LOCAL = "local"  # Direct Python process
    CONTAINER = "container"  # Docker/K3s container
    SERVERLESS = "serverless"  # OpenFaaS function
    HYBRID = "hybrid"  # Mixed environment


@dataclass
class RuntimeConfig:
    """Configuration for a runtime."""

    type: RuntimeType
    name: str

    # Connection/deployment info
    endpoint: str = ""
    image: str = ""  # Container image
    function_name: str = ""  # Serverless function

    # Resources
    cpu_limit: float = 1.0  # CPU cores
    memory_limit_mb: int = 512
    timeout_seconds: int = 300
    gpu_required: bool = False

    # Scaling
    min_replicas: int = 1
    max_replicas: int = 10
    scale_threshold: float = 0.8  # CPU utilization trigger

    # Cost
    cost_per_hour: float = 0.0
    cost_per_invocation: float = 0.0

    # Health
    health_endpoint: str = "/health"
    health_interval_seconds: int = 30

    # Metadata
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "name": self.name,
            "endpoint": self.endpoint,
            "image": self.image,
            "function_name": self.function_name,
            "cpu_limit": self.cpu_limit,
            "memory_limit_mb": self.memory_limit_mb,
            "timeout_seconds": self.timeout_seconds,
            "gpu_required": self.gpu_required,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "cost_per_hour": self.cost_per_hour,
            "labels": self.labels,
        }


@dataclass
class RuntimeCapabilities:
    """What a runtime can do."""

    supports_gpu: bool = False
    supports_long_running: bool = False  # > 5 min
    supports_web_access: bool = True
    supports_file_system: bool = True
    max_memory_mb: int = 1024
    max_timeout_seconds: int = 300


@dataclass
class AgentRequirements:
    """What an agent needs."""

    needs_gpu: bool = False
    needs_long_running: bool = False
    needs_web_access: bool = False
    needs_file_system: bool = False
    min_memory_mb: int = 256
    expected_duration_seconds: int = 60

    # Preferences
    prefer_low_latency: bool = False
    prefer_low_cost: bool = False
    prefer_isolation: bool = False


class RuntimeSelector:
    """
    Select optimal runtime for an agent.

    Considers:
    - Agent requirements (GPU, memory, duration)
    - Runtime capabilities
    - Cost optimization
    - Current load
    """

    def __init__(self) -> None:
        self._runtimes: dict[str, RuntimeConfig] = {}
        self._capabilities: dict[str, RuntimeCapabilities] = {}

        # Default runtimes
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default runtime configurations."""
        # Local runtime
        self.register_runtime(
            RuntimeConfig(
                type=RuntimeType.LOCAL,
                name="local",
                cpu_limit=4.0,
                memory_limit_mb=8192,
                timeout_seconds=3600,
                cost_per_hour=0.0,
            ),
            RuntimeCapabilities(
                supports_gpu=True,
                supports_long_running=True,
                max_memory_mb=8192,
                max_timeout_seconds=3600,
            ),
        )

        # Container runtime
        self.register_runtime(
            RuntimeConfig(
                type=RuntimeType.CONTAINER,
                name="k3s",
                endpoint="http://localhost:6443",
                image="titan/agent:latest",
                cpu_limit=2.0,
                memory_limit_mb=2048,
                timeout_seconds=1800,
                cost_per_hour=0.05,
            ),
            RuntimeCapabilities(
                supports_gpu=False,
                supports_long_running=True,
                max_memory_mb=2048,
                max_timeout_seconds=1800,
            ),
        )

        # Serverless runtime
        self.register_runtime(
            RuntimeConfig(
                type=RuntimeType.SERVERLESS,
                name="openfaas",
                endpoint="http://localhost:8080",
                function_name="titan-agent",
                cpu_limit=1.0,
                memory_limit_mb=512,
                timeout_seconds=300,
                cost_per_invocation=0.001,
            ),
            RuntimeCapabilities(
                supports_gpu=False,
                supports_long_running=False,
                max_memory_mb=512,
                max_timeout_seconds=300,
            ),
        )

    def register_runtime(
        self,
        config: RuntimeConfig,
        capabilities: RuntimeCapabilities,
    ) -> None:
        """Register a runtime."""
        self._runtimes[config.name] = config
        self._capabilities[config.name] = capabilities
        logger.info(f"Registered runtime: {config.name} ({config.type.value})")

    def get_runtime(self, name: str) -> RuntimeConfig | None:
        """Get runtime by name."""
        return self._runtimes.get(name)

    def list_runtimes(self) -> list[RuntimeConfig]:
        """List all registered runtimes."""
        return list(self._runtimes.values())

    def select(
        self,
        requirements: AgentRequirements,
        exclude: list[str] | None = None,
    ) -> RuntimeConfig:
        """
        Select the best runtime for given requirements.

        Args:
            requirements: What the agent needs
            exclude: Runtimes to exclude from selection

        Returns:
            Best matching RuntimeConfig
        """
        exclude = exclude or []
        candidates: list[tuple[RuntimeConfig, float]] = []

        for name, config in self._runtimes.items():
            if name in exclude:
                continue

            caps = self._capabilities.get(name)
            if not caps:
                continue

            # Check hard requirements
            if requirements.needs_gpu and not caps.supports_gpu:
                continue
            if requirements.needs_long_running and not caps.supports_long_running:
                continue
            if requirements.min_memory_mb > caps.max_memory_mb:
                continue
            if requirements.expected_duration_seconds > caps.max_timeout_seconds:
                continue

            # Score the runtime
            score = self._score_runtime(requirements, config, caps)
            candidates.append((config, score))

        if not candidates:
            logger.warning("No suitable runtime found, falling back to local")
            return self._runtimes.get("local") or RuntimeConfig(
                type=RuntimeType.LOCAL,
                name="fallback",
            )

        # Select highest score
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = candidates[0][0]

        logger.info(
            f"Selected runtime: {selected.name} "
            f"(score={candidates[0][1]:.2f})"
        )
        return selected

    def _score_runtime(
        self,
        requirements: AgentRequirements,
        config: RuntimeConfig,
        caps: RuntimeCapabilities,
    ) -> float:
        """
        Score a runtime based on requirements.

        Higher score = better match.
        """
        score = 50.0  # Base score

        # Cost optimization
        if requirements.prefer_low_cost:
            if config.type == RuntimeType.LOCAL:
                score += 20
            elif config.type == RuntimeType.SERVERLESS:
                score += 10
            # Penalize expensive options
            score -= config.cost_per_hour * 100

        # Low latency preference
        if requirements.prefer_low_latency:
            if config.type == RuntimeType.LOCAL:
                score += 25
            elif config.type == RuntimeType.CONTAINER:
                score += 10
            # Serverless has cold start latency

        # Isolation preference
        if requirements.prefer_isolation:
            if config.type == RuntimeType.CONTAINER:
                score += 25
            elif config.type == RuntimeType.SERVERLESS:
                score += 20
            # Local has no isolation

        # Resource fit
        memory_ratio = requirements.min_memory_mb / caps.max_memory_mb
        if 0.3 < memory_ratio < 0.8:
            score += 10  # Good fit
        elif memory_ratio > 0.9:
            score -= 10  # Too tight

        # Duration fit
        duration_ratio = requirements.expected_duration_seconds / caps.max_timeout_seconds
        if duration_ratio < 0.5:
            score += 5  # Plenty of headroom
        elif duration_ratio > 0.8:
            score -= 5  # Cutting it close

        return score

    def select_for_burst(self, count: int) -> RuntimeConfig:
        """
        Select runtime for burst traffic (many agents).

        Prefers scalable options.
        """
        requirements = AgentRequirements(
            prefer_low_cost=True,
        )

        if count > 20:
            # Serverless for large bursts
            return self._runtimes.get("openfaas") or self.select(requirements)
        elif count > 5:
            # Containers for medium bursts
            return self._runtimes.get("k3s") or self.select(requirements)
        else:
            # Local for small counts
            return self._runtimes.get("local") or self.select(requirements)

    def select_for_gpu(self) -> RuntimeConfig:
        """Select runtime with GPU support."""
        for name, caps in self._capabilities.items():
            if caps.supports_gpu:
                return self._runtimes[name]

        logger.warning("No GPU runtime available")
        return self._runtimes.get("local") or RuntimeConfig(
            type=RuntimeType.LOCAL,
            name="fallback",
        )
