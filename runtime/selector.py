"""
Runtime Selector - Intelligent runtime selection.

Automatically chooses the best runtime based on:
- Task requirements (GPU, memory, CPU)
- Scale requirements (instances, auto-scaling)
- Cost optimization preferences
- Fault tolerance needs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from runtime.base import (
    Runtime,
    RuntimeType,
    RuntimeConfig,
    RuntimeConstraints,
    AgentProcess,
)
from runtime.local import LocalRuntime
from runtime.docker import DockerRuntime
from runtime.openfaas import OpenFaaSRuntime

logger = logging.getLogger("titan.runtime.selector")


class SelectionStrategy(str, Enum):
    """Strategy for runtime selection."""

    AUTO = "auto"              # Intelligent selection based on constraints
    PREFER_LOCAL = "local"     # Prefer local, fallback to others
    PREFER_DOCKER = "docker"   # Prefer Docker containers
    COST_OPTIMIZED = "cost"    # Minimize resource usage
    PERFORMANCE = "perf"       # Maximum performance


@dataclass
class RuntimeScore:
    """Score for a runtime option."""

    runtime_type: RuntimeType
    score: float  # 0-100
    reasons: list[str]
    available: bool = True


class RuntimeSelector:
    """
    Intelligent runtime selector.

    Analyzes constraints and task requirements to select
    the optimal runtime for agent execution.
    """

    def __init__(
        self,
        strategy: SelectionStrategy = SelectionStrategy.AUTO,
        runtimes: dict[RuntimeType, Runtime] | None = None,
    ) -> None:
        self.strategy = strategy
        self._runtimes: dict[RuntimeType, Runtime] = runtimes or {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize available runtimes."""
        logger.info("Initializing runtime selector...")

        # Create default runtimes if not provided
        if not self._runtimes:
            self._runtimes = {
                RuntimeType.LOCAL: LocalRuntime(),
                RuntimeType.DOCKER: DockerRuntime(),
            }

        # Initialize each runtime
        for runtime_type, runtime in self._runtimes.items():
            try:
                await runtime.initialize()
                logger.info(f"Runtime {runtime_type.value} initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize {runtime_type.value}: {e}")

        self._initialized = True
        logger.info("Runtime selector initialized")

    async def shutdown(self) -> None:
        """Shutdown all runtimes."""
        for runtime in self._runtimes.values():
            try:
                await runtime.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down runtime: {e}")

        self._initialized = False

    def select(
        self,
        constraints: RuntimeConstraints | None = None,
        agent_spec: dict[str, Any] | None = None,
    ) -> RuntimeType:
        """
        Select the best runtime for given constraints.

        Args:
            constraints: Runtime constraints
            agent_spec: Agent specification (optional)

        Returns:
            Selected runtime type
        """
        constraints = constraints or RuntimeConstraints()

        # Score each runtime
        scores = self._score_runtimes(constraints, agent_spec)

        # Sort by score (descending)
        scores.sort(key=lambda s: s.score, reverse=True)

        # Log selection rationale
        logger.info("Runtime selection scores:")
        for score in scores:
            status = "✓" if score.available else "✗"
            logger.info(f"  {status} {score.runtime_type.value}: {score.score:.1f} - {', '.join(score.reasons)}")

        # Return highest scoring available runtime
        for score in scores:
            if score.available:
                return score.runtime_type

        # Fallback to local
        logger.warning("No suitable runtime found, falling back to local")
        return RuntimeType.LOCAL

    def _score_runtimes(
        self,
        constraints: RuntimeConstraints,
        agent_spec: dict[str, Any] | None,
    ) -> list[RuntimeScore]:
        """Score all available runtimes."""
        scores = []

        for runtime_type in [RuntimeType.LOCAL, RuntimeType.DOCKER]:
            score = self._score_runtime(runtime_type, constraints, agent_spec)
            scores.append(score)

        return scores

    def _score_runtime(
        self,
        runtime_type: RuntimeType,
        constraints: RuntimeConstraints,
        agent_spec: dict[str, Any] | None,
    ) -> RuntimeScore:
        """Score a specific runtime."""
        score = 50.0  # Base score
        reasons: list[str] = []

        # Check availability
        runtime = self._runtimes.get(runtime_type)
        available = runtime is not None and runtime._initialized

        if runtime_type == RuntimeType.LOCAL:
            score, reasons = self._score_local(constraints, agent_spec)
        elif runtime_type == RuntimeType.DOCKER:
            score, reasons = self._score_docker(constraints, agent_spec)

        # Apply strategy modifiers
        score = self._apply_strategy(runtime_type, score, constraints)

        return RuntimeScore(
            runtime_type=runtime_type,
            score=score,
            reasons=reasons,
            available=available,
        )

    def _score_local(
        self,
        constraints: RuntimeConstraints,
        agent_spec: dict[str, Any] | None,
    ) -> tuple[float, list[str]]:
        """Score local runtime."""
        score = 60.0  # Good baseline
        reasons = []

        # Advantages of local
        if constraints.prefer_local:
            score += 20
            reasons.append("Local preferred")

        if not constraints.needs_isolation:
            score += 10
            reasons.append("No isolation needed")

        # Local is fastest for single instances
        if constraints.expected_instances <= 2:
            score += 15
            reasons.append("Low instance count")

        # GPU access (local has direct GPU)
        if constraints.requires_gpu:
            score += 10
            reasons.append("Direct GPU access")

        # Disadvantages
        if constraints.expected_instances > 5:
            score -= 20
            reasons.append("High scale reduces score")

        if constraints.needs_isolation:
            score -= 15
            reasons.append("Needs isolation")

        if constraints.auto_scale:
            score -= 10
            reasons.append("No auto-scaling")

        return score, reasons

    def _score_docker(
        self,
        constraints: RuntimeConstraints,
        agent_spec: dict[str, Any] | None,
    ) -> tuple[float, list[str]]:
        """Score Docker runtime."""
        score = 50.0  # Good baseline
        reasons = []

        # Advantages of Docker
        if constraints.needs_isolation:
            score += 25
            reasons.append("Isolation provided")

        if constraints.expected_instances > 3:
            score += 20
            reasons.append("Good for scaling")

        if constraints.needs_persistence:
            score += 10
            reasons.append("Volume support")

        # Resource limits enforceable
        if constraints.min_memory_mb > 512:
            score += 5
            reasons.append("Resource limits enforced")

        # Disadvantages
        if constraints.prefer_local:
            score -= 15
            reasons.append("Local preferred")

        if constraints.requires_gpu:
            # GPU in Docker is more complex
            score -= 5
            reasons.append("GPU overhead")

        # Check if agent spec has container config
        if agent_spec:
            runtimes = agent_spec.get("spec", {}).get("runtimes", {})
            if "container" in runtimes:
                score += 15
                reasons.append("Container config present")

        return score, reasons

    def _apply_strategy(
        self,
        runtime_type: RuntimeType,
        score: float,
        constraints: RuntimeConstraints,
    ) -> float:
        """Apply strategy modifiers to score."""
        if self.strategy == SelectionStrategy.PREFER_LOCAL:
            if runtime_type == RuntimeType.LOCAL:
                score += 30
        elif self.strategy == SelectionStrategy.PREFER_DOCKER:
            if runtime_type == RuntimeType.DOCKER:
                score += 30
        elif self.strategy == SelectionStrategy.COST_OPTIMIZED:
            # Local is cheapest
            if runtime_type == RuntimeType.LOCAL:
                score += 20
        elif self.strategy == SelectionStrategy.PERFORMANCE:
            # Local is fastest for simple tasks
            if runtime_type == RuntimeType.LOCAL and not constraints.needs_isolation:
                score += 15

        return score

    def get_runtime(self, runtime_type: RuntimeType) -> Runtime | None:
        """Get a specific runtime instance."""
        return self._runtimes.get(runtime_type)

    async def spawn(
        self,
        agent_id: str,
        agent_spec: dict[str, Any],
        prompt: str | None = None,
        constraints: RuntimeConstraints | None = None,
        runtime_type: RuntimeType | None = None,
    ) -> AgentProcess:
        """
        Spawn an agent with automatic runtime selection.

        Args:
            agent_id: Unique agent identifier
            agent_spec: Agent specification
            prompt: Optional initial prompt
            constraints: Runtime constraints (for selection)
            runtime_type: Override automatic selection

        Returns:
            AgentProcess from selected runtime
        """
        # Select runtime
        if runtime_type is None:
            runtime_type = self.select(constraints, agent_spec)

        runtime = self._runtimes.get(runtime_type)
        if not runtime:
            raise RuntimeError(f"Runtime {runtime_type.value} not available")

        logger.info(f"Spawning agent {agent_id} on {runtime_type.value}")
        return await runtime.spawn(agent_id, agent_spec, prompt)

    async def health_check(self) -> dict[str, Any]:
        """Check health of all runtimes."""
        results = {
            "strategy": self.strategy.value,
            "initialized": self._initialized,
            "runtimes": {},
        }

        for runtime_type, runtime in self._runtimes.items():
            try:
                health = await runtime.health_check()
                results["runtimes"][runtime_type.value] = health
            except Exception as e:
                results["runtimes"][runtime_type.value] = {
                    "error": str(e),
                    "available": False,
                }

        return results

    def suggest(
        self,
        constraints: RuntimeConstraints | None = None,
        agent_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Get a runtime suggestion with explanation.

        Args:
            constraints: Runtime constraints
            agent_spec: Agent specification

        Returns:
            Dict with recommendation and reasoning
        """
        constraints = constraints or RuntimeConstraints()
        scores = self._score_runtimes(constraints, agent_spec)
        scores.sort(key=lambda s: s.score, reverse=True)

        best = scores[0]

        return {
            "recommended": best.runtime_type.value,
            "score": best.score,
            "reasons": best.reasons,
            "available": best.available,
            "alternatives": [
                {
                    "type": s.runtime_type.value,
                    "score": s.score,
                    "reasons": s.reasons,
                    "available": s.available,
                }
                for s in scores[1:]
            ],
            "constraints_summary": {
                "requires_gpu": constraints.requires_gpu,
                "needs_isolation": constraints.needs_isolation,
                "expected_instances": constraints.expected_instances,
                "auto_scale": constraints.auto_scale,
                "cost_sensitive": constraints.cost_sensitive,
            },
        }
