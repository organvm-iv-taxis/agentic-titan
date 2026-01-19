"""
Agentic Titan - Error Hierarchy

Provides a structured error hierarchy for the agent system, enabling
fine-grained error handling and recovery strategies.

Ported from: metasystem-core/agent_utils/errors.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    LOW = "low"           # Informational, can be ignored
    MEDIUM = "medium"     # Should be logged, may affect results
    HIGH = "high"         # Requires attention, operation may fail
    CRITICAL = "critical" # System cannot continue


class RecoveryStrategy(Enum):
    """Suggested recovery strategies for errors."""

    RETRY = "retry"               # Retry the operation
    RETRY_WITH_BACKOFF = "retry_with_backoff"  # Retry with exponential backoff
    SKIP = "skip"                 # Skip this operation
    ABORT = "abort"               # Abort the entire task
    ESCALATE = "escalate"         # Escalate to parent agent/human
    CIRCUIT_BREAK = "circuit_break"  # Open circuit breaker
    FALLBACK = "fallback"         # Use fallback behavior


@dataclass
class ErrorContext:
    """Rich context for error reporting and debugging."""

    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: str | None = None
    session_id: str | None = None
    operation: str | None = None
    input_data: dict[str, Any] | None = None
    stack_trace: str | None = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "operation": self.operation,
            "input_data": self.input_data,
            "stack_trace": self.stack_trace,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }


class TitanError(Exception):
    """
    Base exception for all Agentic Titan errors.

    Provides structured error information including severity,
    recovery strategy suggestions, and rich context.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "TITAN_ERROR",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery: RecoveryStrategy = RecoveryStrategy.ABORT,
        recoverable: bool = True,
        context: ErrorContext | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.severity = severity
        self.recovery = recovery
        self.recoverable = recoverable
        self.context = context or ErrorContext()
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "recovery_strategy": self.recovery.value,
            "recoverable": self.recoverable,
            "context": self.context.to_dict(),
            "cause": str(self.cause) if self.cause else None,
        }

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"code={self.code!r}, "
            f"message={self.message!r}, "
            f"severity={self.severity.value})"
        )


class AgentError(TitanError):
    """Errors related to agent execution."""

    def __init__(
        self,
        message: str,
        *,
        agent_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", None) or ErrorContext()
        context.agent_id = agent_id
        super().__init__(message, code="AGENT_ERROR", context=context, **kwargs)
        self.agent_id = agent_id


class AgentNotFoundError(AgentError):
    """Agent with given ID was not found."""

    def __init__(self, agent_id: str, **kwargs: Any) -> None:
        super().__init__(
            f"Agent not found: {agent_id}",
            agent_id=agent_id,
            code="AGENT_NOT_FOUND",
            severity=ErrorSeverity.HIGH,
            recovery=RecoveryStrategy.ABORT,
            recoverable=False,
            **kwargs,
        )


class AgentSpawnError(AgentError):
    """Failed to spawn an agent."""

    def __init__(self, agent_id: str, reason: str, **kwargs: Any) -> None:
        super().__init__(
            f"Failed to spawn agent {agent_id}: {reason}",
            agent_id=agent_id,
            code="AGENT_SPAWN_ERROR",
            severity=ErrorSeverity.HIGH,
            recovery=RecoveryStrategy.RETRY_WITH_BACKOFF,
            **kwargs,
        )


class AgentTimeoutError(AgentError):
    """Agent execution timed out."""

    def __init__(self, agent_id: str, timeout_ms: int, **kwargs: Any) -> None:
        super().__init__(
            f"Agent {agent_id} timed out after {timeout_ms}ms",
            agent_id=agent_id,
            code="AGENT_TIMEOUT",
            severity=ErrorSeverity.HIGH,
            recovery=RecoveryStrategy.RETRY_WITH_BACKOFF,
            **kwargs,
        )
        self.timeout_ms = timeout_ms


class HiveMindError(TitanError):
    """Errors related to the Hive Mind (shared memory/communication)."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            code="HIVE_MIND_ERROR",
            severity=ErrorSeverity.HIGH,
            recovery=RecoveryStrategy.RETRY,
            **kwargs,
        )


class MemoryError(HiveMindError):
    """Failed to read/write to shared memory."""

    def __init__(self, operation: str, key: str, **kwargs: Any) -> None:
        super().__init__(
            f"Memory {operation} failed for key: {key}",
            code="MEMORY_ERROR",
            **kwargs,
        )
        self.operation = operation
        self.key = key


class CommunicationError(HiveMindError):
    """Failed to communicate between agents."""

    def __init__(self, source: str, target: str, **kwargs: Any) -> None:
        super().__init__(
            f"Communication failed: {source} -> {target}",
            code="COMMUNICATION_ERROR",
            **kwargs,
        )
        self.source = source
        self.target = target


class TopologyError(TitanError):
    """Errors related to agent topology management."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            code="TOPOLOGY_ERROR",
            severity=ErrorSeverity.MEDIUM,
            recovery=RecoveryStrategy.FALLBACK,
            **kwargs,
        )


class InvalidTopologyError(TopologyError):
    """Requested topology is invalid or unsupported."""

    def __init__(self, topology_type: str, **kwargs: Any) -> None:
        super().__init__(
            f"Invalid topology type: {topology_type}",
            code="INVALID_TOPOLOGY",
            severity=ErrorSeverity.HIGH,
            recovery=RecoveryStrategy.ABORT,
            recoverable=False,
            **kwargs,
        )
        self.topology_type = topology_type


class LLMAdapterError(TitanError):
    """Errors related to LLM adapter operations."""

    def __init__(self, message: str, *, provider: str | None = None, **kwargs: Any) -> None:
        super().__init__(
            message,
            code="LLM_ADAPTER_ERROR",
            severity=ErrorSeverity.HIGH,
            recovery=RecoveryStrategy.RETRY_WITH_BACKOFF,
            **kwargs,
        )
        self.provider = provider


class LLMRateLimitError(LLMAdapterError):
    """LLM API rate limit exceeded."""

    def __init__(self, provider: str, retry_after: int | None = None, **kwargs: Any) -> None:
        super().__init__(
            f"Rate limit exceeded for {provider}"
            + (f", retry after {retry_after}s" if retry_after else ""),
            provider=provider,
            code="LLM_RATE_LIMIT",
            recovery=RecoveryStrategy.RETRY_WITH_BACKOFF,
            **kwargs,
        )
        self.retry_after = retry_after


class LLMContextExceededError(LLMAdapterError):
    """Context window exceeded."""

    def __init__(self, provider: str, tokens_used: int, max_tokens: int, **kwargs: Any) -> None:
        super().__init__(
            f"Context exceeded for {provider}: {tokens_used}/{max_tokens} tokens",
            provider=provider,
            code="LLM_CONTEXT_EXCEEDED",
            recovery=RecoveryStrategy.ABORT,
            **kwargs,
        )
        self.tokens_used = tokens_used
        self.max_tokens = max_tokens


class CircuitBreakerError(TitanError):
    """Circuit breaker is open, operation cannot proceed."""

    def __init__(self, service: str, **kwargs: Any) -> None:
        super().__init__(
            f"Circuit breaker open for service: {service}",
            code="CIRCUIT_BREAKER_OPEN",
            severity=ErrorSeverity.HIGH,
            recovery=RecoveryStrategy.CIRCUIT_BREAK,
            **kwargs,
        )
        self.service = service


class SpecValidationError(TitanError):
    """Agent specification validation failed."""

    def __init__(self, spec_path: str, errors: list[str], **kwargs: Any) -> None:
        super().__init__(
            f"Invalid agent spec at {spec_path}: {'; '.join(errors)}",
            code="SPEC_VALIDATION_ERROR",
            severity=ErrorSeverity.HIGH,
            recovery=RecoveryStrategy.ABORT,
            recoverable=False,
            **kwargs,
        )
        self.spec_path = spec_path
        self.validation_errors = errors
