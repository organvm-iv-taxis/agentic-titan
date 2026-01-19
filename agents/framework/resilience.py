"""
Agentic Titan - Resilience Patterns

Provides fault tolerance mechanisms for agents:
- Circuit Breaker: Prevent cascading failures
- Retry with backoff: Automatic retry with exponential backoff
- Bulkhead: Isolate failures between components

Ported from: metasystem-core/agents/resilience.py
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

from agents.framework.errors import CircuitBreakerError, TitanError

logger = logging.getLogger("titan.resilience")

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failures exceeded threshold, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5       # Failures before opening
    success_threshold: int = 2       # Successes to close from half-open
    timeout_seconds: float = 30.0    # Time before trying half-open
    half_open_max_calls: int = 3     # Max calls in half-open state


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "state_changes": self.state_changes,
            "failure_rate": self.failed_calls / max(1, self.total_calls),
        }


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.

    States:
    - CLOSED: Normal operation. Failures are counted.
    - OPEN: Circuit is tripped. All calls fail immediately.
    - HALF_OPEN: Testing recovery. Limited calls allowed.

    Usage:
        breaker = CircuitBreaker("external-api")

        @breaker
        async def call_api():
            ...

        # Or explicitly:
        async with breaker.call():
            await call_api()
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0

        self._stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()

        self._on_state_change: list[Callable[[CircuitState, CircuitState], None]] = []

        logger.info(f"Circuit breaker '{name}' created")

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Circuit breaker statistics."""
        return self._stats

    def on_state_change(
        self, handler: Callable[[CircuitState, CircuitState], None]
    ) -> None:
        """Register state change handler."""
        self._on_state_change.append(handler)

    async def _set_state(self, new_state: CircuitState) -> None:
        """Set state with notification."""
        old_state = self._state
        if old_state != new_state:
            self._state = new_state
            self._stats.state_changes += 1
            logger.info(
                f"Circuit '{self.name}' state change: "
                f"{old_state.value} -> {new_state.value}"
            )
            for handler in self._on_state_change:
                try:
                    handler(old_state, new_state)
                except Exception as e:
                    logger.warning(f"State change handler error: {e}")

    async def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on circuit state."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if timeout has passed
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.config.timeout_seconds:
                        await self._set_state(CircuitState.HALF_OPEN)
                        self._half_open_calls = 0
                        return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited calls in half-open state
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    async def _record_success(self) -> None:
        """Record successful call."""
        async with self._lock:
            self._stats.successful_calls += 1
            self._stats.last_success_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    await self._set_state(CircuitState.CLOSED)
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    async def _record_failure(self, error: Exception) -> None:
        """Record failed call."""
        async with self._lock:
            self._stats.failed_calls += 1
            self._last_failure_time = time.time()
            self._stats.last_failure_time = self._last_failure_time

            if self._state == CircuitState.HALF_OPEN:
                # Immediately open circuit on failure in half-open
                await self._set_state(CircuitState.OPEN)
                self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    await self._set_state(CircuitState.OPEN)

    async def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to call (sync or async)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        self._stats.total_calls += 1

        if not await self._should_allow_request():
            self._stats.rejected_calls += 1
            raise CircuitBreakerError(self.name)

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure(e)
            raise

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to wrap a function with circuit breaker.

        Usage:
            @breaker
            async def my_function():
                ...
        """

        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await self.call(func, *args, **kwargs)

        return wrapper  # type: ignore

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        logger.info(f"Circuit '{self.name}' reset")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay_ms: int = 1000
    max_delay_ms: int = 30000
    backoff_multiplier: float = 2.0
    retryable_errors: list[type[Exception]] = field(
        default_factory=lambda: [ConnectionError, TimeoutError]
    )


async def retry_with_backoff(
    func: Callable[..., T],
    config: RetryConfig | None = None,
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Execute a function with automatic retry and exponential backoff.

    Args:
        func: Function to call (sync or async)
        config: Retry configuration
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Function result

    Raises:
        Last exception if all retries exhausted
    """
    config = config or RetryConfig()
    delay_ms = config.initial_delay_ms
    last_error: Exception | None = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except tuple(config.retryable_errors) as e:
            last_error = e
            if attempt == config.max_attempts:
                logger.error(
                    f"All {config.max_attempts} retry attempts exhausted: {e}"
                )
                raise

            logger.warning(
                f"Attempt {attempt}/{config.max_attempts} failed: {e}. "
                f"Retrying in {delay_ms}ms..."
            )
            await asyncio.sleep(delay_ms / 1000)
            delay_ms = min(
                int(delay_ms * config.backoff_multiplier),
                config.max_delay_ms,
            )
        except Exception:
            # Non-retryable error
            raise

    # Should never reach here, but satisfy type checker
    if last_error:
        raise last_error
    raise TitanError("Unexpected retry loop exit")


class Bulkhead:
    """
    Bulkhead pattern for isolating concurrent operations.

    Limits the number of concurrent calls to prevent resource exhaustion.

    Usage:
        bulkhead = Bulkhead("database", max_concurrent=10)

        async with bulkhead.acquire():
            await db.query(...)
    """

    def __init__(self, name: str, max_concurrent: int = 10) -> None:
        self.name = name
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_calls = 0
        self._rejected_calls = 0

    @property
    def active_calls(self) -> int:
        """Number of currently active calls."""
        return self._active_calls

    @property
    def available_slots(self) -> int:
        """Number of available slots."""
        return self.max_concurrent - self._active_calls

    async def acquire(self) -> asyncio.Semaphore:
        """
        Acquire a slot in the bulkhead.

        Usage:
            async with bulkhead.acquire():
                ...
        """
        await self._semaphore.acquire()
        self._active_calls += 1
        return self._semaphore

    def release(self) -> None:
        """Release a slot in the bulkhead."""
        self._active_calls -= 1
        self._semaphore.release()

    async def __aenter__(self) -> Bulkhead:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()
