"""
Langfuse Integration - LLM tracing and observability.

Provides:
- LLM call tracing
- Cost tracking
- Latency monitoring
- Prompt versioning

Reference: vendor/cli/terminal-ai Langfuse integration
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator

logger = logging.getLogger("titan.observability.langfuse")


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class TraceSpan:
    """A span within a trace."""

    id: str
    name: str
    trace_id: str
    parent_id: str | None = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    # LLM-specific
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Cache metrics
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    # Cost
    cost_usd: float = 0.0

    # Content
    input_content: str | None = None
    output_content: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "running"  # running, success, error
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() * 1000
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": self.cost_usd,
            "status": self.status,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class Trace:
    """A complete trace of an operation."""

    id: str
    name: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    # Spans
    spans: list[TraceSpan] = field(default_factory=list)

    # Aggregated metrics
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    # Context
    user_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    status: str = "running"

    @property
    def duration_ms(self) -> float:
        """Get total duration in milliseconds."""
        if self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() * 1000
        return 0.0

    def add_span(self, span: TraceSpan) -> None:
        """Add a span to the trace."""
        self.spans.append(span)
        self.total_tokens += span.total_tokens
        self.total_cost_usd += span.cost_usd

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "status": self.status,
            "tags": self.tags,
            "metadata": self.metadata,
            "spans": [s.to_dict() for s in self.spans],
        }


# ============================================================================
# Cost Calculator
# ============================================================================


# Pricing per 1M tokens (as of 2024)
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku": {"input": 0.8, "output": 4.0},
    # OpenAI
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    # Default fallback
    "default": {"input": 1.0, "output": 3.0},
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """
    Calculate cost for an LLM call.

    Args:
        model: Model identifier
        input_tokens: Input token count
        output_tokens: Output token count
        cache_read_tokens: Cached input tokens (90% discount)
        cache_write_tokens: Cache write tokens (25% premium)

    Returns:
        Cost in USD
    """
    # Find matching pricing
    pricing = MODEL_PRICING.get("default")
    for model_key, model_pricing in MODEL_PRICING.items():
        if model_key in model.lower():
            pricing = model_pricing
            break

    if not pricing:
        return 0.0

    # Calculate base cost
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    # Cache adjustments (Anthropic pricing)
    cache_read_cost = (cache_read_tokens / 1_000_000) * pricing["input"] * 0.1  # 90% discount
    cache_write_cost = (cache_write_tokens / 1_000_000) * pricing["input"] * 1.25  # 25% premium

    return input_cost + output_cost + cache_read_cost + cache_write_cost


# ============================================================================
# Langfuse Tracer
# ============================================================================


class LangfuseTracer:
    """
    LLM tracing with Langfuse-compatible output.

    Can operate in two modes:
    1. Local-only: Stores traces in memory for local analysis
    2. Langfuse: Sends traces to Langfuse server (requires langfuse package)

    Example:
        tracer = LangfuseTracer()

        with tracer.trace("agent_run") as trace:
            with tracer.span("llm_call", model="claude-3-sonnet") as span:
                # Make LLM call
                span.input_content = prompt
                span.output_content = response
                span.input_tokens = 100
                span.output_tokens = 50
    """

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str = "https://cloud.langfuse.com",
        enabled: bool = True,
        local_only: bool = False,
    ) -> None:
        """
        Initialize the tracer.

        Args:
            public_key: Langfuse public key
            secret_key: Langfuse secret key
            host: Langfuse host URL
            enabled: Enable tracing
            local_only: Store traces locally only (don't send to Langfuse)
        """
        self.enabled = enabled
        self.local_only = local_only
        self.host = host

        # Local trace storage
        self._traces: dict[str, Trace] = {}
        self._current_trace_id: str | None = None
        self._current_span_id: str | None = None

        # Langfuse client (if available)
        self._langfuse = None
        if not local_only and public_key and secret_key:
            try:
                from langfuse import Langfuse
                self._langfuse = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                logger.info("Langfuse client initialized")
            except ImportError:
                logger.warning("langfuse package not installed, using local-only mode")
                self.local_only = True
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse: {e}")
                self.local_only = True

    @contextmanager
    def trace(
        self,
        name: str,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[Trace, None, None]:
        """
        Create a new trace context.

        Args:
            name: Trace name
            user_id: User identifier
            session_id: Session identifier
            agent_id: Agent identifier
            tags: Trace tags
            metadata: Additional metadata

        Yields:
            Trace object
        """
        if not self.enabled:
            # Return dummy trace
            yield Trace(id="disabled", name=name)
            return

        trace_id = f"trace-{uuid.uuid4().hex[:12]}"
        trace = Trace(
            id=trace_id,
            name=name,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            tags=tags or [],
            metadata=metadata or {},
        )

        self._traces[trace_id] = trace
        previous_trace = self._current_trace_id
        self._current_trace_id = trace_id

        # Create Langfuse trace if available
        langfuse_trace = None
        if self._langfuse:
            try:
                langfuse_trace = self._langfuse.trace(
                    id=trace_id,
                    name=name,
                    user_id=user_id,
                    session_id=session_id,
                    tags=tags,
                    metadata=metadata,
                )
            except Exception as e:
                logger.warning(f"Failed to create Langfuse trace: {e}")

        try:
            yield trace
            trace.status = "success"
        except Exception as e:
            trace.status = "error"
            trace.metadata["error"] = str(e)
            raise
        finally:
            trace.end_time = datetime.now()
            self._current_trace_id = previous_trace

            # Update Langfuse
            if langfuse_trace:
                try:
                    langfuse_trace.update(
                        output=trace.metadata,
                        status_message=trace.status,
                    )
                except Exception as e:
                    logger.warning(f"Failed to update Langfuse trace: {e}")

            logger.debug(
                f"Trace {trace_id} completed: {trace.duration_ms:.1f}ms, "
                f"{trace.total_tokens} tokens, ${trace.total_cost_usd:.4f}"
            )

    @contextmanager
    def span(
        self,
        name: str,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[TraceSpan, None, None]:
        """
        Create a span within the current trace.

        Args:
            name: Span name
            model: Model identifier for LLM calls
            metadata: Additional metadata

        Yields:
            TraceSpan object
        """
        if not self.enabled or not self._current_trace_id:
            # Return dummy span
            yield TraceSpan(id="disabled", name=name, trace_id="disabled")
            return

        span_id = f"span-{uuid.uuid4().hex[:12]}"
        span = TraceSpan(
            id=span_id,
            name=name,
            trace_id=self._current_trace_id,
            parent_id=self._current_span_id,
            model=model,
            metadata=metadata or {},
        )

        previous_span = self._current_span_id
        self._current_span_id = span_id

        # Create Langfuse generation if available
        langfuse_gen = None
        if self._langfuse and model:
            try:
                langfuse_gen = self._langfuse.generation(
                    trace_id=self._current_trace_id,
                    id=span_id,
                    name=name,
                    model=model,
                    metadata=metadata,
                )
            except Exception as e:
                logger.warning(f"Failed to create Langfuse generation: {e}")

        try:
            yield span
            span.status = "success"
        except Exception as e:
            span.status = "error"
            span.error = str(e)
            raise
        finally:
            span.end_time = datetime.now()
            self._current_span_id = previous_span

            # Calculate cost
            if span.model:
                span.total_tokens = span.input_tokens + span.output_tokens
                span.cost_usd = calculate_cost(
                    span.model,
                    span.input_tokens,
                    span.output_tokens,
                    span.cache_read_tokens,
                    span.cache_write_tokens,
                )

            # Add to trace
            trace = self._traces.get(span.trace_id)
            if trace:
                trace.add_span(span)

            # Update Langfuse
            if langfuse_gen:
                try:
                    langfuse_gen.end(
                        output=span.output_content,
                        usage={
                            "input": span.input_tokens,
                            "output": span.output_tokens,
                            "total": span.total_tokens,
                        },
                        status_message=span.status,
                    )
                except Exception as e:
                    logger.warning(f"Failed to update Langfuse generation: {e}")

            logger.debug(
                f"Span {span_id} completed: {span.duration_ms:.1f}ms, "
                f"{span.total_tokens} tokens"
            )

    def get_trace(self, trace_id: str) -> Trace | None:
        """Get a trace by ID."""
        return self._traces.get(trace_id)

    def get_traces(
        self,
        limit: int = 100,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> list[Trace]:
        """
        Get recent traces.

        Args:
            limit: Maximum traces to return
            user_id: Filter by user
            session_id: Filter by session

        Returns:
            List of traces
        """
        traces = list(self._traces.values())

        # Filter
        if user_id:
            traces = [t for t in traces if t.user_id == user_id]
        if session_id:
            traces = [t for t in traces if t.session_id == session_id]

        # Sort by start time (newest first)
        traces.sort(key=lambda t: t.start_time, reverse=True)

        return traces[:limit]

    def get_metrics(self) -> dict[str, Any]:
        """Get aggregated metrics."""
        traces = list(self._traces.values())

        total_tokens = sum(t.total_tokens for t in traces)
        total_cost = sum(t.total_cost_usd for t in traces)
        total_duration = sum(t.duration_ms for t in traces)

        return {
            "trace_count": len(traces),
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "total_duration_ms": total_duration,
            "avg_tokens_per_trace": total_tokens / len(traces) if traces else 0,
            "avg_cost_per_trace": total_cost / len(traces) if traces else 0,
            "avg_duration_ms": total_duration / len(traces) if traces else 0,
        }

    def flush(self) -> None:
        """Flush pending traces to Langfuse."""
        if self._langfuse:
            try:
                self._langfuse.flush()
            except Exception as e:
                logger.warning(f"Failed to flush Langfuse: {e}")

    def shutdown(self) -> None:
        """Shutdown the tracer."""
        self.flush()
        if self._langfuse:
            try:
                self._langfuse.shutdown()
            except Exception as e:
                logger.warning(f"Failed to shutdown Langfuse: {e}")


# ============================================================================
# Convenience Functions
# ============================================================================


# Global tracer instance
_tracer: LangfuseTracer | None = None


def create_langfuse_tracer(
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str = "https://cloud.langfuse.com",
    enabled: bool = True,
    local_only: bool = False,
) -> LangfuseTracer:
    """
    Create and set the global Langfuse tracer.

    Args:
        public_key: Langfuse public key (or set LANGFUSE_PUBLIC_KEY env var)
        secret_key: Langfuse secret key (or set LANGFUSE_SECRET_KEY env var)
        host: Langfuse host URL
        enabled: Enable tracing
        local_only: Store traces locally only

    Returns:
        LangfuseTracer instance
    """
    import os

    global _tracer

    _tracer = LangfuseTracer(
        public_key=public_key or os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=secret_key or os.environ.get("LANGFUSE_SECRET_KEY"),
        host=host or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        enabled=enabled,
        local_only=local_only,
    )

    return _tracer


def get_tracer() -> LangfuseTracer:
    """Get the global tracer, creating one if needed."""
    global _tracer
    if _tracer is None:
        _tracer = LangfuseTracer(local_only=True)
    return _tracer


@contextmanager
def trace_llm_call(
    name: str,
    model: str,
    input_content: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Generator[TraceSpan, None, None]:
    """
    Convenience function to trace an LLM call.

    Creates both a trace and span for a single LLM call.

    Args:
        name: Call name
        model: Model identifier
        input_content: Input prompt
        user_id: User identifier
        session_id: Session identifier
        agent_id: Agent identifier
        metadata: Additional metadata

    Yields:
        TraceSpan for the LLM call

    Example:
        with trace_llm_call("chat", "claude-3-sonnet", prompt) as span:
            response = llm.generate(prompt)
            span.output_content = response
            span.input_tokens = 100
            span.output_tokens = 50
    """
    tracer = get_tracer()

    with tracer.trace(
        name=f"{name}_trace",
        user_id=user_id,
        session_id=session_id,
        agent_id=agent_id,
        metadata=metadata,
    ):
        with tracer.span(name=name, model=model, metadata=metadata) as span:
            span.input_content = input_content
            yield span
