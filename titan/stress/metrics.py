"""
Stress Testing Metrics Collection

Collects and aggregates metrics during stress tests:
- Latency distributions (p50, p95, p99)
- Throughput measurements
- Memory/CPU usage
- Error rates
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyHistogram:
    """
    Tracks latency measurements and computes percentiles.

    Usage:
        hist = LatencyHistogram()
        hist.observe(100)  # 100ms
        hist.observe(150)  # 150ms
        print(hist.p50, hist.p95, hist.p99)
    """

    values: list[float] = field(default_factory=list)
    _sorted_cache: list[float] | None = field(default=None, repr=False)

    def observe(self, value_ms: float) -> None:
        """Record a latency observation in milliseconds."""
        self.values.append(value_ms)
        self._sorted_cache = None

    def _sorted(self) -> list[float]:
        """Get sorted values (cached)."""
        if self._sorted_cache is None:
            self._sorted_cache = sorted(self.values)
        return self._sorted_cache

    @property
    def count(self) -> int:
        """Number of observations."""
        return len(self.values)

    @property
    def mean(self) -> float:
        """Mean latency."""
        return statistics.mean(self.values) if self.values else 0.0

    @property
    def min(self) -> float:
        """Minimum latency."""
        return min(self.values) if self.values else 0.0

    @property
    def max(self) -> float:
        """Maximum latency."""
        return max(self.values) if self.values else 0.0

    @property
    def stddev(self) -> float:
        """Standard deviation."""
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    def percentile(self, p: float) -> float:
        """
        Compute percentile (0-100).

        Args:
            p: Percentile (e.g., 50 for median, 95 for p95)

        Returns:
            Value at percentile
        """
        if not self.values:
            return 0.0

        sorted_vals = self._sorted()
        k = (len(sorted_vals) - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f < len(sorted_vals) - 1 else f

        if f == c:
            return sorted_vals[f]

        return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

    @property
    def p50(self) -> float:
        """50th percentile (median)."""
        return self.percentile(50)

    @property
    def p90(self) -> float:
        """90th percentile."""
        return self.percentile(90)

    @property
    def p95(self) -> float:
        """95th percentile."""
        return self.percentile(95)

    @property
    def p99(self) -> float:
        """99th percentile."""
        return self.percentile(99)

    def to_dict(self) -> dict[str, float]:
        """Export metrics as dictionary."""
        return {
            "count": self.count,
            "mean": round(self.mean, 2),
            "min": round(self.min, 2),
            "max": round(self.max, 2),
            "stddev": round(self.stddev, 2),
            "p50": round(self.p50, 2),
            "p90": round(self.p90, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
        }


@dataclass
class ThroughputCounter:
    """
    Tracks throughput (operations per second).

    Usage:
        counter = ThroughputCounter()
        counter.start()
        # ... do work ...
        counter.increment(10)  # 10 operations
        counter.stop()
        print(counter.ops_per_second)
    """

    total_ops: int = 0
    start_time: float | None = None
    end_time: float | None = None
    _window_ops: list[tuple[float, int]] = field(default_factory=list)

    def start(self) -> None:
        """Start the counter."""
        self.start_time = time.time()
        self.total_ops = 0
        self._window_ops = []

    def stop(self) -> None:
        """Stop the counter."""
        self.end_time = time.time()

    def increment(self, count: int = 1) -> None:
        """Record operations."""
        self.total_ops += count
        self._window_ops.append((time.time(), count))

        # Keep only last 60 seconds for windowed rate
        cutoff = time.time() - 60
        self._window_ops = [(t, c) for t, c in self._window_ops if t > cutoff]

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def ops_per_second(self) -> float:
        """Overall operations per second."""
        elapsed = self.elapsed_seconds
        return self.total_ops / elapsed if elapsed > 0 else 0.0

    @property
    def current_rate(self) -> float:
        """Current rate (ops/sec over last 60 seconds)."""
        if not self._window_ops:
            return 0.0

        window_start = self._window_ops[0][0]
        window_end = self._window_ops[-1][0]
        window_duration = window_end - window_start

        if window_duration <= 0:
            return 0.0

        window_total = sum(c for _, c in self._window_ops)
        return window_total / window_duration

    def to_dict(self) -> dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "total_ops": self.total_ops,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "ops_per_second": round(self.ops_per_second, 2),
            "current_rate": round(self.current_rate, 2),
        }


@dataclass
class StressMetrics:
    """
    Aggregated metrics for a stress test run.

    Collects:
    - Agent lifecycle latencies
    - Task execution latencies
    - Throughput
    - Error counts
    - Memory/resource usage
    """

    # Latency metrics
    agent_spawn_latency: LatencyHistogram = field(default_factory=LatencyHistogram)
    agent_init_latency: LatencyHistogram = field(default_factory=LatencyHistogram)
    agent_work_latency: LatencyHistogram = field(default_factory=LatencyHistogram)
    agent_shutdown_latency: LatencyHistogram = field(default_factory=LatencyHistogram)
    total_agent_latency: LatencyHistogram = field(default_factory=LatencyHistogram)

    # Topology metrics
    topology_switch_latency: LatencyHistogram = field(default_factory=LatencyHistogram)
    agent_migration_latency: LatencyHistogram = field(default_factory=LatencyHistogram)

    # Communication metrics
    message_latency: LatencyHistogram = field(default_factory=LatencyHistogram)
    broadcast_latency: LatencyHistogram = field(default_factory=LatencyHistogram)

    # Throughput
    agent_throughput: ThroughputCounter = field(default_factory=ThroughputCounter)
    message_throughput: ThroughputCounter = field(default_factory=ThroughputCounter)

    # Error tracking
    error_count: int = 0
    timeout_count: int = 0
    cancelled_count: int = 0

    # Resource tracking
    peak_memory_mb: float = 0.0
    peak_agents: int = 0

    def record_error(self) -> None:
        """Record an error occurrence."""
        self.error_count += 1

    def record_timeout(self) -> None:
        """Record a timeout occurrence."""
        self.timeout_count += 1

    def record_cancelled(self) -> None:
        """Record a cancellation."""
        self.cancelled_count += 1

    def update_peak_agents(self, count: int) -> None:
        """Update peak concurrent agents."""
        if count > self.peak_agents:
            self.peak_agents = count

    def update_peak_memory(self, memory_mb: float) -> None:
        """Update peak memory usage."""
        if memory_mb > self.peak_memory_mb:
            self.peak_memory_mb = memory_mb

    @property
    def success_count(self) -> int:
        """Number of successful agent executions."""
        return self.total_agent_latency.count - self.error_count - self.timeout_count

    @property
    def success_rate(self) -> float:
        """Success rate (0-1)."""
        total = self.total_agent_latency.count
        return self.success_count / total if total > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Error rate (0-1)."""
        total = self.total_agent_latency.count
        return self.error_count / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Export all metrics as dictionary."""
        return {
            "latency": {
                "agent_spawn": self.agent_spawn_latency.to_dict(),
                "agent_init": self.agent_init_latency.to_dict(),
                "agent_work": self.agent_work_latency.to_dict(),
                "agent_shutdown": self.agent_shutdown_latency.to_dict(),
                "total_agent": self.total_agent_latency.to_dict(),
                "topology_switch": self.topology_switch_latency.to_dict(),
                "agent_migration": self.agent_migration_latency.to_dict(),
                "message": self.message_latency.to_dict(),
                "broadcast": self.broadcast_latency.to_dict(),
            },
            "throughput": {
                "agents": self.agent_throughput.to_dict(),
                "messages": self.message_throughput.to_dict(),
            },
            "errors": {
                "error_count": self.error_count,
                "timeout_count": self.timeout_count,
                "cancelled_count": self.cancelled_count,
                "success_count": self.success_count,
                "success_rate": round(self.success_rate, 4),
                "error_rate": round(self.error_rate, 4),
            },
            "resources": {
                "peak_memory_mb": round(self.peak_memory_mb, 2),
                "peak_agents": self.peak_agents,
            },
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=== Stress Test Results ===",
            "",
            f"Agents Completed: {self.total_agent_latency.count}",
            f"Success Rate: {self.success_rate:.1%}",
            f"Error Rate: {self.error_rate:.1%}",
            "",
            "Latency (ms):",
            f"  Total Agent: p50={self.total_agent_latency.p50:.0f}, p95={self.total_agent_latency.p95:.0f}, p99={self.total_agent_latency.p99:.0f}",
            f"  Spawn: p50={self.agent_spawn_latency.p50:.0f}, p95={self.agent_spawn_latency.p95:.0f}",
            f"  Work: p50={self.agent_work_latency.p50:.0f}, p95={self.agent_work_latency.p95:.0f}",
            "",
            "Throughput:",
            f"  Agents/sec: {self.agent_throughput.ops_per_second:.2f}",
            f"  Messages/sec: {self.message_throughput.ops_per_second:.2f}",
            "",
            "Resources:",
            f"  Peak Memory: {self.peak_memory_mb:.1f} MB",
            f"  Peak Concurrent Agents: {self.peak_agents}",
            "",
            "Errors:",
            f"  Errors: {self.error_count}",
            f"  Timeouts: {self.timeout_count}",
            f"  Cancelled: {self.cancelled_count}",
        ]
        return "\n".join(lines)
