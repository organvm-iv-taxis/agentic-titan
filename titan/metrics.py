"""
Agentic Titan - Prometheus Metrics

Provides Prometheus metrics instrumentation for the agent swarm:
- Agent lifecycle metrics
- Topology metrics
- Communication metrics
- Resource metrics

Start the metrics server with:
    titan metrics serve --port 9100
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, TypeVar

logger = logging.getLogger("titan.metrics")

# Try to import prometheus_client, fall back to no-op if not available
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Info,
        Summary,
        CollectorRegistry,
        start_http_server,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed, metrics will be no-ops")


# Type variable for decorators
F = TypeVar("F", bound=Callable[..., Any])


# ============================================================================
# Registry
# ============================================================================

if PROMETHEUS_AVAILABLE:
    # Create a custom registry to avoid conflicts
    REGISTRY = CollectorRegistry()

    # ========================================================================
    # Agent Metrics
    # ========================================================================

    AGENT_SPAWNED = Counter(
        "titan_agents_spawned_total",
        "Total number of agents spawned",
        ["archetype"],
        registry=REGISTRY,
    )

    AGENT_COMPLETED = Counter(
        "titan_agents_completed_total",
        "Total number of agents completed",
        ["archetype", "status"],
        registry=REGISTRY,
    )

    AGENT_ACTIVE = Gauge(
        "titan_agents_active",
        "Number of currently active agents",
        ["archetype"],
        registry=REGISTRY,
    )

    AGENT_DURATION = Histogram(
        "titan_agent_duration_seconds",
        "Agent execution duration in seconds",
        ["archetype"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
        registry=REGISTRY,
    )

    AGENT_TURNS = Histogram(
        "titan_agent_turns",
        "Number of turns taken by agents",
        ["archetype"],
        buckets=(1, 2, 3, 5, 10, 15, 20, 30, 50),
        registry=REGISTRY,
    )

    AGENT_ERRORS = Counter(
        "titan_agent_errors_total",
        "Total number of agent errors",
        ["archetype", "error_type"],
        registry=REGISTRY,
    )

    # ========================================================================
    # Topology Metrics
    # ========================================================================

    TOPOLOGY_CURRENT = Gauge(
        "titan_topology_current",
        "Current active topology (1 = active)",
        ["topology_type"],
        registry=REGISTRY,
    )

    TOPOLOGY_SWITCHES = Counter(
        "titan_topology_switches_total",
        "Total number of topology switches",
        ["from_type", "to_type"],
        registry=REGISTRY,
    )

    TOPOLOGY_SWITCH_DURATION = Histogram(
        "titan_topology_switch_duration_seconds",
        "Time to switch topology",
        ["from_type", "to_type"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        registry=REGISTRY,
    )

    TOPOLOGY_AGENT_COUNT = Gauge(
        "titan_topology_agents",
        "Number of agents in topology",
        ["topology_type"],
        registry=REGISTRY,
    )

    # ========================================================================
    # Communication Metrics
    # ========================================================================

    MESSAGES_SENT = Counter(
        "titan_messages_sent_total",
        "Total messages sent",
        ["type"],  # broadcast, direct, pubsub
        registry=REGISTRY,
    )

    MESSAGE_LATENCY = Histogram(
        "titan_message_latency_seconds",
        "Message delivery latency",
        ["type"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
        registry=REGISTRY,
    )

    PUBSUB_SUBSCRIBERS = Gauge(
        "titan_pubsub_subscribers",
        "Number of active subscribers",
        ["topic"],
        registry=REGISTRY,
    )

    # ========================================================================
    # Batch Metrics
    # ========================================================================

    BATCH_SUBMITTED = Counter(
        "titan_batch_submitted_total",
        "Total batch jobs submitted",
        ["workflow"],
        registry=REGISTRY,
    )

    BATCH_STARTED = Counter(
        "titan_batch_started_total",
        "Total batch jobs started",
        ["workflow"],
        registry=REGISTRY,
    )

    BATCH_COMPLETED = Counter(
        "titan_batch_completed_total",
        "Total batch jobs completed",
        ["workflow", "status"],
        registry=REGISTRY,
    )

    BATCH_ACTIVE = Gauge(
        "titan_batch_active",
        "Number of active batch jobs",
        registry=REGISTRY,
    )

    BATCH_SESSION_STARTED = Counter(
        "titan_batch_session_started_total",
        "Total batch sessions started",
        registry=REGISTRY,
    )

    BATCH_SESSION_COMPLETED = Counter(
        "titan_batch_session_completed_total",
        "Total batch sessions completed",
        ["status"],
        registry=REGISTRY,
    )

    BATCH_TOKENS = Counter(
        "titan_batch_tokens_total",
        "Total tokens used by batch processing",
        registry=REGISTRY,
    )

    BATCH_COST = Counter(
        "titan_batch_cost_usd_total",
        "Total cost of batch processing in USD",
        registry=REGISTRY,
    )

    BATCH_DURATION = Histogram(
        "titan_batch_duration_seconds",
        "Batch job duration in seconds",
        ["workflow"],
        buckets=(60.0, 120.0, 300.0, 600.0, 1200.0, 1800.0, 3600.0, 7200.0),
        registry=REGISTRY,
    )

    # ========================================================================
    # Memory Metrics
    # ========================================================================

    MEMORY_OPERATIONS = Counter(
        "titan_memory_operations_total",
        "Total memory operations",
        ["operation"],  # remember, recall, forget
        registry=REGISTRY,
    )

    MEMORY_LATENCY = Histogram(
        "titan_memory_latency_seconds",
        "Memory operation latency",
        ["operation"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        registry=REGISTRY,
    )

    MEMORY_SIZE = Gauge(
        "titan_memory_entries",
        "Number of entries in memory stores",
        ["store"],  # short_term, long_term, working
        registry=REGISTRY,
    )

    # ========================================================================
    # LLM Metrics
    # ========================================================================

    LLM_REQUESTS = Counter(
        "titan_llm_requests_total",
        "Total LLM requests",
        ["provider", "model"],
        registry=REGISTRY,
    )

    LLM_ERRORS = Counter(
        "titan_llm_errors_total",
        "Total LLM errors",
        ["provider", "error_type"],
        registry=REGISTRY,
    )

    LLM_LATENCY = Histogram(
        "titan_llm_latency_seconds",
        "LLM request latency",
        ["provider", "model"],
        buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0),
        registry=REGISTRY,
    )

    LLM_TOKENS = Counter(
        "titan_llm_tokens_total",
        "Total LLM tokens",
        ["provider", "direction"],  # input, output
        registry=REGISTRY,
    )

    # ========================================================================
    # Runtime Metrics
    # ========================================================================

    RUNTIME_ACTIVE = Gauge(
        "titan_runtime_active",
        "Active runtimes (1 = active)",
        ["runtime_type"],  # local, docker, openfaas
        registry=REGISTRY,
    )

    RUNTIME_SPAWNS = Counter(
        "titan_runtime_spawns_total",
        "Total runtime spawns",
        ["runtime_type"],
        registry=REGISTRY,
    )

    RUNTIME_ERRORS = Counter(
        "titan_runtime_errors_total",
        "Total runtime errors",
        ["runtime_type", "error_type"],
        registry=REGISTRY,
    )

    # ========================================================================
    # Learning Metrics
    # ========================================================================

    EPISODES_RECORDED = Counter(
        "titan_episodes_recorded_total",
        "Total episodes recorded",
        ["topology"],
        registry=REGISTRY,
    )

    EPISODE_SCORE = Histogram(
        "titan_episode_score",
        "Episode outcome scores",
        ["topology"],
        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        registry=REGISTRY,
    )

    LEARNING_RECOMMENDATIONS = Counter(
        "titan_learning_recommendations_total",
        "Total learning-based recommendations",
        ["recommended_topology"],
        registry=REGISTRY,
    )

    LEARNING_SIGNALS = Counter(
        "titan_learning_signals_total",
        "Total learning signals recorded",
        ["signal_type"],
        registry=REGISTRY,
    )

    LEARNING_REWARD = Histogram(
        "titan_learning_reward",
        "Reward values from learning pipeline",
        buckets=(-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0),
        registry=REGISTRY,
    )

    LEARNING_CONFIDENCE = Histogram(
        "titan_learning_confidence",
        "Confidence values from learning pipeline",
        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        registry=REGISTRY,
    )

    RLHF_SAMPLES = Counter(
        "titan_rlhf_samples_total",
        "Total RLHF samples collected",
        ["feedback_type"],
        registry=REGISTRY,
    )

    # ========================================================================
    # Assembly Theory Metrics
    # ========================================================================

    ASSEMBLY_INDEX = Gauge(
        "titan_assembly_index",
        "Current assembly index (minimum steps to construct)",
        ["session_id", "decision_type"],
        registry=REGISTRY,
    )

    TOTAL_ASSEMBLY = Gauge(
        "titan_total_assembly",
        "Total assembly metric A = Σ(e^ai × (ni - 1)) / NT",
        ["session_id"],
        registry=REGISTRY,
    )

    SELECTION_SIGNAL = Gauge(
        "titan_selection_signal",
        "Selection signal strength (0=ABSENT, 1=WEAK, 2=MODERATE, 3=STRONG)",
        ["session_id"],
        registry=REGISTRY,
    )

    ASSEMBLY_PATHS = Counter(
        "titan_assembly_paths_total",
        "Total assembly paths recorded",
        ["session_id", "path_type"],
        registry=REGISTRY,
    )

    ASSEMBLY_STEPS = Counter(
        "titan_assembly_steps_total",
        "Total assembly steps recorded",
        ["session_id", "step_type"],
        registry=REGISTRY,
    )

    ASSEMBLY_PATH_LENGTH = Histogram(
        "titan_assembly_path_length",
        "Distribution of assembly path lengths",
        ["path_type"],
        buckets=(1, 2, 3, 5, 8, 13, 21, 34, 55),
        registry=REGISTRY,
    )

    # ========================================================================
    # Stigmergy Metrics
    # ========================================================================

    PHEROMONE_DEPOSITS = Counter(
        "titan_pheromone_deposits_total",
        "Total pheromone deposits",
        ["trace_type"],
        registry=REGISTRY,
    )

    PHEROMONE_INTENSITY = Gauge(
        "titan_pheromone_intensity",
        "Current pheromone intensity at location",
        ["location", "trace_type"],
        registry=REGISTRY,
    )

    TRAIL_FOLLOWS = Counter(
        "titan_trail_follows_total",
        "Total trail follows",
        ["trace_type"],
        registry=REGISTRY,
    )

    # ========================================================================
    # Neighborhood Metrics
    # ========================================================================

    NEIGHBOR_INTERACTIONS = Counter(
        "titan_neighbor_interactions_total",
        "Total topological neighbor interactions",
        ["interaction_type"],
        registry=REGISTRY,
    )

    INFORMATION_PROPAGATION = Counter(
        "titan_information_propagation_total",
        "Total information propagation events",
        registry=REGISTRY,
    )

    NEIGHBOR_COUNT = Gauge(
        "titan_neighbor_count",
        "Current number of topological neighbors per agent",
        ["agent_id"],
        registry=REGISTRY,
    )

    # ========================================================================
    # Extended Topology Metrics
    # ========================================================================

    TERRITORY_COUNT = Gauge(
        "titan_territory_count",
        "Number of active territories",
        registry=REGISTRY,
    )

    DETERRITORIALIZATIONS = Counter(
        "titan_deterritorializations_total",
        "Total deterritorialization events",
        ["type"],  # external, internal, relative, absolute
        registry=REGISTRY,
    )

    TERRITORIALIZATIONS = Counter(
        "titan_territorializations_total",
        "Total territorialization events",
        ["type"],  # boundary, coding, overcoding
        registry=REGISTRY,
    )

    LINES_OF_FLIGHT = Counter(
        "titan_lines_of_flight_total",
        "Total lines of flight initiated",
        registry=REGISTRY,
    )

    ASSEMBLY_STATE = Gauge(
        "titan_assembly_state",
        "Current assembly state (0=FORMING, 1=STABLE, 2=UNSTABLE, 3=DISSOLVING, 4=TRANSFORMING)",
        registry=REGISTRY,
    )

    # ========================================================================
    # System Info
    # ========================================================================

    TITAN_INFO = Info(
        "titan",
        "Titan system information",
        registry=REGISTRY,
    )


# ============================================================================
# Instrumentation Helpers
# ============================================================================

class MetricsCollector:
    """
    Centralized metrics collector for Titan.

    Usage:
        metrics = MetricsCollector()
        metrics.agent_spawned("researcher")
        with metrics.track_agent_duration("coder"):
            # ... agent work ...
    """

    def __init__(self) -> None:
        self._enabled = PROMETHEUS_AVAILABLE

    @property
    def enabled(self) -> bool:
        return self._enabled

    def disable(self) -> None:
        """Disable metrics collection."""
        self._enabled = False

    def enable(self) -> None:
        """Enable metrics collection."""
        self._enabled = PROMETHEUS_AVAILABLE

    # ========================================================================
    # Agent Metrics
    # ========================================================================

    def agent_spawned(self, archetype: str) -> None:
        """Record agent spawn."""
        if not self._enabled:
            return
        AGENT_SPAWNED.labels(archetype=archetype).inc()
        AGENT_ACTIVE.labels(archetype=archetype).inc()

    def agent_completed(self, archetype: str, status: str, duration_seconds: float, turns: int) -> None:
        """Record agent completion."""
        if not self._enabled:
            return
        AGENT_COMPLETED.labels(archetype=archetype, status=status).inc()
        AGENT_ACTIVE.labels(archetype=archetype).dec()
        AGENT_DURATION.labels(archetype=archetype).observe(duration_seconds)
        AGENT_TURNS.labels(archetype=archetype).observe(turns)

    def agent_error(self, archetype: str, error_type: str) -> None:
        """Record agent error."""
        if not self._enabled:
            return
        AGENT_ERRORS.labels(archetype=archetype, error_type=error_type).inc()

    @contextmanager
    def track_agent_duration(self, archetype: str) -> Generator[None, None, None]:
        """Context manager to track agent duration."""
        if not self._enabled:
            yield
            return

        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            AGENT_DURATION.labels(archetype=archetype).observe(duration)

    # ========================================================================
    # Batch Metrics
    # ========================================================================

    def batch_submitted(self, workflow: str) -> None:
        """Record batch submission."""
        if not self._enabled:
            return
        BATCH_SUBMITTED.labels(workflow=workflow).inc()

    def batch_started(self, workflow: str) -> None:
        """Record batch started."""
        if not self._enabled:
            return
        BATCH_STARTED.labels(workflow=workflow).inc()
        BATCH_ACTIVE.inc()

    def batch_completed(
        self,
        workflow: str,
        status: str,
        duration_seconds: float,
        tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record batch completion."""
        if not self._enabled:
            return
        BATCH_COMPLETED.labels(workflow=workflow, status=status).inc()
        BATCH_ACTIVE.dec()
        BATCH_DURATION.labels(workflow=workflow).observe(duration_seconds)
        if tokens > 0:
            BATCH_TOKENS.inc(tokens)
        if cost_usd > 0:
            BATCH_COST.inc(cost_usd)

    def batch_session_started(self) -> None:
        """Record batch session started."""
        if not self._enabled:
            return
        BATCH_SESSION_STARTED.inc()

    def batch_session_completed(self, status: str, tokens: int = 0, cost_usd: float = 0.0) -> None:
        """Record batch session completion."""
        if not self._enabled:
            return
        BATCH_SESSION_COMPLETED.labels(status=status).inc()
        if tokens > 0:
            BATCH_TOKENS.inc(tokens)
        if cost_usd > 0:
            BATCH_COST.inc(cost_usd)

    # ========================================================================
    # Topology Metrics
    # ========================================================================

    def set_topology(self, topology_type: str, agent_count: int) -> None:
        """Set current topology."""
        if not self._enabled:
            return
        # Clear all topology gauges (including extended types)
        all_topologies = [
            "swarm", "hierarchy", "pipeline", "mesh", "ring", "star",
            "rhizomatic", "arborescent", "territorialized", "deterritorialized",
        ]
        for t in all_topologies:
            TOPOLOGY_CURRENT.labels(topology_type=t).set(0)
        # Set current
        TOPOLOGY_CURRENT.labels(topology_type=topology_type).set(1)
        TOPOLOGY_AGENT_COUNT.labels(topology_type=topology_type).set(agent_count)

    def topology_switch(self, from_type: str, to_type: str, duration_seconds: float) -> None:
        """Record topology switch."""
        if not self._enabled:
            return
        TOPOLOGY_SWITCHES.labels(from_type=from_type, to_type=to_type).inc()
        TOPOLOGY_SWITCH_DURATION.labels(from_type=from_type, to_type=to_type).observe(duration_seconds)

    @contextmanager
    def track_topology_switch(self, from_type: str, to_type: str) -> Generator[None, None, None]:
        """Context manager to track topology switch duration."""
        if not self._enabled:
            yield
            return

        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.topology_switch(from_type, to_type, duration)

    # ========================================================================
    # Communication Metrics
    # ========================================================================

    def message_sent(self, msg_type: str, latency_seconds: float | None = None) -> None:
        """Record message sent."""
        if not self._enabled:
            return
        MESSAGES_SENT.labels(type=msg_type).inc()
        if latency_seconds is not None:
            MESSAGE_LATENCY.labels(type=msg_type).observe(latency_seconds)

    def set_subscribers(self, topic: str, count: int) -> None:
        """Set subscriber count for a topic."""
        if not self._enabled:
            return
        PUBSUB_SUBSCRIBERS.labels(topic=topic).set(count)

    # ========================================================================
    # Memory Metrics
    # ========================================================================

    def memory_operation(self, operation: str, latency_seconds: float) -> None:
        """Record memory operation."""
        if not self._enabled:
            return
        MEMORY_OPERATIONS.labels(operation=operation).inc()
        MEMORY_LATENCY.labels(operation=operation).observe(latency_seconds)

    def set_memory_size(self, store: str, count: int) -> None:
        """Set memory store size."""
        if not self._enabled:
            return
        MEMORY_SIZE.labels(store=store).set(count)

    @contextmanager
    def track_memory_operation(self, operation: str) -> Generator[None, None, None]:
        """Context manager to track memory operation."""
        if not self._enabled:
            yield
            return

        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.memory_operation(operation, duration)

    # ========================================================================
    # LLM Metrics
    # ========================================================================

    def llm_request(
        self,
        provider: str,
        model: str,
        latency_seconds: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record LLM request."""
        if not self._enabled:
            return
        LLM_REQUESTS.labels(provider=provider, model=model).inc()
        LLM_LATENCY.labels(provider=provider, model=model).observe(latency_seconds)
        if input_tokens > 0:
            LLM_TOKENS.labels(provider=provider, direction="input").inc(input_tokens)
        if output_tokens > 0:
            LLM_TOKENS.labels(provider=provider, direction="output").inc(output_tokens)

    def llm_error(self, provider: str, error_type: str) -> None:
        """Record LLM error."""
        if not self._enabled:
            return
        LLM_ERRORS.labels(provider=provider, error_type=error_type).inc()

    @contextmanager
    def track_llm_request(self, provider: str, model: str) -> Generator[None, None, None]:
        """Context manager to track LLM request."""
        if not self._enabled:
            yield
            return

        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            LLM_REQUESTS.labels(provider=provider, model=model).inc()
            LLM_LATENCY.labels(provider=provider, model=model).observe(duration)

    # ========================================================================
    # Runtime Metrics
    # ========================================================================

    def runtime_spawn(self, runtime_type: str) -> None:
        """Record runtime spawn."""
        if not self._enabled:
            return
        RUNTIME_SPAWNS.labels(runtime_type=runtime_type).inc()
        RUNTIME_ACTIVE.labels(runtime_type=runtime_type).inc()

    def runtime_stopped(self, runtime_type: str) -> None:
        """Record runtime stop."""
        if not self._enabled:
            return
        RUNTIME_ACTIVE.labels(runtime_type=runtime_type).dec()

    def runtime_error(self, runtime_type: str, error_type: str) -> None:
        """Record runtime error."""
        if not self._enabled:
            return
        RUNTIME_ERRORS.labels(runtime_type=runtime_type, error_type=error_type).inc()

    # ========================================================================
    # Learning Metrics
    # ========================================================================

    def episode_recorded(self, topology: str, score: float) -> None:
        """Record learning episode."""
        if not self._enabled:
            return
        EPISODES_RECORDED.labels(topology=topology).inc()
        EPISODE_SCORE.labels(topology=topology).observe(score)

    def learning_recommendation(self, recommended_topology: str) -> None:
        """Record learning-based recommendation."""
        if not self._enabled:
            return
        LEARNING_RECOMMENDATIONS.labels(recommended_topology=recommended_topology).inc()

    def learning_signal_recorded(self, reward: float, confidence: float) -> None:
        """Record learning signal from pipeline."""
        if not self._enabled:
            return
        LEARNING_REWARD.observe(reward)
        LEARNING_CONFIDENCE.observe(confidence)
        LEARNING_SIGNALS.labels(signal_type="reward").inc()

    def rlhf_sample_collected(self, feedback_type: str) -> None:
        """Record RLHF sample collection."""
        if not self._enabled:
            return
        RLHF_SAMPLES.labels(feedback_type=feedback_type).inc()

    # ========================================================================
    # Assembly Theory Metrics
    # ========================================================================

    def set_assembly_index(
        self, session_id: str, decision_type: str, index: int
    ) -> None:
        """Set the current assembly index for a session."""
        if not self._enabled:
            return
        ASSEMBLY_INDEX.labels(session_id=session_id, decision_type=decision_type).set(
            index
        )

    def set_total_assembly(self, session_id: str, value: float) -> None:
        """Set the total assembly metric A."""
        if not self._enabled:
            return
        TOTAL_ASSEMBLY.labels(session_id=session_id).set(value)

    def set_selection_signal(self, session_id: str, signal: str) -> None:
        """Set selection signal strength (ABSENT=0, WEAK=1, MODERATE=2, STRONG=3)."""
        if not self._enabled:
            return
        signal_map = {"ABSENT": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3}
        value = signal_map.get(signal.upper(), 0)
        SELECTION_SIGNAL.labels(session_id=session_id).set(value)

    def assembly_path_recorded(
        self, session_id: str, path_type: str, length: int
    ) -> None:
        """Record an assembly path."""
        if not self._enabled:
            return
        ASSEMBLY_PATHS.labels(session_id=session_id, path_type=path_type).inc()
        ASSEMBLY_PATH_LENGTH.labels(path_type=path_type).observe(length)

    def assembly_step_recorded(self, session_id: str, step_type: str) -> None:
        """Record an assembly step."""
        if not self._enabled:
            return
        ASSEMBLY_STEPS.labels(session_id=session_id, step_type=step_type).inc()

    # ========================================================================
    # Stigmergy Metrics
    # ========================================================================

    def pheromone_deposited(self, trace_type: str, location: str, intensity: float) -> None:
        """Record pheromone deposit."""
        if not self._enabled:
            return
        PHEROMONE_DEPOSITS.labels(trace_type=trace_type).inc()
        PHEROMONE_INTENSITY.labels(location=location, trace_type=trace_type).set(intensity)

    def trail_followed(self, trace_type: str) -> None:
        """Record trail follow."""
        if not self._enabled:
            return
        TRAIL_FOLLOWS.labels(trace_type=trace_type).inc()

    # ========================================================================
    # Neighborhood Metrics
    # ========================================================================

    def neighbor_interaction(self, interaction_type: str) -> None:
        """Record neighbor interaction."""
        if not self._enabled:
            return
        NEIGHBOR_INTERACTIONS.labels(interaction_type=interaction_type).inc()

    def information_propagated(self) -> None:
        """Record information propagation."""
        if not self._enabled:
            return
        INFORMATION_PROPAGATION.inc()

    def set_neighbor_count(self, agent_id: str, count: int) -> None:
        """Set neighbor count for an agent."""
        if not self._enabled:
            return
        NEIGHBOR_COUNT.labels(agent_id=agent_id).set(count)

    # ========================================================================
    # Extended Topology Metrics
    # ========================================================================

    def set_territory_count(self, count: int) -> None:
        """Set number of active territories."""
        if not self._enabled:
            return
        TERRITORY_COUNT.set(count)

    def deterritorialization(self, dtype: str) -> None:
        """Record deterritorialization event."""
        if not self._enabled:
            return
        DETERRITORIALIZATIONS.labels(type=dtype).inc()

    def territorialization(self, ttype: str) -> None:
        """Record territorialization event."""
        if not self._enabled:
            return
        TERRITORIALIZATIONS.labels(type=ttype).inc()

    def line_of_flight(self) -> None:
        """Record line of flight."""
        if not self._enabled:
            return
        LINES_OF_FLIGHT.inc()

    def set_assembly_state(self, state: str) -> None:
        """Set assembly state (FORMING=0, STABLE=1, UNSTABLE=2, DISSOLVING=3, TRANSFORMING=4)."""
        if not self._enabled:
            return
        state_map = {
            "FORMING": 0,
            "STABLE": 1,
            "UNSTABLE": 2,
            "DISSOLVING": 3,
            "TRANSFORMING": 4,
        }
        value = state_map.get(state.upper(), 0)
        ASSEMBLY_STATE.set(value)

    # ========================================================================
    # System Info
    # ========================================================================

    def set_info(self, version: str, **labels: str) -> None:
        """Set system info."""
        if not self._enabled:
            return
        TITAN_INFO.info({"version": version, **labels})


# Global metrics collector instance
_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


# ============================================================================
# Server Functions
# ============================================================================

def start_metrics_server(port: int = 9100, host: str = "0.0.0.0") -> None:
    """
    Start Prometheus metrics HTTP server.

    Args:
        port: Port to listen on
        host: Host to bind to
    """
    if not PROMETHEUS_AVAILABLE:
        logger.error("prometheus_client not installed, cannot start metrics server")
        return

    start_http_server(port, addr=host, registry=REGISTRY)
    logger.info(f"Metrics server started on http://{host}:{port}/metrics")


def get_metrics_text() -> str:
    """
    Get metrics in Prometheus text format.

    Returns:
        Metrics as string
    """
    if not PROMETHEUS_AVAILABLE:
        return "# prometheus_client not installed"

    return generate_latest(REGISTRY).decode("utf-8")


def get_content_type() -> str:
    """Get Prometheus content type."""
    if not PROMETHEUS_AVAILABLE:
        return "text/plain"
    return CONTENT_TYPE_LATEST


# ============================================================================
# Decorator
# ============================================================================

def instrument(
    archetype: str | None = None,
    track_duration: bool = True,
    track_errors: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to instrument functions with metrics.

    Args:
        archetype: Agent archetype name
        track_duration: Whether to track duration
        track_errors: Whether to track errors

    Usage:
        @instrument(archetype="researcher")
        async def my_function():
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            metrics = get_metrics()
            arch = archetype or func.__name__

            if track_duration:
                start = time.time()

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                if track_errors:
                    metrics.agent_error(arch, type(e).__name__)
                raise
            finally:
                if track_duration and metrics.enabled:
                    duration = time.time() - start
                    AGENT_DURATION.labels(archetype=arch).observe(duration)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            metrics = get_metrics()
            arch = archetype or func.__name__

            if track_duration:
                start = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                if track_errors:
                    metrics.agent_error(arch, type(e).__name__)
                raise
            finally:
                if track_duration and metrics.enabled:
                    duration = time.time() - start
                    AGENT_DURATION.labels(archetype=arch).observe(duration)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator
