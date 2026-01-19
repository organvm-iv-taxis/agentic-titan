"""
Agentic Titan - Stress Testing Framework

Provides tools for stress testing agent swarms at scale (50-100+ agents).
Measures throughput, latency, memory usage, and identifies bottlenecks.
"""

from titan.stress.runner import (
    StressTestRunner,
    StressTestConfig,
    StressTestResult,
    AgentFactory,
)
from titan.stress.scenarios import (
    Scenario,
    SwarmBrainstormScenario,
    PipelineWorkflowScenario,
    HierarchyDelegationScenario,
    ChaosScenario,
)
from titan.stress.metrics import (
    StressMetrics,
    LatencyHistogram,
    ThroughputCounter,
)

__all__ = [
    # Runner
    "StressTestRunner",
    "StressTestConfig",
    "StressTestResult",
    "AgentFactory",
    # Scenarios
    "Scenario",
    "SwarmBrainstormScenario",
    "PipelineWorkflowScenario",
    "HierarchyDelegationScenario",
    "ChaosScenario",
    # Metrics
    "StressMetrics",
    "LatencyHistogram",
    "ThroughputCounter",
]
