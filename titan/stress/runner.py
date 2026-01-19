"""
Stress Test Runner

Orchestrates stress tests:
- Spawns agents according to scenario
- Collects metrics
- Reports results
- Handles graceful shutdown
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from titan.stress.metrics import StressMetrics
from titan.stress.scenarios import Scenario, ScenarioPhase, get_scenario

if TYPE_CHECKING:
    from agents.framework.base_agent import BaseAgent

logger = logging.getLogger("titan.stress")


# Type alias for agent factory
AgentFactory = Callable[[int], Coroutine[Any, Any, "BaseAgent"]]


@dataclass
class StressTestConfig:
    """Configuration for a stress test run."""

    scenario_name: str = "swarm"
    target_agents: int = 50
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    warmup_seconds: int = 5

    # Concurrency control
    max_concurrent: int = 20
    batch_size: int = 5
    batch_delay_ms: int = 100

    # Failure injection
    failure_rate: float = 0.0
    timeout_rate: float = 0.0
    topology_switch_interval: int = 0

    # Output
    output_dir: str = ".stress-results"
    save_results: bool = True
    verbose: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "target_agents": self.target_agents,
            "duration_seconds": self.duration_seconds,
            "ramp_up_seconds": self.ramp_up_seconds,
            "warmup_seconds": self.warmup_seconds,
            "max_concurrent": self.max_concurrent,
            "batch_size": self.batch_size,
            "batch_delay_ms": self.batch_delay_ms,
            "failure_rate": self.failure_rate,
            "timeout_rate": self.timeout_rate,
            "topology_switch_interval": self.topology_switch_interval,
        }


@dataclass
class StressTestResult:
    """Results from a stress test run."""

    config: StressTestConfig
    metrics: StressMetrics
    scenario_name: str
    start_time: datetime
    end_time: datetime
    success: bool
    error: str | None = None

    # Detailed results
    agent_results: list[dict[str, Any]] = field(default_factory=list)
    topology_switches: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Total test duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Export results as dictionary."""
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "scenario_name": self.scenario_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error": self.error,
            "agent_results": self.agent_results,
            "topology_switches": self.topology_switches,
            "events": self.events,
        }

    def save(self, path: Path | str | None = None) -> Path:
        """Save results to JSON file."""
        if path is None:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            path = output_dir / f"stress_{self.scenario_name}_{timestamp}.json"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return path


class StressTestRunner:
    """
    Runs stress tests against the Titan agent swarm.

    Usage:
        runner = StressTestRunner(config)
        result = await runner.run()
        print(result.metrics.summary())
    """

    def __init__(
        self,
        config: StressTestConfig | None = None,
        hive_mind: Any | None = None,
        topology_engine: Any | None = None,
    ) -> None:
        """
        Initialize stress test runner.

        Args:
            config: Test configuration
            hive_mind: Shared HiveMind instance
            topology_engine: Topology engine for switching
        """
        self.config = config or StressTestConfig()
        self.hive_mind = hive_mind
        self.topology_engine = topology_engine

        self.metrics = StressMetrics()
        self.scenario: Scenario | None = None

        # State
        self._running = False
        self._cancelled = False
        self._active_agents: dict[str, "BaseAgent"] = {}
        self._agent_tasks: dict[str, asyncio.Task[Any]] = {}
        self._semaphore: asyncio.Semaphore | None = None

        # Memory tracking
        self._memory_tracker_task: asyncio.Task[Any] | None = None

    async def run(self) -> StressTestResult:
        """
        Execute the stress test.

        Returns:
            StressTestResult with metrics and details
        """
        start_time = datetime.now()
        self._running = True
        self._cancelled = False
        error: str | None = None

        try:
            # Initialize scenario
            self.scenario = get_scenario(
                self.config.scenario_name,
                target_agents=self.config.target_agents,
                duration_seconds=self.config.duration_seconds,
                ramp_up_seconds=self.config.ramp_up_seconds,
                warmup_seconds=self.config.warmup_seconds,
                failure_rate=self.config.failure_rate,
                timeout_rate=self.config.timeout_rate,
                topology_switch_interval=self.config.topology_switch_interval,
            )

            # Setup concurrency control
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

            # Start metrics collection
            self.metrics.agent_throughput.start()
            self.metrics.message_throughput.start()

            # Start memory tracking
            self._memory_tracker_task = asyncio.create_task(self._track_memory())

            # Setup signal handlers
            self._setup_signal_handlers()

            # Run test phases
            await self._run_warmup()
            await self._run_stress()
            await self._run_cooldown()

        except asyncio.CancelledError:
            logger.info("Stress test cancelled")
            self._cancelled = True
            error = "Test cancelled"

        except Exception as e:
            logger.exception("Stress test failed")
            error = str(e)

        finally:
            self._running = False

            # Stop metrics
            self.metrics.agent_throughput.stop()
            self.metrics.message_throughput.stop()

            # Cancel memory tracker
            if self._memory_tracker_task:
                self._memory_tracker_task.cancel()
                try:
                    await self._memory_tracker_task
                except asyncio.CancelledError:
                    pass

            # Cleanup remaining agents
            await self._cleanup_agents()

        end_time = datetime.now()

        result = StressTestResult(
            config=self.config,
            metrics=self.metrics,
            scenario_name=self.config.scenario_name,
            start_time=start_time,
            end_time=end_time,
            success=error is None,
            error=error,
        )

        if self.config.save_results:
            path = result.save()
            logger.info(f"Results saved to {path}")

        return result

    async def _run_warmup(self) -> None:
        """Run warmup phase."""
        if not self.scenario:
            return

        self.scenario.phase = ScenarioPhase.WARMUP
        logger.info(f"Warmup phase: {self.config.warmup_seconds}s")

        # Spawn initial agents
        warmup_agents = min(5, self.config.target_agents)
        await self._spawn_agents_batch(0, warmup_agents)

        # Wait for warmup
        await asyncio.sleep(self.config.warmup_seconds)

    async def _run_stress(self) -> None:
        """Run main stress phase."""
        if not self.scenario:
            return

        self.scenario.phase = ScenarioPhase.STRESS
        logger.info(f"Stress phase: {self.config.duration_seconds}s, {self.config.target_agents} agents")

        # Calculate ramp-up schedule
        remaining_agents = self.config.target_agents - len(self._active_agents)
        if remaining_agents > 0 and self.config.ramp_up_seconds > 0:
            agents_per_second = remaining_agents / self.config.ramp_up_seconds
            batch_interval = self.config.batch_size / agents_per_second if agents_per_second > 0 else 1.0

            # Spawn agents during ramp-up
            spawned = len(self._active_agents)
            ramp_start = time.time()

            while spawned < self.config.target_agents and not self._cancelled:
                elapsed = time.time() - ramp_start
                if elapsed >= self.config.ramp_up_seconds:
                    break

                batch_end = min(spawned + self.config.batch_size, self.config.target_agents)
                await self._spawn_agents_batch(spawned, batch_end)
                spawned = batch_end

                await asyncio.sleep(batch_interval)

        # Main stress duration
        stress_end = time.time() + self.config.duration_seconds - self.config.ramp_up_seconds

        # Topology switching task
        topology_task = None
        if self.config.topology_switch_interval > 0:
            topology_task = asyncio.create_task(self._topology_switcher())

        try:
            while time.time() < stress_end and not self._cancelled:
                # Update peak agents
                self.metrics.update_peak_agents(len(self._active_agents))

                # Brief sleep to allow async tasks
                await asyncio.sleep(0.5)

                # Log progress
                if self.config.verbose:
                    logger.info(
                        f"Active: {len(self._active_agents)}, "
                        f"Completed: {self.metrics.total_agent_latency.count}, "
                        f"Errors: {self.metrics.error_count}"
                    )

        finally:
            if topology_task:
                topology_task.cancel()
                try:
                    await topology_task
                except asyncio.CancelledError:
                    pass

    async def _run_cooldown(self) -> None:
        """Run cooldown phase - wait for remaining agents."""
        if not self.scenario:
            return

        self.scenario.phase = ScenarioPhase.COOLDOWN
        logger.info("Cooldown phase: waiting for remaining agents")

        # Wait for all agent tasks to complete (with timeout)
        if self._agent_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._agent_tasks.values(), return_exceptions=True),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Cooldown timeout, cancelling remaining agents")

    async def _spawn_agents_batch(self, start_index: int, end_index: int) -> None:
        """Spawn a batch of agents."""
        if not self.scenario:
            return

        tasks = []
        for i in range(start_index, end_index):
            if self._cancelled:
                break
            tasks.append(self._spawn_and_run_agent(i))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _spawn_and_run_agent(self, index: int) -> None:
        """Spawn an agent and run it."""
        if not self.scenario or not self._semaphore:
            return

        spawn_start = time.time()

        try:
            # Spawn agent
            agent = await self.scenario.spawn_agent(index)

            # Inject HiveMind and topology engine
            if self.hive_mind:
                agent._hive_mind = self.hive_mind
            if self.topology_engine:
                agent._topology_engine = self.topology_engine

            spawn_time = (time.time() - spawn_start) * 1000
            self.metrics.agent_spawn_latency.observe(spawn_time)

            # Track active agent
            self._active_agents[agent.agent_id] = agent
            self.scenario.active_agents.append(agent.agent_id)

            # Create task for agent execution
            task = asyncio.create_task(self._run_agent(agent))
            self._agent_tasks[agent.agent_id] = task

        except Exception as e:
            logger.error(f"Failed to spawn agent {index}: {e}")
            self.metrics.record_error()

    async def _run_agent(self, agent: "BaseAgent") -> None:
        """Run a single agent with metrics collection."""
        if not self.scenario or not self._semaphore:
            return

        agent_start = time.time()

        try:
            # Acquire semaphore for concurrency control
            async with self._semaphore:
                # Get task
                task = await self.scenario.get_task(agent)

                # Check for chaos injection
                if "CHAOS_INJECT_FAILURE" in task:
                    raise RuntimeError("Injected failure")
                if "CHAOS_INJECT_TIMEOUT" in task:
                    await asyncio.sleep(100)  # Will timeout

                # Run agent
                result = await agent.run(task)

                # Record metrics
                total_time = (time.time() - agent_start) * 1000
                self.metrics.total_agent_latency.observe(total_time)
                self.metrics.agent_work_latency.observe(result.execution_time_ms)
                self.metrics.agent_throughput.increment()

                if not result.success:
                    self.metrics.record_error()

        except asyncio.TimeoutError:
            self.metrics.record_timeout()
            self.metrics.total_agent_latency.observe((time.time() - agent_start) * 1000)

        except asyncio.CancelledError:
            self.metrics.record_cancelled()
            raise

        except Exception as e:
            logger.debug(f"Agent {agent.agent_id} failed: {e}")
            self.metrics.record_error()
            self.metrics.total_agent_latency.observe((time.time() - agent_start) * 1000)

        finally:
            # Cleanup
            if agent.agent_id in self._active_agents:
                del self._active_agents[agent.agent_id]
            if agent.agent_id in self._agent_tasks:
                del self._agent_tasks[agent.agent_id]
            if self.scenario and agent.agent_id in self.scenario.active_agents:
                self.scenario.active_agents.remove(agent.agent_id)

    async def _topology_switcher(self) -> None:
        """Periodically switch topologies during chaos testing."""
        if not self.topology_engine or self.config.topology_switch_interval <= 0:
            return

        if not self.scenario or not hasattr(self.scenario, 'get_next_topology'):
            return

        current_topology = "swarm"

        while not self._cancelled:
            await asyncio.sleep(self.config.topology_switch_interval)

            if self._cancelled:
                break

            try:
                switch_start = time.time()
                new_topology = self.scenario.get_next_topology(current_topology)

                logger.info(f"Switching topology: {current_topology} -> {new_topology}")
                await self.topology_engine.switch_topology(new_topology)

                switch_time = (time.time() - switch_start) * 1000
                self.metrics.topology_switch_latency.observe(switch_time)
                current_topology = new_topology

            except Exception as e:
                logger.error(f"Topology switch failed: {e}")

    async def _track_memory(self) -> None:
        """Track memory usage periodically."""
        try:
            import psutil
            process = psutil.Process(os.getpid())

            while self._running:
                memory_mb = process.memory_info().rss / (1024 * 1024)
                self.metrics.update_peak_memory(memory_mb)
                await asyncio.sleep(1)

        except ImportError:
            # psutil not available
            pass
        except asyncio.CancelledError:
            pass

    async def _cleanup_agents(self) -> None:
        """Cancel and cleanup all remaining agents."""
        # Cancel all agent tasks
        for task in self._agent_tasks.values():
            task.cancel()

        if self._agent_tasks:
            await asyncio.gather(*self._agent_tasks.values(), return_exceptions=True)

        self._agent_tasks.clear()
        self._active_agents.clear()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def handler(signum: int, frame: Any) -> None:
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self._cancelled = True

        try:
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
        except Exception:
            # Signal handling may not work in all contexts
            pass

    def cancel(self) -> None:
        """Cancel the running test."""
        self._cancelled = True


async def run_stress_test(
    scenario: str = "swarm",
    agents: int = 50,
    duration: int = 60,
    **kwargs: Any,
) -> StressTestResult:
    """
    Convenience function to run a stress test.

    Args:
        scenario: Scenario name (swarm, pipeline, hierarchy, chaos, scale)
        agents: Number of agents to spawn
        duration: Test duration in seconds
        **kwargs: Additional config options

    Returns:
        StressTestResult
    """
    config = StressTestConfig(
        scenario_name=scenario,
        target_agents=agents,
        duration_seconds=duration,
        **kwargs,
    )

    runner = StressTestRunner(config)
    return await runner.run()
