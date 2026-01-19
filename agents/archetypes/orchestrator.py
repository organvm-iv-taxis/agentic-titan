"""
Orchestrator Agent - Coordinates multi-agent workflows.

Capabilities:
- Task decomposition
- Agent selection
- Workflow management
- Result aggregation
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agents.framework.base_agent import BaseAgent, AgentResult, AgentState
from agents.personas import ORCHESTRATOR, say, think, announce
from adapters.base import LLMMessage
from adapters.router import get_router

if TYPE_CHECKING:
    from hive.topology import TopologyEngine, BaseTopology

logger = logging.getLogger("titan.agents.orchestrator")


@dataclass
class Subtask:
    """A subtask in the workflow."""

    id: str
    description: str
    agent_type: str
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"
    result: Any = None


@dataclass
class Workflow:
    """A multi-agent workflow."""

    task: str
    subtasks: list[Subtask] = field(default_factory=list)
    topology: str = "pipeline"
    status: str = "pending"
    results: dict[str, Any] = field(default_factory=dict)


class OrchestratorAgent(BaseAgent):
    """
    Agent that coordinates multi-agent workflows.

    Responsibilities:
    - Decompose tasks into subtasks
    - Select appropriate agents
    - Manage topology
    - Monitor progress
    - Aggregate results
    """

    def __init__(
        self,
        task: str | None = None,
        available_agents: list[str] | None = None,
        topology_engine: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="orchestrator",
            capabilities=["planning", "execution"],
            **kwargs,
        )
        self.task = task
        self.available_agents = available_agents or ["researcher", "coder", "reviewer"]
        self._topology_engine = topology_engine
        self.workflow: Workflow | None = None
        self._router = get_router()

    async def initialize(self) -> None:
        """Initialize the orchestrator."""
        announce(ORCHESTRATOR, "Initializing", {
            "Task": self.task[:50] if self.task else "None",
            "Available Agents": ", ".join(self.available_agents),
        })

        await self._router.initialize()

    async def work(self) -> Workflow:
        """
        Execute orchestration workflow.

        Returns:
            Workflow with results
        """
        if not self.task:
            say(ORCHESTRATOR, "No task specified")
            return Workflow(task="")

        say(ORCHESTRATOR, f"Orchestrating: {self.task}")
        self.workflow = Workflow(task=self.task)

        # Step 1: Analyze task and select topology
        think(ORCHESTRATOR, "Analyzing task requirements...")
        self.increment_turn()
        topology_type = await self._select_topology()
        self.workflow.topology = topology_type
        say(ORCHESTRATOR, f"Selected topology: {topology_type}")

        # Step 2: Decompose into subtasks
        think(ORCHESTRATOR, "Decomposing task into subtasks...")
        self.increment_turn()
        subtasks = await self._decompose_task()
        self.workflow.subtasks = subtasks
        say(ORCHESTRATOR, f"Created {len(subtasks)} subtasks")

        # Step 3: Execute workflow based on topology
        think(ORCHESTRATOR, "Executing workflow...")
        await self._execute_workflow()

        # Step 4: Aggregate results
        think(ORCHESTRATOR, "Aggregating results...")
        self.increment_turn()
        final_result = await self._aggregate_results()
        self.workflow.results["final"] = final_result
        self.workflow.status = "completed"

        # Log decision
        await self.log_decision(
            decision=f"Orchestrated {len(subtasks)} subtasks with {topology_type} topology",
            category="orchestration",
            rationale=f"Task: {self.task[:100]}",
            tags=["orchestration", topology_type],
        )

        say(ORCHESTRATOR, "Orchestration complete")
        return self.workflow

    async def shutdown(self) -> None:
        """Cleanup orchestrator."""
        say(ORCHESTRATOR, "Orchestrator shutting down")

        # Store workflow pattern
        if self._hive_mind and self.workflow:
            await self.remember(
                content=f"Workflow for {self.task}:\n"
                f"Topology: {self.workflow.topology}\n"
                f"Subtasks: {len(self.workflow.subtasks)}",
                importance=0.7,
                tags=["workflow", "orchestration"],
            )

    async def _select_topology(self) -> str:
        """Select appropriate topology for the task."""
        if self._topology_engine:
            suggestion = self._topology_engine.suggest_topology(self.task)
            return suggestion["recommended"]

        # Simple heuristic if no engine
        task_lower = self.task.lower()

        if any(kw in task_lower for kw in ["consensus", "agree", "brainstorm"]):
            return "swarm"
        elif any(kw in task_lower for kw in ["review", "then", "after"]):
            return "pipeline"
        elif any(kw in task_lower for kw in ["coordinate", "manage"]):
            return "hierarchy"
        else:
            return "pipeline"  # Default

    async def _decompose_task(self) -> list[Subtask]:
        """Decompose task into subtasks."""
        messages = [
            LLMMessage(
                role="user",
                content=f"""Decompose this task into subtasks for a multi-agent system:

Task: {self.task}

Available agent types: {', '.join(self.available_agents)}

For each subtask, specify:
1. Description
2. Which agent type should handle it
3. Dependencies on other subtasks (if any)

Format:
SUBTASK: <description>
AGENT: <agent_type>
DEPENDS: <subtask_ids or "none">
---

Create 2-5 subtasks.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a task planner. Break down complex tasks efficiently.",
            max_tokens=800,
        )

        return self._parse_subtasks(response.content)

    def _parse_subtasks(self, content: str) -> list[Subtask]:
        """Parse subtasks from LLM response."""
        subtasks = []
        current: dict[str, Any] = {}
        subtask_id = 0

        for line in content.split("\n"):
            line = line.strip()

            if line.startswith("SUBTASK:"):
                if current:
                    subtasks.append(Subtask(
                        id=f"st-{subtask_id}",
                        description=current.get("description", ""),
                        agent_type=current.get("agent", "researcher"),
                        dependencies=current.get("depends", []),
                    ))
                    subtask_id += 1
                current = {"description": line[8:].strip()}

            elif line.startswith("AGENT:"):
                agent = line[6:].strip().lower()
                if agent in self.available_agents:
                    current["agent"] = agent
                else:
                    current["agent"] = "researcher"  # Fallback

            elif line.startswith("DEPENDS:"):
                deps = line[8:].strip().lower()
                if deps != "none":
                    current["depends"] = [d.strip() for d in deps.split(",")]
                else:
                    current["depends"] = []

        # Add last subtask
        if current:
            subtasks.append(Subtask(
                id=f"st-{subtask_id}",
                description=current.get("description", ""),
                agent_type=current.get("agent", "researcher"),
                dependencies=current.get("depends", []),
            ))

        return subtasks

    async def _execute_workflow(self) -> None:
        """Execute the workflow based on topology."""
        if not self.workflow:
            return

        # Simple sequential execution for now
        # TODO: Implement proper topology-based execution
        for subtask in self.workflow.subtasks:
            say(ORCHESTRATOR, f"Executing subtask: {subtask.description[:40]}...")
            self.increment_turn()

            subtask.status = "running"

            # Simulate agent execution (would spawn real agents)
            result = await self._simulate_agent(subtask)

            subtask.result = result
            subtask.status = "completed"
            self.workflow.results[subtask.id] = result

            say(ORCHESTRATOR, f"Subtask {subtask.id} completed")

    async def _simulate_agent(self, subtask: Subtask) -> str:
        """Simulate agent execution (placeholder)."""
        # In real implementation, this would spawn actual agents
        messages = [
            LLMMessage(
                role="user",
                content=f"""You are a {subtask.agent_type} agent. Complete this subtask:

{subtask.description}

Provide a brief result (under 100 words).""",
            )
        ]

        response = await self._router.complete(
            messages,
            system=f"You are a {subtask.agent_type} agent completing a subtask.",
            max_tokens=200,
        )

        return response.content

    async def _aggregate_results(self) -> str:
        """Aggregate results from all subtasks."""
        if not self.workflow:
            return ""

        results_text = "\n\n".join(
            f"[{st.agent_type}] {st.description}:\n{st.result}"
            for st in self.workflow.subtasks
            if st.result
        )

        messages = [
            LLMMessage(
                role="user",
                content=f"""Aggregate these subtask results into a final response:

Original Task: {self.workflow.task}

Subtask Results:
{results_text}

Provide a coherent final answer that addresses the original task.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a synthesizer. Combine results into a coherent response.",
            max_tokens=500,
        )

        return response.content
