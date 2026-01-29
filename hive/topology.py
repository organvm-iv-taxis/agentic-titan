"""
Hive Mind - Topology Engine

Enables dynamic restructuring of agent relationships based on task type.
Agents can organize into different patterns for optimal collaboration.

Supported Topologies:
- Swarm: All-to-all, emergent behavior (brainstorming, exploration)
- Hierarchy: Tree structure (command chains, delegation)
- Pipeline: DAG/sequential (workflows with stages)
- Mesh: Resilient grid (fault-tolerant tasks)
- Ring: Token passing (voting, sequential processing)
- Star: Hub and spoke (coordinator pattern)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from agents.framework.errors import InvalidTopologyError, TopologyError
from titan.metrics import get_metrics

if TYPE_CHECKING:
    from hive.memory import HiveMind

logger = logging.getLogger("titan.hive.topology")


class TopologyType(Enum):
    """Supported topology types."""

    SWARM = "swarm"         # All-to-all, emergent
    HIERARCHY = "hierarchy"  # Tree structure
    PIPELINE = "pipeline"    # DAG/sequential
    MESH = "mesh"           # Resilient grid
    RING = "ring"           # Token passing
    STAR = "star"           # Hub and spoke


@dataclass
class AgentNode:
    """Represents an agent in the topology."""

    agent_id: str
    name: str
    capabilities: list[str]
    role: str | None = None  # Role within topology
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)
    neighbors: list[str] = field(default_factory=list)  # For mesh/ring
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "capabilities": self.capabilities,
            "role": self.role,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "neighbors": self.neighbors,
            "metadata": self.metadata,
        }


@dataclass
class TaskProfile:
    """Profile of a task used for topology selection."""

    requires_consensus: bool = False
    has_sequential_stages: bool = False
    needs_fault_tolerance: bool = False
    has_clear_leader: bool = False
    is_voting_based: bool = False
    parallel_subtasks: int = 0
    complexity: str = "medium"  # low, medium, high
    estimated_agents: int = 2

    @classmethod
    def from_task(cls, task_description: str) -> TaskProfile:
        """Analyze task description to create profile."""
        import re

        description_lower = task_description.lower()

        # Extract agent count from description
        estimated_agents = 2  # default
        # Look for patterns like "20 agents", "team of 10", "5 workers"
        agent_patterns = [
            r"(\d+)\s*agents?",
            r"team\s+of\s+(\d+)",
            r"(\d+)\s*workers?",
            r"(\d+)\s*members?",
            r"large\s+team",  # implies > 5
        ]
        for pattern in agent_patterns:
            match = re.search(pattern, description_lower)
            if match:
                if "large" in pattern:
                    estimated_agents = 10
                else:
                    estimated_agents = int(match.group(1))
                break

        # Detect complexity
        complexity = "medium"
        if any(kw in description_lower for kw in ["simple", "basic", "easy", "trivial"]):
            complexity = "low"
        elif any(kw in description_lower for kw in ["complex", "advanced", "sophisticated", "large"]):
            complexity = "high"

        return cls(
            requires_consensus=any(
                kw in description_lower
                for kw in ["consensus", "agree", "vote", "decide together", "brainstorm"]
            ),
            has_sequential_stages=any(
                kw in description_lower
                for kw in ["then", "after", "step", "stage", "pipeline", "workflow"]
            ),
            needs_fault_tolerance=any(
                kw in description_lower
                for kw in ["critical", "important", "must not fail", "reliable", "resilient"]
            ),
            has_clear_leader=any(
                kw in description_lower
                for kw in ["coordinate", "lead", "manage", "orchestrate", "delegate"]
            ),
            is_voting_based=any(
                kw in description_lower for kw in ["vote", "poll", "election", "rank"]
            ),
            estimated_agents=estimated_agents,
            complexity=complexity,
        )


# ============================================================================
# Base Topology
# ============================================================================


class BaseTopology(ABC):
    """Abstract base class for topology implementations."""

    topology_type: TopologyType

    def __init__(self) -> None:
        self.nodes: dict[str, AgentNode] = {}
        self._initialized = False

    @abstractmethod
    def add_agent(
        self,
        agent_id: str,
        name: str,
        capabilities: list[str],
        **kwargs: Any,
    ) -> AgentNode:
        """Add an agent to the topology."""
        pass

    @abstractmethod
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the topology."""
        pass

    @abstractmethod
    def get_message_targets(
        self,
        source_agent_id: str,
        message_type: str = "broadcast",
    ) -> list[str]:
        """Get agent IDs that should receive a message."""
        pass

    @abstractmethod
    def get_routing_path(
        self,
        source_agent_id: str,
        target_agent_id: str,
    ) -> list[str]:
        """Get the path for routing a message between agents."""
        pass

    def get_agent(self, agent_id: str) -> AgentNode | None:
        """Get an agent node."""
        return self.nodes.get(agent_id)

    def list_agents(self) -> list[AgentNode]:
        """List all agents in topology."""
        return list(self.nodes.values())

    def to_dict(self) -> dict[str, Any]:
        """Serialize topology state."""
        return {
            "type": self.topology_type.value,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
        }


# ============================================================================
# Topology Implementations
# ============================================================================


class SwarmTopology(BaseTopology):
    """
    Swarm topology - All agents can communicate with all others.

    Best for:
    - Brainstorming
    - Consensus building
    - Exploration tasks
    - Emergent behavior
    """

    topology_type = TopologyType.SWARM

    def add_agent(
        self,
        agent_id: str,
        name: str,
        capabilities: list[str],
        **kwargs: Any,
    ) -> AgentNode:
        node = AgentNode(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            role="peer",
            neighbors=list(self.nodes.keys()),  # All existing agents
            metadata=kwargs,
        )

        # Update existing agents to include new neighbor
        for existing in self.nodes.values():
            existing.neighbors.append(agent_id)

        self.nodes[agent_id] = node

        # Record neighbor interaction metrics
        metrics = get_metrics()
        metrics.neighbor_interaction("swarm_add")

        logger.debug(f"Added agent {agent_id} to swarm")
        return node

    def remove_agent(self, agent_id: str) -> bool:
        if agent_id not in self.nodes:
            return False

        del self.nodes[agent_id]

        # Remove from all neighbors
        for node in self.nodes.values():
            if agent_id in node.neighbors:
                node.neighbors.remove(agent_id)

        return True

    def get_message_targets(
        self,
        source_agent_id: str,
        message_type: str = "broadcast",
    ) -> list[str]:
        # In swarm, everyone gets everything
        return [aid for aid in self.nodes.keys() if aid != source_agent_id]

    def get_routing_path(
        self,
        source_agent_id: str,
        target_agent_id: str,
    ) -> list[str]:
        # Direct connection
        return [source_agent_id, target_agent_id]


class HierarchyTopology(BaseTopology):
    """
    Hierarchy topology - Tree structure with parent-child relationships.

    Best for:
    - Command chains
    - Delegation patterns
    - Reporting structures
    """

    topology_type = TopologyType.HIERARCHY

    def __init__(self) -> None:
        super().__init__()
        self.root_id: str | None = None

    def add_agent(
        self,
        agent_id: str,
        name: str,
        capabilities: list[str],
        *,
        parent_id: str | None = None,
        role: str = "worker",
        **kwargs: Any,
    ) -> AgentNode:
        # First agent becomes root
        if not self.nodes:
            role = "root"
            self.root_id = agent_id

        # Validate parent
        if parent_id and parent_id not in self.nodes:
            raise TopologyError(f"Parent agent {parent_id} not found")

        node = AgentNode(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            role=role,
            parent_id=parent_id,
            metadata=kwargs,
        )

        # Add to parent's children
        if parent_id:
            self.nodes[parent_id].child_ids.append(agent_id)

        self.nodes[agent_id] = node
        logger.debug(f"Added agent {agent_id} to hierarchy under {parent_id}")
        return node

    def remove_agent(self, agent_id: str) -> bool:
        node = self.nodes.get(agent_id)
        if not node:
            return False

        # Reassign children to parent
        if node.child_ids:
            for child_id in node.child_ids:
                child = self.nodes.get(child_id)
                if child:
                    child.parent_id = node.parent_id

        # Remove from parent's children
        if node.parent_id:
            parent = self.nodes.get(node.parent_id)
            if parent and agent_id in parent.child_ids:
                parent.child_ids.remove(agent_id)
                parent.child_ids.extend(node.child_ids)

        del self.nodes[agent_id]
        return True

    def get_message_targets(
        self,
        source_agent_id: str,
        message_type: str = "broadcast",
    ) -> list[str]:
        node = self.nodes.get(source_agent_id)
        if not node:
            return []

        if message_type == "up":
            # Message to parent only
            return [node.parent_id] if node.parent_id else []
        elif message_type == "down":
            # Message to children only
            return node.child_ids.copy()
        else:
            # Broadcast to parent and children
            targets = node.child_ids.copy()
            if node.parent_id:
                targets.append(node.parent_id)
            return targets

    def get_routing_path(
        self,
        source_agent_id: str,
        target_agent_id: str,
    ) -> list[str]:
        # Find path through tree
        source_path = self._path_to_root(source_agent_id)
        target_path = self._path_to_root(target_agent_id)

        # Find common ancestor
        source_set = set(source_path)
        common_ancestor = None
        for node_id in target_path:
            if node_id in source_set:
                common_ancestor = node_id
                break

        if not common_ancestor:
            return []

        # Build path through common ancestor
        up_path = []
        for node_id in source_path:
            up_path.append(node_id)
            if node_id == common_ancestor:
                break

        down_path = []
        for node_id in target_path:
            if node_id == common_ancestor:
                break
            down_path.append(node_id)

        return up_path + list(reversed(down_path))

    def _path_to_root(self, agent_id: str) -> list[str]:
        """Get path from agent to root."""
        path = []
        current = agent_id
        while current:
            path.append(current)
            node = self.nodes.get(current)
            current = node.parent_id if node else None
        return path


class PipelineTopology(BaseTopology):
    """
    Pipeline topology - Sequential stages with directed flow.

    Best for:
    - Workflows with clear stages
    - ETL processes
    - Review chains
    """

    topology_type = TopologyType.PIPELINE

    def __init__(self) -> None:
        super().__init__()
        self.stages: list[list[str]] = []  # Agents at each stage

    def add_agent(
        self,
        agent_id: str,
        name: str,
        capabilities: list[str],
        *,
        stage: int = -1,  # -1 means append new stage
        **kwargs: Any,
    ) -> AgentNode:
        if stage == -1:
            # Create new stage
            stage = len(self.stages)
            self.stages.append([])

        while len(self.stages) <= stage:
            self.stages.append([])

        node = AgentNode(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            role=f"stage-{stage}",
            metadata={"stage": stage, **kwargs},
        )

        # Previous stage agents are "parents", next stage are "children"
        if stage > 0:
            node.parent_id = self.stages[stage - 1][0] if self.stages[stage - 1] else None
        if stage < len(self.stages) - 1 and self.stages[stage + 1]:
            node.child_ids = self.stages[stage + 1].copy()

        self.stages[stage].append(agent_id)
        self.nodes[agent_id] = node

        # Update neighbors within same stage
        for peer_id in self.stages[stage]:
            if peer_id != agent_id:
                node.neighbors.append(peer_id)
                self.nodes[peer_id].neighbors.append(agent_id)

        logger.debug(f"Added agent {agent_id} to pipeline stage {stage}")
        return node

    def remove_agent(self, agent_id: str) -> bool:
        node = self.nodes.get(agent_id)
        if not node:
            return False

        stage = node.metadata.get("stage", 0)
        if stage < len(self.stages) and agent_id in self.stages[stage]:
            self.stages[stage].remove(agent_id)

        del self.nodes[agent_id]
        return True

    def get_message_targets(
        self,
        source_agent_id: str,
        message_type: str = "broadcast",
    ) -> list[str]:
        node = self.nodes.get(source_agent_id)
        if not node:
            return []

        stage = node.metadata.get("stage", 0)

        if message_type == "next":
            # Next stage only
            if stage + 1 < len(self.stages):
                return self.stages[stage + 1].copy()
            return []
        elif message_type == "prev":
            # Previous stage only
            if stage > 0:
                return self.stages[stage - 1].copy()
            return []
        elif message_type == "stage":
            # Same stage peers
            return node.neighbors.copy()
        else:
            # All adjacent stages + peers
            targets = node.neighbors.copy()
            if stage > 0:
                targets.extend(self.stages[stage - 1])
            if stage + 1 < len(self.stages):
                targets.extend(self.stages[stage + 1])
            return targets

    def get_routing_path(
        self,
        source_agent_id: str,
        target_agent_id: str,
    ) -> list[str]:
        source = self.nodes.get(source_agent_id)
        target = self.nodes.get(target_agent_id)
        if not source or not target:
            return []

        source_stage = source.metadata.get("stage", 0)
        target_stage = target.metadata.get("stage", 0)

        path = [source_agent_id]
        if source_stage < target_stage:
            for s in range(source_stage + 1, target_stage):
                if self.stages[s]:
                    path.append(self.stages[s][0])
        elif source_stage > target_stage:
            for s in range(source_stage - 1, target_stage, -1):
                if self.stages[s]:
                    path.append(self.stages[s][0])
        path.append(target_agent_id)

        return path


class MeshTopology(BaseTopology):
    """
    Mesh topology - Resilient grid with multiple paths.

    Best for:
    - Fault-tolerant tasks
    - Distributed processing
    - High availability
    """

    topology_type = TopologyType.MESH

    def __init__(self, connectivity: int = 3) -> None:
        super().__init__()
        self.connectivity = connectivity  # Target number of neighbors per node

    def add_agent(
        self,
        agent_id: str,
        name: str,
        capabilities: list[str],
        **kwargs: Any,
    ) -> AgentNode:
        node = AgentNode(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            role="mesh-node",
            metadata=kwargs,
        )

        # Connect to existing nodes (up to connectivity limit)
        existing = list(self.nodes.keys())
        neighbors = existing[: self.connectivity]

        node.neighbors = neighbors
        for neighbor_id in neighbors:
            self.nodes[neighbor_id].neighbors.append(agent_id)

        self.nodes[agent_id] = node
        logger.debug(f"Added agent {agent_id} to mesh")
        return node

    def remove_agent(self, agent_id: str) -> bool:
        if agent_id not in self.nodes:
            return False

        # Remove from all neighbors
        for node in self.nodes.values():
            if agent_id in node.neighbors:
                node.neighbors.remove(agent_id)

        del self.nodes[agent_id]
        return True

    def get_message_targets(
        self,
        source_agent_id: str,
        message_type: str = "broadcast",
    ) -> list[str]:
        node = self.nodes.get(source_agent_id)
        if not node:
            return []

        if message_type == "neighbors":
            return node.neighbors.copy()
        else:
            # Broadcast reaches everyone
            return [aid for aid in self.nodes.keys() if aid != source_agent_id]

    def get_routing_path(
        self,
        source_agent_id: str,
        target_agent_id: str,
    ) -> list[str]:
        # BFS for shortest path
        if source_agent_id not in self.nodes or target_agent_id not in self.nodes:
            return []

        visited = {source_agent_id}
        queue = [(source_agent_id, [source_agent_id])]

        while queue:
            current, path = queue.pop(0)

            if current == target_agent_id:
                return path

            node = self.nodes[current]
            for neighbor in node.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []


class RingTopology(BaseTopology):
    """
    Ring topology - Circular token passing.

    Best for:
    - Voting
    - Sequential processing
    - Round-robin tasks
    """

    topology_type = TopologyType.RING

    def __init__(self) -> None:
        super().__init__()
        self.order: list[str] = []  # Ring order

    def add_agent(
        self,
        agent_id: str,
        name: str,
        capabilities: list[str],
        **kwargs: Any,
    ) -> AgentNode:
        node = AgentNode(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            role="ring-member",
            metadata=kwargs,
        )

        # Insert into ring
        if self.order:
            # Connect to previous last and first
            prev_id = self.order[-1]
            first_id = self.order[0]

            self.nodes[prev_id].neighbors = [agent_id]
            node.neighbors = [first_id]
        else:
            node.neighbors = []

        self.order.append(agent_id)
        self.nodes[agent_id] = node

        # Update first node if ring has more than one
        if len(self.order) > 1:
            self.nodes[self.order[0]].neighbors = [self.order[1]]
            node.neighbors = [self.order[0]]

        logger.debug(f"Added agent {agent_id} to ring")
        return node

    def remove_agent(self, agent_id: str) -> bool:
        if agent_id not in self.nodes:
            return False

        idx = self.order.index(agent_id)
        self.order.remove(agent_id)

        # Reconnect ring
        if len(self.order) > 1:
            prev_idx = (idx - 1) % len(self.order)
            next_idx = idx % len(self.order)
            self.nodes[self.order[prev_idx]].neighbors = [self.order[next_idx]]

        del self.nodes[agent_id]
        return True

    def get_message_targets(
        self,
        source_agent_id: str,
        message_type: str = "broadcast",
    ) -> list[str]:
        node = self.nodes.get(source_agent_id)
        if not node:
            return []

        if message_type == "next":
            return node.neighbors.copy()
        else:
            # In ring, broadcast goes around
            return [aid for aid in self.order if aid != source_agent_id]

    def get_routing_path(
        self,
        source_agent_id: str,
        target_agent_id: str,
    ) -> list[str]:
        if source_agent_id not in self.order or target_agent_id not in self.order:
            return []

        source_idx = self.order.index(source_agent_id)
        target_idx = self.order.index(target_agent_id)

        # Go around the ring
        path = []
        current = source_idx
        while current != target_idx:
            path.append(self.order[current])
            current = (current + 1) % len(self.order)
        path.append(self.order[target_idx])

        return path


class StarTopology(BaseTopology):
    """
    Star topology - Central hub with spokes.

    Best for:
    - Coordinator pattern
    - Central orchestration
    - Hub-and-spoke workflows
    """

    topology_type = TopologyType.STAR

    def __init__(self) -> None:
        super().__init__()
        self.hub_id: str | None = None

    def add_agent(
        self,
        agent_id: str,
        name: str,
        capabilities: list[str],
        *,
        is_hub: bool = False,
        **kwargs: Any,
    ) -> AgentNode:
        # First agent or explicit hub becomes the hub
        if not self.nodes or is_hub:
            role = "hub"
            if self.hub_id and is_hub:
                # Demote old hub
                self.nodes[self.hub_id].role = "spoke"
            self.hub_id = agent_id
        else:
            role = "spoke"

        node = AgentNode(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            role=role,
            parent_id=self.hub_id if role == "spoke" else None,
            metadata=kwargs,
        )

        if role == "spoke" and self.hub_id:
            node.neighbors = [self.hub_id]
            self.nodes[self.hub_id].child_ids.append(agent_id)
            self.nodes[self.hub_id].neighbors.append(agent_id)

        self.nodes[agent_id] = node
        logger.debug(f"Added agent {agent_id} to star as {role}")
        return node

    def remove_agent(self, agent_id: str) -> bool:
        if agent_id not in self.nodes:
            return False

        node = self.nodes[agent_id]

        if node.role == "hub":
            # Promote another agent to hub
            if node.child_ids:
                new_hub_id = node.child_ids[0]
                self.hub_id = new_hub_id
                new_hub = self.nodes[new_hub_id]
                new_hub.role = "hub"
                new_hub.parent_id = None
                new_hub.child_ids = [
                    cid for cid in node.child_ids if cid != new_hub_id
                ]
                new_hub.neighbors = new_hub.child_ids.copy()

                # Update all spokes
                for spoke_id in new_hub.child_ids:
                    self.nodes[spoke_id].parent_id = new_hub_id
                    self.nodes[spoke_id].neighbors = [new_hub_id]
            else:
                self.hub_id = None
        else:
            # Remove spoke from hub
            if self.hub_id:
                hub = self.nodes[self.hub_id]
                if agent_id in hub.child_ids:
                    hub.child_ids.remove(agent_id)
                if agent_id in hub.neighbors:
                    hub.neighbors.remove(agent_id)

        del self.nodes[agent_id]
        return True

    def get_message_targets(
        self,
        source_agent_id: str,
        message_type: str = "broadcast",
    ) -> list[str]:
        node = self.nodes.get(source_agent_id)
        if not node:
            return []

        if node.role == "hub":
            # Hub can reach all spokes
            return node.child_ids.copy()
        else:
            # Spokes can only reach hub
            return [self.hub_id] if self.hub_id else []

    def get_routing_path(
        self,
        source_agent_id: str,
        target_agent_id: str,
    ) -> list[str]:
        source = self.nodes.get(source_agent_id)
        target = self.nodes.get(target_agent_id)

        if not source or not target:
            return []

        if source.role == "hub" or target.role == "hub":
            return [source_agent_id, target_agent_id]
        else:
            # Spoke to spoke goes through hub
            return [source_agent_id, self.hub_id, target_agent_id] if self.hub_id else []


# ============================================================================
# Topology Engine
# ============================================================================


class TopologyEngine:
    """
    Engine for managing and switching between topologies.

    Integrates:
    - Task analysis (keyword + LLM-powered)
    - Dynamic topology switching with events
    - Episodic learning from outcomes

    Analyzes tasks and selects appropriate topologies.
    Handles topology transitions and state management.
    """

    TOPOLOGY_CLASSES: dict[TopologyType, type[BaseTopology]] = {
        TopologyType.SWARM: SwarmTopology,
        TopologyType.HIERARCHY: HierarchyTopology,
        TopologyType.PIPELINE: PipelineTopology,
        TopologyType.MESH: MeshTopology,
        TopologyType.RING: RingTopology,
        TopologyType.STAR: StarTopology,
    }

    def __init__(
        self,
        hive_mind: HiveMind | None = None,
        event_bus: Any | None = None,  # EventBus type
        task_analyzer: Any | None = None,  # TaskAnalyzer type
        episodic_learner: Any | None = None,  # EpisodicLearner type
    ) -> None:
        self.hive_mind = hive_mind
        self._event_bus = event_bus
        self._task_analyzer = task_analyzer
        self._episodic_learner = episodic_learner
        self._current_topology: BaseTopology | None = None
        self._topology_history: list[dict[str, Any]] = []
        self._current_task: str | None = None
        self._transition_lock = False

    @property
    def current_topology(self) -> BaseTopology | None:
        """Get current topology."""
        return self._current_topology

    @property
    def is_transitioning(self) -> bool:
        """Check if topology is currently transitioning."""
        return self._transition_lock

    def create_topology(
        self,
        topology_type: TopologyType | str,
        **kwargs: Any,
    ) -> BaseTopology:
        """
        Create a new topology.

        Args:
            topology_type: Type of topology
            **kwargs: Topology-specific configuration

        Returns:
            New topology instance
        """
        if isinstance(topology_type, str):
            try:
                topology_type = TopologyType(topology_type)
            except ValueError as exc:
                raise InvalidTopologyError(topology_type) from exc

        topology_class = self.TOPOLOGY_CLASSES.get(topology_type)
        if not topology_class:
            raise InvalidTopologyError(topology_type.value)

        topology = topology_class(**kwargs)
        self._current_topology = topology

        logger.info(f"Created {topology_type.value} topology")
        return topology

    def select_topology(self, task: str | TaskProfile) -> TopologyType:
        """
        Select the best topology for a task.

        Uses episodic learning if available, otherwise falls back to rules.

        Args:
            task: Task description or profile

        Returns:
            Recommended topology type
        """
        if isinstance(task, str):
            profile = TaskProfile.from_task(task)
        else:
            profile = task

        # Check episodic learner first (if available)
        if self._episodic_learner:
            recommended, confidence = self._episodic_learner.get_recommendation(profile)
            if confidence > 0.6:  # High confidence from learning
                logger.info(f"Using learned recommendation: {recommended.value} (confidence: {confidence:.2f})")
                return recommended

        # Rule-based selection
        if profile.is_voting_based:
            return TopologyType.RING
        if profile.requires_consensus:
            return TopologyType.SWARM
        if profile.has_sequential_stages:
            return TopologyType.PIPELINE
        if profile.needs_fault_tolerance:
            return TopologyType.MESH
        if profile.has_clear_leader:
            if profile.estimated_agents <= 5:
                return TopologyType.STAR
            return TopologyType.HIERARCHY

        # Default to swarm (most flexible)
        return TopologyType.SWARM

    async def analyze_and_select(
        self,
        task: str,
        use_llm: bool = True,
    ) -> tuple[TopologyType, dict[str, Any]]:
        """
        Analyze task with LLM and select optimal topology.

        Args:
            task: Task description
            use_llm: Whether to use LLM for analysis

        Returns:
            Tuple of (selected topology, analysis details)
        """
        analysis = None

        # Try LLM analysis
        if use_llm and self._task_analyzer:
            try:
                analysis = await self._task_analyzer.analyze(task)
                logger.info(
                    f"LLM analysis: {analysis.recommended_topology.value} "
                    f"(confidence: {analysis.confidence:.2f})"
                )
                return analysis.recommended_topology, analysis.to_dict()
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")

        # Fallback to rule-based
        profile = TaskProfile.from_task(task)
        selected = self.select_topology(profile)

        return selected, {
            "recommended_topology": selected.value,
            "confidence": 0.5,
            "reasoning": "Rule-based selection",
            "profile": {
                "requires_consensus": profile.requires_consensus,
                "has_sequential_stages": profile.has_sequential_stages,
                "needs_fault_tolerance": profile.needs_fault_tolerance,
                "has_clear_leader": profile.has_clear_leader,
                "is_voting_based": profile.is_voting_based,
            },
        }

    def suggest_topology(self, task: str) -> dict[str, Any]:
        """
        Analyze task and suggest topology with reasoning.

        Args:
            task: Task description

        Returns:
            Suggestion with type and reasoning
        """
        profile = TaskProfile.from_task(task)
        selected = self.select_topology(profile)

        reasons = []
        if profile.requires_consensus:
            reasons.append("Task requires consensus building")
        if profile.has_sequential_stages:
            reasons.append("Task has sequential stages")
        if profile.needs_fault_tolerance:
            reasons.append("Task needs fault tolerance")
        if profile.has_clear_leader:
            reasons.append("Task has a clear leader/coordinator")
        if profile.is_voting_based:
            reasons.append("Task is voting-based")

        # Add learning-based suggestion if available
        learning_info = None
        if self._episodic_learner:
            recommended, confidence = self._episodic_learner.get_recommendation(profile)
            learning_info = {
                "learned_recommendation": recommended.value,
                "confidence": confidence,
                "overrides_rules": confidence > 0.6 and recommended != selected,
            }
            if learning_info["overrides_rules"]:
                selected = recommended
                reasons.append(f"Learned from {self._episodic_learner._preferences.get(self._episodic_learner._profile_key(profile), {}).get(recommended, {})}")

        return {
            "recommended": selected.value,
            "reasons": reasons or ["Default selection (most flexible)"],
            "profile": {
                "requires_consensus": profile.requires_consensus,
                "has_sequential_stages": profile.has_sequential_stages,
                "needs_fault_tolerance": profile.needs_fault_tolerance,
                "has_clear_leader": profile.has_clear_leader,
                "is_voting_based": profile.is_voting_based,
            },
            "learning": learning_info,
        }

    async def switch_topology(
        self,
        new_type: TopologyType | str,
        migrate_agents: bool = True,
        reason: str | None = None,
    ) -> BaseTopology:
        """
        Switch to a new topology with event notifications.

        Args:
            new_type: New topology type
            migrate_agents: Whether to migrate existing agents
            reason: Reason for the switch

        Returns:
            New topology
        """
        import time

        if self._transition_lock:
            raise TopologyError("Topology transition already in progress")

        self._transition_lock = True
        start_time = time.time()

        try:
            old_topology = self._current_topology
            old_type = old_topology.topology_type.value if old_topology else "none"
            new_type_value = new_type.value if isinstance(new_type, TopologyType) else new_type
            agents_to_migrate = []

            # Emit TOPOLOGY_CHANGING event
            if self._event_bus:
                from hive.events import emit_topology_changing
                await emit_topology_changing(
                    self._event_bus,
                    old_type=old_type,
                    new_type=new_type_value,
                    agent_count=len(old_topology.nodes) if old_topology else 0,
                )

            if old_topology and migrate_agents:
                agents_to_migrate = old_topology.list_agents()
                self._topology_history.append({
                    **old_topology.to_dict(),
                    "reason": reason,
                    "timestamp": time.time(),
                })

            new_topology = self.create_topology(new_type)

            # Migrate agents with events
            for agent in agents_to_migrate:
                if self._event_bus:
                    from hive.events import emit_agent_migrating, emit_agent_migrated
                    await emit_agent_migrating(
                        self._event_bus,
                        agent_id=agent.agent_id,
                        from_topology=old_type,
                        to_topology=new_type_value,
                    )

                node = new_topology.add_agent(
                    agent_id=agent.agent_id,
                    name=agent.name,
                    capabilities=agent.capabilities,
                )

                if self._event_bus:
                    await emit_agent_migrated(
                        self._event_bus,
                        agent_id=agent.agent_id,
                        new_topology=new_type_value,
                        new_role=node.role,
                    )

            # Persist to Hive Mind
            if self.hive_mind:
                await self.hive_mind.set_topology(new_topology.to_dict())

            duration_ms = (time.time() - start_time) * 1000
            duration_seconds = duration_ms / 1000

            # Record topology switch metrics
            metrics = get_metrics()
            metrics.topology_switch(old_type, new_type_value, duration_seconds)
            metrics.set_topology(new_type_value, len(new_topology.nodes))

            # Emit TOPOLOGY_CHANGED event
            if self._event_bus:
                from hive.events import emit_topology_changed
                await emit_topology_changed(
                    self._event_bus,
                    old_type=old_type,
                    new_type=new_type_value,
                    agent_count=len(new_topology.nodes),
                    duration_ms=duration_ms,
                )

            logger.info(
                f"Switched topology: {old_type} -> {new_type_value} "
                f"({len(agents_to_migrate)} agents migrated in {duration_ms:.1f}ms)"
            )

            return new_topology

        finally:
            self._transition_lock = False

    def start_task(self, task: str, topology: TopologyType) -> None:
        """
        Start tracking a task for episodic learning.

        Args:
            task: Task description
            topology: Selected topology
        """
        self._current_task = task
        if self._episodic_learner and self._current_topology:
            profile = TaskProfile.from_task(task)
            self._episodic_learner.start_episode(
                task_description=task,
                selected_topology=topology,
                task_profile=profile,
                agent_count=len(self._current_topology.nodes),
            )

    def end_task(
        self,
        success: bool,
        completion_time_ms: float,
        error_rate: float = 0.0,
        user_feedback: float | None = None,
    ) -> None:
        """
        End task tracking and record outcome for learning.

        Args:
            success: Whether task completed successfully
            completion_time_ms: Time to complete
            error_rate: Error rate during execution
            user_feedback: Optional user feedback (-1 to 1)
        """
        # Calculate actual utilization metrics
        utilization = self.get_agent_utilization()
        communication = self.get_communication_overhead()

        if self._episodic_learner:
            from hive.learning import EpisodeOutcome
            outcome = EpisodeOutcome(
                success=success,
                completion_time_ms=completion_time_ms,
                agent_utilization=utilization.get("average", 0.7),
                communication_overhead=communication.get("overhead_ratio", 1.0),
                topology_switches=len(self._topology_history),
                error_rate=error_rate,
                user_feedback=user_feedback,
            )
            self._episodic_learner.end_episode(outcome)

        self._current_task = None

    def get_learning_stats(self) -> dict[str, Any] | None:
        """Get episodic learning statistics."""
        if self._episodic_learner:
            return self._episodic_learner.get_statistics()
        return None

    # =========================================================================
    # Agent Utilization Metrics
    # =========================================================================

    def get_agent_utilization(self) -> dict[str, Any]:
        """
        Calculate actual agent utilization metrics.

        Returns:
            Dictionary with utilization metrics per agent and aggregate stats.
        """
        if not self._current_topology:
            return {"average": 0.0, "per_agent": {}}

        agents = self._current_topology.list_agents()
        if not agents:
            return {"average": 0.0, "per_agent": {}}

        per_agent: dict[str, dict[str, Any]] = {}
        total_utilization = 0.0

        for agent in agents:
            # Get metrics from agent metadata if available
            metadata = agent.metadata

            # Calculate utilization based on available metrics
            tasks_completed = metadata.get("tasks_completed", 0)
            tasks_assigned = metadata.get("tasks_assigned", 1)
            active_time_ms = metadata.get("active_time_ms", 0)
            total_time_ms = metadata.get("total_time_ms", 1)
            tokens_used = metadata.get("tokens_used", 0)
            token_budget = metadata.get("token_budget", 1)

            # Task completion ratio
            task_ratio = tasks_completed / max(tasks_assigned, 1)

            # Time utilization (active vs idle)
            time_ratio = active_time_ms / max(total_time_ms, 1)

            # Token utilization
            token_ratio = tokens_used / max(token_budget, 1)

            # Weighted utilization score
            utilization = (
                task_ratio * 0.4 +
                time_ratio * 0.4 +
                min(token_ratio, 1.0) * 0.2  # Cap at 100%
            )

            per_agent[agent.agent_id] = {
                "utilization": round(utilization, 3),
                "task_completion_ratio": round(task_ratio, 3),
                "time_utilization": round(time_ratio, 3),
                "token_utilization": round(min(token_ratio, 1.0), 3),
                "tasks_completed": tasks_completed,
                "tasks_assigned": tasks_assigned,
                "active_time_ms": active_time_ms,
                "tokens_used": tokens_used,
                "role": agent.role,
            }

            total_utilization += utilization

        average = total_utilization / len(agents) if agents else 0.0

        return {
            "average": round(average, 3),
            "agent_count": len(agents),
            "per_agent": per_agent,
            "topology_type": self._current_topology.topology_type.value,
        }

    def get_communication_overhead(self) -> dict[str, Any]:
        """
        Calculate communication overhead metrics.

        Returns:
            Dictionary with communication metrics.
        """
        if not self._current_topology:
            return {"overhead_ratio": 1.0, "messages": 0}

        agents = self._current_topology.list_agents()
        if not agents:
            return {"overhead_ratio": 1.0, "messages": 0}

        total_messages_sent = 0
        total_messages_received = 0
        total_bytes_transferred = 0
        topology_switches = len(self._topology_history)

        for agent in agents:
            metadata = agent.metadata
            total_messages_sent += metadata.get("messages_sent", 0)
            total_messages_received += metadata.get("messages_received", 0)
            total_bytes_transferred += metadata.get("bytes_transferred", 0)

        # Calculate efficiency metrics
        total_messages = total_messages_sent + total_messages_received
        agent_count = len(agents)

        # Theoretical minimum messages for different topologies
        topology_type = self._current_topology.topology_type
        if topology_type == TopologyType.SWARM:
            # O(n²) communication
            theoretical_min = agent_count * (agent_count - 1) if agent_count > 1 else 1
        elif topology_type == TopologyType.STAR:
            # O(n) communication through hub
            theoretical_min = 2 * (agent_count - 1) if agent_count > 1 else 1
        elif topology_type == TopologyType.RING:
            # O(n) for round-robin
            theoretical_min = agent_count if agent_count > 0 else 1
        elif topology_type == TopologyType.PIPELINE:
            # O(n) for sequential stages
            theoretical_min = agent_count - 1 if agent_count > 1 else 1
        elif topology_type == TopologyType.HIERARCHY:
            # O(log n) for tree
            import math
            theoretical_min = int(math.log2(max(agent_count, 2))) * agent_count
        else:  # MESH
            # O(n) with multiple paths
            theoretical_min = agent_count * 3 if agent_count > 0 else 1

        # Overhead ratio: actual / theoretical (1.0 is optimal)
        overhead_ratio = total_messages / max(theoretical_min, 1) if total_messages > 0 else 1.0

        return {
            "overhead_ratio": round(overhead_ratio, 3),
            "total_messages": total_messages,
            "messages_sent": total_messages_sent,
            "messages_received": total_messages_received,
            "bytes_transferred": total_bytes_transferred,
            "theoretical_minimum": theoretical_min,
            "topology_switches": topology_switches,
            "topology_type": topology_type.value,
            "agent_count": agent_count,
        }

    def update_agent_metrics(
        self,
        agent_id: str,
        *,
        tasks_completed: int | None = None,
        tasks_assigned: int | None = None,
        active_time_ms: int | None = None,
        total_time_ms: int | None = None,
        tokens_used: int | None = None,
        token_budget: int | None = None,
        messages_sent: int | None = None,
        messages_received: int | None = None,
        bytes_transferred: int | None = None,
    ) -> bool:
        """
        Update metrics for a specific agent.

        Args:
            agent_id: Agent to update
            **kwargs: Metric values to update

        Returns:
            True if update successful
        """
        if not self._current_topology:
            return False

        agent = self._current_topology.get_agent(agent_id)
        if not agent:
            return False

        # Update metadata with provided values
        updates = {
            "tasks_completed": tasks_completed,
            "tasks_assigned": tasks_assigned,
            "active_time_ms": active_time_ms,
            "total_time_ms": total_time_ms,
            "tokens_used": tokens_used,
            "token_budget": token_budget,
            "messages_sent": messages_sent,
            "messages_received": messages_received,
            "bytes_transferred": bytes_transferred,
        }

        for key, value in updates.items():
            if value is not None:
                agent.metadata[key] = value

        return True

    def increment_agent_counter(
        self,
        agent_id: str,
        counter: str,
        amount: int = 1,
    ) -> bool:
        """
        Increment a counter metric for an agent.

        Args:
            agent_id: Agent to update
            counter: Counter name (e.g., "tasks_completed", "messages_sent")
            amount: Amount to increment by

        Returns:
            True if update successful
        """
        if not self._current_topology:
            return False

        agent = self._current_topology.get_agent(agent_id)
        if not agent:
            return False

        current = agent.metadata.get(counter, 0)
        agent.metadata[counter] = current + amount
        return True

    def get_prometheus_metrics(self) -> list[tuple[str, str, float, dict[str, str]]]:
        """
        Get metrics formatted for Prometheus exposition.

        Returns:
            List of (metric_name, metric_type, value, labels) tuples
        """
        metrics: list[tuple[str, str, float, dict[str, str]]] = []

        if not self._current_topology:
            return metrics

        topology_type = self._current_topology.topology_type.value

        # Utilization metrics
        utilization = self.get_agent_utilization()
        metrics.append((
            "titan_agent_utilization_average",
            "gauge",
            utilization["average"],
            {"topology": topology_type},
        ))

        for agent_id, agent_metrics in utilization.get("per_agent", {}).items():
            labels = {"agent_id": agent_id, "topology": topology_type, "role": agent_metrics.get("role", "unknown")}
            metrics.append(("titan_agent_utilization", "gauge", agent_metrics["utilization"], labels))
            metrics.append(("titan_agent_tasks_completed", "counter", agent_metrics["tasks_completed"], labels))
            metrics.append(("titan_agent_tasks_assigned", "counter", agent_metrics["tasks_assigned"], labels))
            metrics.append(("titan_agent_tokens_used", "counter", agent_metrics["tokens_used"], labels))

        # Communication metrics
        comm = self.get_communication_overhead()
        metrics.append((
            "titan_communication_overhead_ratio",
            "gauge",
            comm["overhead_ratio"],
            {"topology": topology_type},
        ))
        metrics.append((
            "titan_messages_total",
            "counter",
            float(comm["total_messages"]),
            {"topology": topology_type},
        ))
        metrics.append((
            "titan_bytes_transferred_total",
            "counter",
            float(comm["bytes_transferred"]),
            {"topology": topology_type},
        ))
        metrics.append((
            "titan_topology_switches_total",
            "counter",
            float(comm["topology_switches"]),
            {},
        ))

        # Topology info
        metrics.append((
            "titan_agent_count",
            "gauge",
            float(len(self._current_topology.nodes)),
            {"topology": topology_type},
        ))

        return metrics

    def __repr__(self) -> str:
        current = self._current_topology
        features = []
        if self._event_bus:
            features.append("events")
        if self._task_analyzer:
            features.append("llm")
        if self._episodic_learner:
            features.append("learning")
        features_str = f" [{', '.join(features)}]" if features else ""
        return (
            f"<TopologyEngine "
            f"current={current.topology_type.value if current else 'none'} "
            f"agents={len(current.nodes) if current else 0}{features_str}>"
        )
