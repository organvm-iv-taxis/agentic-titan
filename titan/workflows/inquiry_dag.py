"""
Titan Workflows - Inquiry DAG (Directed Acyclic Graph) Support

Provides dependency graph management for inquiry workflow stages.
Enables parallel execution of independent stages and proper ordering
of dependent stages based on the DAG structure.

Reference: DependencyGraph pattern from agents/archetypes/orchestrator.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from titan.workflows.inquiry_config import InquiryWorkflow

logger = logging.getLogger("titan.workflows.inquiry_dag")


class ExecutionMode(StrEnum):
    """Execution modes for inquiry workflow."""

    SEQUENTIAL = "sequential"  # Execute one stage at a time in order
    PARALLEL = "parallel"  # Execute all ready stages in parallel
    STAGED = "staged"  # Execute stages level by level (respecting dependencies)


@dataclass
class StageNode:
    """A stage node in the dependency graph."""

    index: int
    name: str
    dependencies: list[int] = field(default_factory=list)
    dependents: list[int] = field(default_factory=list)  # Stages that depend on this one
    status: str = "pending"

    @property
    def is_ready(self) -> bool:
        """Check if stage has no unmet dependencies (for execution)."""
        return self.status == "pending" and len(self.dependencies) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "name": self.name,
            "dependencies": self.dependencies,
            "dependents": self.dependents,
            "status": self.status,
        }


@dataclass
class InquiryDependencyGraph:
    """
    Dependency graph for inquiry workflow stage execution ordering.

    Manages stage dependencies and provides methods for:
    - Finding ready stages (dependencies satisfied)
    - Topological sorting for level-based execution
    - Context stage identification (which stages provide context)
    """

    nodes: dict[int, StageNode]
    _adjacency: dict[int, list[int]]  # stage_index -> dependent stage indices

    @classmethod
    def from_workflow(cls, workflow: InquiryWorkflow) -> InquiryDependencyGraph:
        """
        Build dependency graph from an InquiryWorkflow.

        If stages have explicit dependencies, use them.
        Otherwise, fall back to sequential ordering (each stage depends on previous).

        Args:
            workflow: The inquiry workflow configuration

        Returns:
            InquiryDependencyGraph instance
        """
        nodes: dict[int, StageNode] = {}
        adjacency: dict[int, list[int]] = {}

        # Check if any stage has explicit dependencies
        has_explicit_deps = any(stage.dependencies is not None for stage in workflow.stages)

        for idx, stage in enumerate(workflow.stages):
            # Determine dependencies
            if has_explicit_deps and stage.dependencies is not None:
                # Use explicit dependencies
                deps = [d for d in stage.dependencies if 0 <= d < len(workflow.stages)]
            elif has_explicit_deps:
                # No explicit deps but others have them - this stage has no dependencies
                deps = []
            else:
                # Fall back to sequential: depend on previous stage
                deps = [idx - 1] if idx > 0 else []

            node = StageNode(
                index=idx,
                name=stage.name,
                dependencies=list(deps),
            )
            nodes[idx] = node
            adjacency[idx] = []

        # Build reverse adjacency (dependents)
        for idx, node in nodes.items():
            for dep_idx in node.dependencies:
                if dep_idx in adjacency:
                    adjacency[dep_idx].append(idx)
                    nodes[dep_idx].dependents.append(idx)

        graph = cls(nodes=nodes, _adjacency=adjacency)

        # Validate no cycles
        if graph._has_cycle():
            logger.warning("Cycle detected in stage dependencies, falling back to sequential")
            return cls._create_sequential_graph(workflow)

        return graph

    @classmethod
    def _create_sequential_graph(cls, workflow: InquiryWorkflow) -> InquiryDependencyGraph:
        """Create a simple sequential dependency graph."""
        nodes: dict[int, StageNode] = {}
        adjacency: dict[int, list[int]] = {}

        for idx, stage in enumerate(workflow.stages):
            deps = [idx - 1] if idx > 0 else []
            node = StageNode(
                index=idx,
                name=stage.name,
                dependencies=deps,
            )
            nodes[idx] = node
            adjacency[idx] = [idx + 1] if idx < len(workflow.stages) - 1 else []

        return cls(nodes=nodes, _adjacency=adjacency)

    def _has_cycle(self) -> bool:
        """Check if the graph has a cycle using DFS."""
        white, gray, black = 0, 1, 2
        color = {idx: white for idx in self.nodes}

        def dfs(node_idx: int) -> bool:
            color[node_idx] = gray
            for dep_idx in self._adjacency.get(node_idx, []):
                if color[dep_idx] == gray:
                    return True  # Back edge found - cycle
                if color[dep_idx] == white and dfs(dep_idx):
                    return True
            color[node_idx] = black
            return False

        for node_idx in self.nodes:
            if color[node_idx] == white:
                if dfs(node_idx):
                    return True
        return False

    def get_ready_stages(self, completed: set[int]) -> list[int]:
        """
        Get stages whose dependencies are all completed.

        Args:
            completed: Set of completed stage indices

        Returns:
            List of stage indices ready for execution
        """
        ready = []
        for idx, node in self.nodes.items():
            if node.status == "completed" or idx in completed:
                continue
            # Check if all dependencies are completed
            if all(dep_idx in completed for dep_idx in node.dependencies):
                ready.append(idx)
        return sorted(ready)  # Return in order for determinism

    def topological_sort(self) -> list[list[int]]:
        """
        Return stages grouped by execution level.

        Stages at the same level can be executed in parallel.
        Each level's stages depend only on stages in previous levels.

        Returns:
            List of lists, where each inner list contains stage indices
            that can be executed in parallel.
        """
        # Calculate in-degree for each node
        in_degree = {idx: len(node.dependencies) for idx, node in self.nodes.items()}
        levels: list[list[int]] = []
        remaining = set(in_degree.keys())

        while remaining:
            # Find all nodes with no remaining dependencies
            level = [idx for idx in remaining if in_degree[idx] == 0]
            if not level:
                # Cycle detected or all done
                logger.warning("Topological sort incomplete - possible cycle")
                break

            levels.append(sorted(level))  # Sort for determinism

            # Remove this level from consideration
            for idx in level:
                remaining.remove(idx)
                # Decrease in-degree for dependents
                for dependent_idx in self._adjacency.get(idx, []):
                    if dependent_idx in in_degree:
                        in_degree[dependent_idx] -= 1

        return levels

    def get_context_stages(self, stage_idx: int) -> list[int]:
        """
        Get stages that provide context for a given stage.

        In DAG mode, context comes only from dependency stages,
        not all previous stages (unlike sequential mode).

        Args:
            stage_idx: Index of the stage needing context

        Returns:
            List of stage indices that provide context (dependencies)
        """
        if stage_idx not in self.nodes:
            return []

        node = self.nodes[stage_idx]

        # Also include transitive dependencies for complete context
        visited: set[int] = set()
        stack = list(node.dependencies)

        while stack:
            dep_idx = stack.pop()
            if dep_idx in visited:
                continue
            visited.add(dep_idx)
            if dep_idx in self.nodes:
                stack.extend(self.nodes[dep_idx].dependencies)

        return sorted(visited)

    def mark_completed(self, stage_idx: int) -> None:
        """Mark a stage as completed."""
        if stage_idx in self.nodes:
            self.nodes[stage_idx].status = "completed"

    def mark_failed(self, stage_idx: int) -> None:
        """Mark a stage as failed."""
        if stage_idx in self.nodes:
            self.nodes[stage_idx].status = "failed"

    def get_all_levels_count(self) -> int:
        """Get the number of execution levels."""
        return len(self.topological_sort())

    def get_stage_level(self, stage_idx: int) -> int:
        """Get the execution level for a specific stage."""
        levels = self.topological_sort()
        for level_num, level_stages in enumerate(levels):
            if stage_idx in level_stages:
                return level_num
        return -1

    def can_parallelize(self) -> bool:
        """Check if any stages can be parallelized."""
        levels = self.topological_sort()
        return any(len(level) > 1 for level in levels)

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            "nodes": {str(idx): node.to_dict() for idx, node in self.nodes.items()},
            "levels": self.topological_sort(),
            "can_parallelize": self.can_parallelize(),
        }

    def __len__(self) -> int:
        """Return number of stages."""
        return len(self.nodes)


def validate_workflow_dependencies(workflow: InquiryWorkflow) -> list[str]:
    """
    Validate stage dependencies in a workflow.

    Args:
        workflow: The workflow to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []
    num_stages = len(workflow.stages)

    for idx, stage in enumerate(workflow.stages):
        if stage.dependencies is None:
            continue

        for dep_idx in stage.dependencies:
            if dep_idx < 0:
                errors.append(
                    f"Stage {idx} ({stage.name}) has negative dependency index: {dep_idx}"
                )
            elif dep_idx >= num_stages:
                errors.append(
                    f"Stage {idx} ({stage.name}) depends on non-existent stage: {dep_idx}"
                )
            elif dep_idx == idx:
                errors.append(f"Stage {idx} ({stage.name}) cannot depend on itself")
            elif dep_idx > idx:
                # Forward reference - check if it creates a cycle
                logger.warning(
                    f"Stage {idx} ({stage.name}) has forward dependency on stage {dep_idx} - "
                    "this may create cycles"
                )

    # Check for cycles
    if not errors:
        graph = InquiryDependencyGraph.from_workflow(workflow)
        if graph._has_cycle():
            errors.append("Workflow has circular dependencies")

    return errors
