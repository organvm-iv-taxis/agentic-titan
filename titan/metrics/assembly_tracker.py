"""Assembly path tracking for agent coordination.

Tracks decision paths through agent coordination to measure
assembly complexity and identify selection signals.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from titan.metrics.assembly import (
    AssemblyMetrics,
    AssemblyPath,
    AssemblyStep,
    StepType,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger("titan.metrics.assembly_tracker")


def _get_prometheus_metrics():
    """Lazy import to avoid circular dependency.

    Uses the parent package's get_metrics which handles the
    shadowing of titan/metrics.py by the titan/metrics/ package.
    """
    from titan.metrics import get_metrics
    return get_metrics()


class AssemblyTracker:
    """Tracks assembly paths through agent coordination.

    Records the sequence of steps that agents take during coordination,
    allowing measurement of assembly complexity and detection of
    selection signals.
    """

    def __init__(
        self,
        ensemble_id: str | None = None,
        max_paths: int = 1000,
        auto_complete_paths: bool = True,
    ) -> None:
        """Initialize the tracker.

        Args:
            ensemble_id: Identifier for this ensemble of paths.
            max_paths: Maximum number of paths to track (oldest dropped).
            auto_complete_paths: Whether to auto-complete paths on new path start.
        """
        self._ensemble_id = ensemble_id or str(uuid.uuid4())[:8]
        self._max_paths = max_paths
        self._auto_complete_paths = auto_complete_paths
        self._metrics = AssemblyMetrics(ensemble_id=self._ensemble_id)
        self._active_paths: dict[str, AssemblyPath] = {}
        self._step_counter = 0

    @property
    def ensemble_id(self) -> str:
        """Get the ensemble ID."""
        return self._ensemble_id

    @property
    def metrics(self) -> AssemblyMetrics:
        """Get the current assembly metrics."""
        return self._metrics

    @property
    def active_path_count(self) -> int:
        """Number of currently active paths."""
        return len(self._active_paths)

    def start_path(
        self,
        path_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AssemblyPath:
        """Start a new assembly path.

        Args:
            path_id: Optional path identifier. Generated if not provided.
            metadata: Optional metadata for the path.

        Returns:
            The newly created AssemblyPath.
        """
        if path_id is None:
            path_id = f"path_{uuid.uuid4().hex[:8]}"

        # Auto-complete existing path with same ID
        if path_id in self._active_paths and self._auto_complete_paths:
            self.complete_path(path_id)

        path = AssemblyPath(
            path_id=path_id,
            metadata=metadata or {},
        )
        self._active_paths[path_id] = path

        logger.debug(f"Started assembly path: {path_id}")
        return path

    def _record_prometheus_metrics(self, path: AssemblyPath) -> None:
        """Record assembly path metrics to Prometheus."""
        metrics = _get_prometheus_metrics()

        # Record path completion
        path_type = path.metadata.get("type", "generic")
        metrics.assembly_path_recorded(
            self._ensemble_id,
            path_type,
            path.assembly_index,
        )

        # Record individual step types
        for step in path.steps:
            metrics.assembly_step_recorded(self._ensemble_id, step.step_type.value)

        # Update aggregate metrics from AssemblyMetrics
        if self._metrics.total_objects > 0:
            metrics.set_assembly_index(
                self._ensemble_id,
                path_type,
                self._metrics.max_assembly_index,
            )
            metrics.set_total_assembly(self._ensemble_id, self._metrics.total_assembly)
            metrics.set_selection_signal(
                self._ensemble_id,
                self._metrics.selection_signal.value,
            )

    def add_step(
        self,
        path_id: str,
        step_type: StepType,
        description: str,
        input_state: dict[str, Any] | None = None,
        output_state: dict[str, Any] | None = None,
        agent_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AssemblyStep | None:
        """Add a step to an active path.

        Args:
            path_id: The path to add the step to.
            step_type: Type of assembly step.
            description: Human-readable description of the step.
            input_state: State before the step.
            output_state: State after the step.
            agent_ids: Agents involved in this step.
            metadata: Additional step metadata.

        Returns:
            The created AssemblyStep, or None if path not found.
        """
        path = self._active_paths.get(path_id)
        if path is None:
            logger.warning(f"Cannot add step: path {path_id} not found")
            return None

        self._step_counter += 1
        step_id = f"step_{self._step_counter}"

        step = AssemblyStep(
            step_id=step_id,
            step_type=step_type,
            description=description,
            input_state=input_state or {},
            output_state=output_state or {},
            agent_ids=agent_ids or [],
            metadata=metadata or {},
        )

        path.add_step(step)
        logger.debug(
            f"Added step {step_id} to path {path_id} "
            f"(total steps: {path.assembly_index})"
        )

        return step

    def complete_path(self, path_id: str) -> AssemblyPath | None:
        """Mark a path as complete and add it to metrics.

        Args:
            path_id: The path to complete.

        Returns:
            The completed path, or None if not found.
        """
        path = self._active_paths.pop(path_id, None)
        if path is None:
            logger.warning(f"Cannot complete: path {path_id} not found")
            return None

        path.complete()
        self._metrics.add_path(path)

        # Record to Prometheus
        self._record_prometheus_metrics(path)

        # Enforce max paths limit
        while len(self._metrics.paths) > self._max_paths:
            self._metrics.paths.pop(0)

        logger.info(
            f"Completed assembly path {path_id} with "
            f"assembly index {path.assembly_index}"
        )

        return path

    def abandon_path(self, path_id: str) -> bool:
        """Abandon an active path without adding to metrics.

        Args:
            path_id: The path to abandon.

        Returns:
            True if abandoned, False if not found.
        """
        if path_id in self._active_paths:
            del self._active_paths[path_id]
            logger.debug(f"Abandoned assembly path: {path_id}")
            return True
        return False

    def get_path(self, path_id: str) -> AssemblyPath | None:
        """Get an active path by ID.

        Args:
            path_id: The path ID.

        Returns:
            The AssemblyPath if found, None otherwise.
        """
        return self._active_paths.get(path_id)

    def record_decision(
        self,
        path_id: str,
        decision_description: str,
        options_considered: list[str],
        chosen_option: str,
        agent_id: str,
        confidence: float = 1.0,
    ) -> AssemblyStep | None:
        """Record a decision as an assembly step.

        Convenience method for recording decision points during
        agent coordination.

        Args:
            path_id: The path to add the decision to.
            decision_description: What was being decided.
            options_considered: All options that were considered.
            chosen_option: The option that was selected.
            agent_id: The agent making the decision.
            confidence: Confidence in the decision (0-1).

        Returns:
            The created AssemblyStep.
        """
        return self.add_step(
            path_id=path_id,
            step_type=StepType.DECISION,
            description=decision_description,
            input_state={"options": options_considered},
            output_state={"chosen": chosen_option},
            agent_ids=[agent_id],
            metadata={"confidence": confidence},
        )

    def record_coordination(
        self,
        path_id: str,
        coordination_description: str,
        agent_ids: list[str],
        coordination_type: str = "generic",
        result: dict[str, Any] | None = None,
    ) -> AssemblyStep | None:
        """Record a coordination event as an assembly step.

        Args:
            path_id: The path to add the coordination to.
            coordination_description: What was coordinated.
            agent_ids: Agents involved in coordination.
            coordination_type: Type of coordination (e.g., "broadcast", "consensus").
            result: Result of the coordination.

        Returns:
            The created AssemblyStep.
        """
        return self.add_step(
            path_id=path_id,
            step_type=StepType.COORDINATION,
            description=coordination_description,
            input_state={"agents": agent_ids, "type": coordination_type},
            output_state=result or {},
            agent_ids=agent_ids,
        )

    def record_transformation(
        self,
        path_id: str,
        transformation_description: str,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
        agent_id: str | None = None,
    ) -> AssemblyStep | None:
        """Record a state transformation as an assembly step.

        Args:
            path_id: The path to add the transformation to.
            transformation_description: What was transformed.
            before_state: State before transformation.
            after_state: State after transformation.
            agent_id: Optional agent performing the transformation.

        Returns:
            The created AssemblyStep.
        """
        return self.add_step(
            path_id=path_id,
            step_type=StepType.TRANSFORMATION,
            description=transformation_description,
            input_state=before_state,
            output_state=after_state,
            agent_ids=[agent_id] if agent_id else [],
        )

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current assembly metrics as a dictionary.

        Returns:
            Dictionary with all current metrics.
        """
        return {
            "ensemble_id": self._ensemble_id,
            "total_completed_paths": self._metrics.total_objects,
            "active_paths": self.active_path_count,
            "mean_assembly_index": self._metrics.mean_assembly_index,
            "max_assembly_index": self._metrics.max_assembly_index,
            "total_assembly": self._metrics.total_assembly,
            "selection_signal": self._metrics.selection_signal.value,
            "complexity_distribution": self._metrics.complexity_distribution,
        }

    def reset(self) -> None:
        """Reset the tracker, clearing all paths and metrics."""
        self._active_paths.clear()
        self._metrics = AssemblyMetrics(ensemble_id=self._ensemble_id)
        self._step_counter = 0
        logger.info(f"Reset assembly tracker: {self._ensemble_id}")

    def export_metrics(self) -> AssemblyMetrics:
        """Export a copy of the current metrics.

        Returns:
            Copy of the current AssemblyMetrics.
        """
        return AssemblyMetrics(
            ensemble_id=self._ensemble_id,
            paths=list(self._metrics.paths),
            created_at=self._metrics.created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracker state to dictionary.

        Returns:
            Dictionary representation of tracker state.
        """
        return {
            "ensemble_id": self._ensemble_id,
            "metrics": self._metrics.to_dict(),
            "active_paths": {
                pid: path.to_dict() for pid, path in self._active_paths.items()
            },
            "step_counter": self._step_counter,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        max_paths: int = 1000,
    ) -> AssemblyTracker:
        """Create tracker from dictionary state.

        Args:
            data: Dictionary from to_dict().
            max_paths: Maximum paths limit.

        Returns:
            Restored AssemblyTracker.
        """
        tracker = cls(
            ensemble_id=data.get("ensemble_id"),
            max_paths=max_paths,
        )

        tracker._metrics = AssemblyMetrics.from_dict(data.get("metrics", {}))
        tracker._step_counter = data.get("step_counter", 0)

        for path_id, path_data in data.get("active_paths", {}).items():
            tracker._active_paths[path_id] = AssemblyPath.from_dict(path_data)

        return tracker
