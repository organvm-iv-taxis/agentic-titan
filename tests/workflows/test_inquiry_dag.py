"""
Tests for Inquiry DAG (Directed Acyclic Graph) Support.

Tests the InquiryDependencyGraph class and DAG-based workflow execution.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from titan.workflows.inquiry_config import (
    InquiryStage,
    InquiryWorkflow,
    CognitiveStyle,
)
from titan.workflows.inquiry_dag import (
    InquiryDependencyGraph,
    ExecutionMode,
    StageNode,
    validate_workflow_dependencies,
)
from titan.workflows.inquiry_engine import InquiryEngine, InquirySession, InquiryStatus


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_sequential_workflow() -> InquiryWorkflow:
    """A simple 3-stage sequential workflow (no explicit deps)."""
    return InquiryWorkflow(
        name="Sequential Test",
        description="Simple sequential workflow",
        stages=[
            InquiryStage(
                name="Stage A",
                role="Role A",
                description="First stage",
                prompt_template="test_a",
            ),
            InquiryStage(
                name="Stage B",
                role="Role B",
                description="Second stage",
                prompt_template="test_b",
            ),
            InquiryStage(
                name="Stage C",
                role="Role C",
                description="Third stage",
                prompt_template="test_c",
            ),
        ],
    )


@pytest.fixture
def parallel_workflow() -> InquiryWorkflow:
    """Workflow where B and C can run in parallel after A."""
    return InquiryWorkflow(
        name="Parallel Test",
        description="B and C depend only on A",
        stages=[
            InquiryStage(
                name="Stage A",
                role="Role A",
                description="First stage",
                prompt_template="test_a",
                dependencies=[],  # No dependencies
            ),
            InquiryStage(
                name="Stage B",
                role="Role B",
                description="Depends on A",
                prompt_template="test_b",
                dependencies=[0],  # Depends on Stage A
            ),
            InquiryStage(
                name="Stage C",
                role="Role C",
                description="Also depends on A",
                prompt_template="test_c",
                dependencies=[0],  # Depends on Stage A
            ),
            InquiryStage(
                name="Stage D",
                role="Role D",
                description="Depends on B and C",
                prompt_template="test_d",
                dependencies=[1, 2],  # Depends on B and C
            ),
        ],
    )


@pytest.fixture
def diamond_workflow() -> InquiryWorkflow:
    """Diamond dependency pattern: A -> B, C -> D."""
    return InquiryWorkflow(
        name="Diamond Test",
        description="Diamond dependency pattern",
        stages=[
            InquiryStage(
                name="Start",
                role="Starter",
                description="Starting point",
                prompt_template="start",
                dependencies=[],
            ),
            InquiryStage(
                name="Path 1",
                role="Worker 1",
                description="First parallel path",
                prompt_template="path1",
                dependencies=[0],
            ),
            InquiryStage(
                name="Path 2",
                role="Worker 2",
                description="Second parallel path",
                prompt_template="path2",
                dependencies=[0],
            ),
            InquiryStage(
                name="Merge",
                role="Merger",
                description="Merge paths",
                prompt_template="merge",
                dependencies=[1, 2],
            ),
        ],
    )


@pytest.fixture
def inquiry_engine() -> InquiryEngine:
    """Create an inquiry engine for testing."""
    return InquiryEngine()


# =============================================================================
# InquiryDependencyGraph Tests
# =============================================================================


class TestInquiryDependencyGraph:
    """Tests for InquiryDependencyGraph class."""

    def test_from_workflow_sequential(self, simple_sequential_workflow: InquiryWorkflow) -> None:
        """Test building graph from sequential workflow."""
        graph = InquiryDependencyGraph.from_workflow(simple_sequential_workflow)

        assert len(graph) == 3
        assert graph.nodes[0].dependencies == []
        assert graph.nodes[1].dependencies == [0]
        assert graph.nodes[2].dependencies == [1]

    def test_from_workflow_parallel(self, parallel_workflow: InquiryWorkflow) -> None:
        """Test building graph from parallel workflow."""
        graph = InquiryDependencyGraph.from_workflow(parallel_workflow)

        assert len(graph) == 4
        assert graph.nodes[0].dependencies == []
        assert graph.nodes[1].dependencies == [0]
        assert graph.nodes[2].dependencies == [0]
        assert sorted(graph.nodes[3].dependencies) == [1, 2]

    def test_topological_sort_sequential(self, simple_sequential_workflow: InquiryWorkflow) -> None:
        """Test topological sort for sequential workflow."""
        graph = InquiryDependencyGraph.from_workflow(simple_sequential_workflow)
        levels = graph.topological_sort()

        assert len(levels) == 3
        assert levels[0] == [0]
        assert levels[1] == [1]
        assert levels[2] == [2]

    def test_topological_sort_parallel(self, parallel_workflow: InquiryWorkflow) -> None:
        """Test topological sort for parallel workflow."""
        graph = InquiryDependencyGraph.from_workflow(parallel_workflow)
        levels = graph.topological_sort()

        assert len(levels) == 3
        assert levels[0] == [0]  # A first
        assert sorted(levels[1]) == [1, 2]  # B and C can run in parallel
        assert levels[2] == [3]  # D last

    def test_topological_sort_diamond(self, diamond_workflow: InquiryWorkflow) -> None:
        """Test topological sort for diamond pattern."""
        graph = InquiryDependencyGraph.from_workflow(diamond_workflow)
        levels = graph.topological_sort()

        assert len(levels) == 3
        assert levels[0] == [0]  # Start
        assert sorted(levels[1]) == [1, 2]  # Path 1 and Path 2
        assert levels[2] == [3]  # Merge

    def test_get_ready_stages(self, parallel_workflow: InquiryWorkflow) -> None:
        """Test getting ready stages."""
        graph = InquiryDependencyGraph.from_workflow(parallel_workflow)

        # Initially only A is ready
        ready = graph.get_ready_stages(completed=set())
        assert ready == [0]

        # After A completes, B and C are ready
        ready = graph.get_ready_stages(completed={0})
        assert sorted(ready) == [1, 2]

        # After B completes, C and D should not be ready (D needs C)
        ready = graph.get_ready_stages(completed={0, 1})
        assert ready == [2]

        # After B and C complete, D is ready
        ready = graph.get_ready_stages(completed={0, 1, 2})
        assert ready == [3]

    def test_get_context_stages(self, parallel_workflow: InquiryWorkflow) -> None:
        """Test getting context stages for a given stage."""
        graph = InquiryDependencyGraph.from_workflow(parallel_workflow)

        # Stage A has no context stages
        assert graph.get_context_stages(0) == []

        # Stage B's context is just A
        assert graph.get_context_stages(1) == [0]

        # Stage D's context is A, B, C (transitive)
        context = graph.get_context_stages(3)
        assert sorted(context) == [0, 1, 2]

    def test_can_parallelize(
        self,
        simple_sequential_workflow: InquiryWorkflow,
        parallel_workflow: InquiryWorkflow,
    ) -> None:
        """Test parallelization detection."""
        seq_graph = InquiryDependencyGraph.from_workflow(simple_sequential_workflow)
        par_graph = InquiryDependencyGraph.from_workflow(parallel_workflow)

        assert not seq_graph.can_parallelize()
        assert par_graph.can_parallelize()

    def test_mark_completed(self, simple_sequential_workflow: InquiryWorkflow) -> None:
        """Test marking stages as completed."""
        graph = InquiryDependencyGraph.from_workflow(simple_sequential_workflow)

        assert graph.nodes[0].status == "pending"
        graph.mark_completed(0)
        assert graph.nodes[0].status == "completed"

    def test_mark_failed(self, simple_sequential_workflow: InquiryWorkflow) -> None:
        """Test marking stages as failed."""
        graph = InquiryDependencyGraph.from_workflow(simple_sequential_workflow)

        graph.mark_failed(1)
        assert graph.nodes[1].status == "failed"

    def test_get_stage_level(self, parallel_workflow: InquiryWorkflow) -> None:
        """Test getting the execution level for a stage."""
        graph = InquiryDependencyGraph.from_workflow(parallel_workflow)

        assert graph.get_stage_level(0) == 0
        assert graph.get_stage_level(1) == 1
        assert graph.get_stage_level(2) == 1
        assert graph.get_stage_level(3) == 2

    def test_to_dict(self, parallel_workflow: InquiryWorkflow) -> None:
        """Test serialization to dictionary."""
        graph = InquiryDependencyGraph.from_workflow(parallel_workflow)
        data = graph.to_dict()

        assert "nodes" in data
        assert "levels" in data
        assert "can_parallelize" in data
        assert len(data["nodes"]) == 4
        assert data["can_parallelize"] is True


class TestCycleDetection:
    """Tests for cycle detection in dependency graphs."""

    def test_no_cycle_sequential(self, simple_sequential_workflow: InquiryWorkflow) -> None:
        """Sequential workflow has no cycle."""
        graph = InquiryDependencyGraph.from_workflow(simple_sequential_workflow)
        assert not graph._has_cycle()

    def test_cycle_detection(self) -> None:
        """Test that cycles are detected and handled."""
        # Create workflow with circular dependency
        workflow = InquiryWorkflow(
            name="Cyclic Test",
            description="Has a cycle",
            stages=[
                InquiryStage(
                    name="A",
                    role="A",
                    description="A",
                    prompt_template="a",
                    dependencies=[2],  # A depends on C
                ),
                InquiryStage(
                    name="B",
                    role="B",
                    description="B",
                    prompt_template="b",
                    dependencies=[0],  # B depends on A
                ),
                InquiryStage(
                    name="C",
                    role="C",
                    description="C",
                    prompt_template="c",
                    dependencies=[1],  # C depends on B -> CYCLE
                ),
            ],
        )

        # Should fall back to sequential on cycle detection
        graph = InquiryDependencyGraph.from_workflow(workflow)
        # Sequential fallback means A depends on nothing, B on A, C on B
        assert graph.nodes[0].dependencies == []
        assert graph.nodes[1].dependencies == [0]
        assert graph.nodes[2].dependencies == [1]


class TestValidation:
    """Tests for workflow dependency validation."""

    def test_valid_workflow(self, parallel_workflow: InquiryWorkflow) -> None:
        """Valid workflow should have no errors."""
        errors = validate_workflow_dependencies(parallel_workflow)
        assert errors == []

    def test_invalid_dependency_index(self) -> None:
        """Invalid dependency index should be caught."""
        workflow = InquiryWorkflow(
            name="Invalid Test",
            description="Has invalid dependency",
            stages=[
                InquiryStage(
                    name="A",
                    role="A",
                    description="A",
                    prompt_template="a",
                    dependencies=[5],  # Invalid - index 5 doesn't exist
                ),
            ],
        )
        errors = validate_workflow_dependencies(workflow)
        assert len(errors) == 1
        assert "non-existent stage" in errors[0]

    def test_negative_dependency_index(self) -> None:
        """Negative dependency index should be caught."""
        workflow = InquiryWorkflow(
            name="Negative Test",
            description="Has negative dependency",
            stages=[
                InquiryStage(
                    name="A",
                    role="A",
                    description="A",
                    prompt_template="a",
                    dependencies=[-1],
                ),
            ],
        )
        errors = validate_workflow_dependencies(workflow)
        assert len(errors) == 1
        assert "negative dependency" in errors[0]

    def test_self_dependency(self) -> None:
        """Self-dependency should be caught."""
        workflow = InquiryWorkflow(
            name="Self Dep Test",
            description="Self dependency",
            stages=[
                InquiryStage(
                    name="A",
                    role="A",
                    description="A",
                    prompt_template="a",
                    dependencies=[0],  # Depends on itself
                ),
            ],
        )
        errors = validate_workflow_dependencies(workflow)
        assert len(errors) == 1
        assert "cannot depend on itself" in errors[0]


# =============================================================================
# DAG Workflow Execution Tests
# =============================================================================


class TestDAGWorkflowExecution:
    """Tests for DAG-based workflow execution."""

    @pytest.mark.asyncio
    async def test_run_dag_workflow_sequential(
        self,
        inquiry_engine: InquiryEngine,
        simple_sequential_workflow: InquiryWorkflow,
    ) -> None:
        """Test DAG execution in sequential mode."""
        session = await inquiry_engine.start_inquiry(
            "Test topic",
            workflow=simple_sequential_workflow,
        )

        result = await inquiry_engine.run_dag_workflow(
            session,
            execution_mode=ExecutionMode.SEQUENTIAL,
        )

        assert result.status == InquiryStatus.COMPLETED
        assert len(result.results) == 3
        assert "dag_info" in result.metadata

    @pytest.mark.asyncio
    async def test_run_dag_workflow_staged(
        self,
        inquiry_engine: InquiryEngine,
        parallel_workflow: InquiryWorkflow,
    ) -> None:
        """Test DAG execution in staged mode."""
        session = await inquiry_engine.start_inquiry(
            "Test topic",
            workflow=parallel_workflow,
        )

        result = await inquiry_engine.run_dag_workflow(
            session,
            execution_mode=ExecutionMode.STAGED,
        )

        assert result.status == InquiryStatus.COMPLETED
        assert len(result.results) == 4
        assert result.metadata["dag_info"]["can_parallelize"] is True

    @pytest.mark.asyncio
    async def test_run_dag_workflow_parallel(
        self,
        inquiry_engine: InquiryEngine,
        parallel_workflow: InquiryWorkflow,
    ) -> None:
        """Test DAG execution in full parallel mode."""
        session = await inquiry_engine.start_inquiry(
            "Test topic",
            workflow=parallel_workflow,
        )

        result = await inquiry_engine.run_dag_workflow(
            session,
            execution_mode=ExecutionMode.PARALLEL,
        )

        assert result.status == InquiryStatus.COMPLETED
        assert len(result.results) == 4

    @pytest.mark.asyncio
    async def test_dag_context_for_stage(
        self,
        inquiry_engine: InquiryEngine,
        parallel_workflow: InquiryWorkflow,
    ) -> None:
        """Test getting DAG-aware context for stages."""
        session = await inquiry_engine.start_inquiry(
            "Test topic",
            workflow=parallel_workflow,
        )

        # Run first stage
        await inquiry_engine.run_stage(session, 0)

        # Get context for stage B (should include A)
        context = inquiry_engine.get_dag_context_for_stage(session, 1)
        assert "Stage A" in context

        # Run B and C
        await inquiry_engine.run_stage(session, 1)
        await inquiry_engine.run_stage(session, 2)

        # Get context for stage D (should include A, B, C)
        context = inquiry_engine.get_dag_context_for_stage(session, 3)
        assert "Stage A" in context
        assert "Stage B" in context
        assert "Stage C" in context

    @pytest.mark.asyncio
    async def test_dag_respects_dependencies(
        self,
        inquiry_engine: InquiryEngine,
        diamond_workflow: InquiryWorkflow,
    ) -> None:
        """Test that DAG execution respects dependencies."""
        session = await inquiry_engine.start_inquiry(
            "Diamond test",
            workflow=diamond_workflow,
        )

        result = await inquiry_engine.run_dag_workflow(
            session,
            execution_mode=ExecutionMode.STAGED,
        )

        assert result.status == InquiryStatus.COMPLETED

        # Verify execution order via stage indices in results
        stage_order = [r.stage_index for r in result.results]

        # Start (0) must be before Path 1 (1) and Path 2 (2)
        assert stage_order.index(0) < stage_order.index(1)
        assert stage_order.index(0) < stage_order.index(2)

        # Path 1 and Path 2 must be before Merge (3)
        assert stage_order.index(1) < stage_order.index(3)
        assert stage_order.index(2) < stage_order.index(3)


class TestStageNode:
    """Tests for StageNode dataclass."""

    def test_is_ready(self) -> None:
        """Test is_ready property."""
        # No dependencies - ready to execute
        node = StageNode(index=0, name="Test", dependencies=[])
        assert node.is_ready is True

        # Has dependencies - not ready (deps not satisfied)
        node_with_deps = StageNode(index=1, name="Test", dependencies=[0])
        assert node_with_deps.is_ready is False  # has unmet deps

        # No deps but already completed - not ready
        completed_node = StageNode(index=0, name="Test", dependencies=[])
        completed_node.status = "completed"
        assert completed_node.is_ready is False  # already done

    def test_to_dict(self) -> None:
        """Test serialization."""
        node = StageNode(index=0, name="Test", dependencies=[1, 2], dependents=[3])
        data = node.to_dict()

        assert data["index"] == 0
        assert data["name"] == "Test"
        assert data["dependencies"] == [1, 2]
        assert data["dependents"] == [3]
        assert data["status"] == "pending"
