"""Tests for core agent archetypes: Orchestrator, Researcher, Coder, Reviewer, Paper2Code."""

from __future__ import annotations

import pytest

from agents.archetypes.orchestrator import OrchestratorAgent, Subtask, DependencyGraph, ExecutionMode
from agents.archetypes.researcher import ResearcherAgent, ResearchTask
from agents.archetypes.coder import CoderAgent, CodeTask
from agents.archetypes.reviewer import ReviewerAgent
from agents.archetypes.paper2code import Paper2CodeAgent
from agents.framework.base_agent import AgentState


class TestCoderAgent:
    """Tests for CoderAgent."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(CoderAgent)
        assert agent.name == "coder"
        assert "code_generation" in agent.capabilities
        assert agent.language == "python"
        assert agent.task_description is None

    def test_constructor_with_params(self, agent_factory):
        """Test initialization with parameters."""
        agent = agent_factory(
            CoderAgent,
            task_description="Write a hello world function",
            language="rust",
        )
        assert agent.language == "rust"
        assert agent.task_description == "Write a hello world function"

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(CoderAgent, task_description="Test task")
        await agent.initialize()
        mock_llm_router.initialize.assert_called()

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic work execution."""
        agent = agent_factory(CoderAgent, task_description="Write hello world")
        await agent.initialize()
        result = await agent.work()

        assert isinstance(result, CodeTask)
        assert result.description == "Write hello world"
        assert result.language == "python"
        assert mock_llm_router.complete.call_count >= 3  # plan, code, tests, review

    @pytest.mark.asyncio
    async def test_work_without_task(self, agent_factory):
        """Test work without task description."""
        agent = agent_factory(CoderAgent)
        await agent.initialize()
        result = await agent.work()

        assert isinstance(result, CodeTask)
        assert result.description == ""

    @pytest.mark.asyncio
    async def test_shutdown(self, agent_factory, mock_hive_mind):
        """Test agent shutdown with memory storage."""
        agent = agent_factory(CoderAgent, task_description="Test")
        await agent.initialize()
        agent.task = CodeTask(
            description="Test",
            language="python",
            code_output="def test(): pass",
        )
        await agent.shutdown()
        # Should store code patterns if hive_mind available
        mock_hive_mind.remember.assert_called()


class TestResearcherAgent:
    """Tests for ResearcherAgent."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(ResearcherAgent)
        assert agent.name == "researcher"
        assert "web_search" in agent.capabilities
        assert agent.topic is None

    def test_constructor_with_topic(self, agent_factory):
        """Test initialization with topic."""
        agent = agent_factory(ResearcherAgent, topic="Machine Learning")
        assert agent.topic == "Machine Learning"

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router, mock_hive_mind):
        """Test agent initialization with recall."""
        agent = agent_factory(ResearcherAgent, topic="AI Research")
        mock_hive_mind.recall.return_value = [{"content": "Previous research"}]
        await agent.initialize()
        mock_llm_router.initialize.assert_called()

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic research work."""
        agent = agent_factory(ResearcherAgent, topic="Quantum Computing")
        await agent.initialize()
        result = await agent.work()

        assert isinstance(result, ResearchTask)
        assert result.topic == "Quantum Computing"
        assert len(result.questions) > 0

    @pytest.mark.asyncio
    async def test_work_without_topic(self, agent_factory):
        """Test work without topic."""
        agent = agent_factory(ResearcherAgent)
        await agent.initialize()
        result = await agent.work()

        assert isinstance(result, ResearchTask)
        assert result.topic == ""

    @pytest.mark.asyncio
    async def test_findings_accumulate(self, agent_factory, mock_llm_router):
        """Test that findings accumulate during research."""
        agent = agent_factory(ResearcherAgent, topic="Test Topic")
        # Make mock return specific questions
        mock_llm_router.complete.side_effect = [
            type("Response", (), {"content": "Question 1\nQuestion 2\nQuestion 3"})(),
            type("Response", (), {"content": "Answer to Q1"})(),
            type("Response", (), {"content": "Answer to Q2"})(),
            type("Response", (), {"content": "Answer to Q3"})(),
            type("Response", (), {"content": "Summary of findings"})(),
        ]
        await agent.initialize()
        result = await agent.work()

        assert len(result.findings) == 3


class TestOrchestratorAgent:
    """Tests for OrchestratorAgent."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(OrchestratorAgent)
        assert agent.name == "orchestrator"
        assert "planning" in agent.capabilities
        assert len(agent.available_agents) > 0

    def test_constructor_with_params(self, agent_factory):
        """Test initialization with parameters."""
        agent = agent_factory(
            OrchestratorAgent,
            task="Build a web app",
            available_agents=["coder", "tester"],
        )
        assert agent.task == "Build a web app"
        assert agent.available_agents == ["coder", "tester"]

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(OrchestratorAgent, task="Test task")
        await agent.initialize()
        mock_llm_router.initialize.assert_called()

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic orchestration work."""
        agent = agent_factory(OrchestratorAgent, task="Create a simple program")
        await agent.initialize()
        result = await agent.work()

        assert result.task == "Create a simple program"
        assert result.topology in ["swarm", "pipeline", "hierarchy", "mesh", "ring", "star"]

    @pytest.mark.asyncio
    async def test_work_without_task(self, agent_factory):
        """Test work without task."""
        agent = agent_factory(OrchestratorAgent)
        await agent.initialize()
        result = await agent.work()
        assert result.task == ""

    def test_dependency_graph_creation(self):
        """Test dependency graph from subtasks."""
        subtasks = [
            Subtask(id="st-0", description="Research", agent_type="researcher"),
            Subtask(id="st-1", description="Code", agent_type="coder", dependencies=["st-0"]),
            Subtask(id="st-2", description="Review", agent_type="reviewer", dependencies=["st-1"]),
        ]
        graph = DependencyGraph.from_subtasks(subtasks)

        assert len(graph.subtasks) == 3
        assert "st-1" in graph.adjacency["st-0"]
        assert "st-2" in graph.adjacency["st-1"]

    def test_dependency_graph_topological_sort(self):
        """Test topological sorting of subtasks."""
        subtasks = [
            Subtask(id="st-0", description="A", agent_type="researcher"),
            Subtask(id="st-1", description="B", agent_type="coder", dependencies=["st-0"]),
            Subtask(id="st-2", description="C", agent_type="coder", dependencies=["st-0"]),
            Subtask(id="st-3", description="D", agent_type="reviewer", dependencies=["st-1", "st-2"]),
        ]
        graph = DependencyGraph.from_subtasks(subtasks)
        levels = graph.topological_sort()

        assert len(levels) == 3  # A, then B+C, then D
        assert "st-0" in levels[0]
        assert set(levels[1]) == {"st-1", "st-2"}
        assert "st-3" in levels[2]


class TestReviewerAgent:
    """Tests for ReviewerAgent."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(ReviewerAgent)
        assert agent.name == "reviewer"
        assert "code_review" in agent.capabilities

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(ReviewerAgent)
        await agent.initialize()
        mock_llm_router.initialize.assert_called()

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic review work."""
        agent = agent_factory(ReviewerAgent, content="def example(): pass")
        await agent.initialize()
        result = await agent.work()
        assert result is not None

    @pytest.mark.asyncio
    async def test_work_without_content(self, agent_factory):
        """Test work without content to review."""
        agent = agent_factory(ReviewerAgent)
        await agent.initialize()
        result = await agent.work()
        # Should handle gracefully
        assert result is not None


class TestPaper2CodeAgent:
    """Tests for Paper2CodeAgent."""

    def test_constructor_defaults(self, agent_factory):
        """Test default initialization."""
        agent = agent_factory(Paper2CodeAgent)
        assert agent.name == "paper2code"
        assert "paper_analysis" in agent.capabilities

    def test_constructor_with_paper(self, agent_factory):
        """Test initialization with paper reference."""
        agent = agent_factory(
            Paper2CodeAgent,
            paper_url="https://arxiv.org/abs/1234.5678",
            target_language="python",
        )
        assert agent.paper_url == "https://arxiv.org/abs/1234.5678"
        assert agent.target_language == "python"

    @pytest.mark.asyncio
    async def test_initialize(self, agent_factory, mock_llm_router):
        """Test agent initialization."""
        agent = agent_factory(Paper2CodeAgent)
        await agent.initialize()
        mock_llm_router.initialize.assert_called()

    @pytest.mark.asyncio
    async def test_work_basic(self, agent_factory, mock_llm_router):
        """Test basic paper2code work."""
        agent = agent_factory(
            Paper2CodeAgent,
            paper_content="This paper describes a novel algorithm...",
        )
        await agent.initialize()
        result = await agent.work()
        assert result is not None

    @pytest.mark.asyncio
    async def test_work_without_paper(self, agent_factory):
        """Test work without paper content."""
        agent = agent_factory(Paper2CodeAgent)
        await agent.initialize()
        result = await agent.work()
        # Should handle gracefully
        assert result is not None
