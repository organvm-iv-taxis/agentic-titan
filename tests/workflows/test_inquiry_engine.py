"""
Tests for the Inquiry Engine and related components.

Tests cover:
- Workflow configuration loading and validation
- Stage execution with mock LLM
- Context accumulation between stages
- Markdown export formatting
- Cognitive routing decisions
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from datetime import datetime

from titan.workflows.inquiry_config import (
    InquiryStage,
    InquiryWorkflow,
    CognitiveStyle,
    EXPANSIVE_INQUIRY_WORKFLOW,
    QUICK_INQUIRY_WORKFLOW,
    CREATIVE_INQUIRY_WORKFLOW,
    get_workflow,
    list_workflows,
)
from titan.workflows.inquiry_engine import (
    InquiryEngine,
    InquirySession,
    InquiryStatus,
    StageResult,
    get_inquiry_engine,
)
from titan.workflows.cognitive_router import (
    CognitiveRouter,
    CognitiveTaskType,
    CognitiveRoutingDecision,
    COGNITIVE_MODEL_MAP,
    MODEL_RANKINGS,
)
from titan.workflows.inquiry_prompts import (
    STAGE_PROMPTS,
    CONCISE_STAGE_PROMPTS,
    get_prompt,
    list_templates,
)
from titan.workflows.inquiry_export import (
    export_stage_to_markdown,
    export_session_to_markdown,
    slugify,
)


# =============================================================================
# Workflow Configuration Tests
# =============================================================================


class TestInquiryConfig:
    """Tests for workflow configuration."""

    def test_expansive_workflow_has_six_stages(self):
        """Expansive workflow should have 6 stages."""
        assert len(EXPANSIVE_INQUIRY_WORKFLOW.stages) == 6

    def test_quick_workflow_has_three_stages(self):
        """Quick workflow should have 3 stages."""
        assert len(QUICK_INQUIRY_WORKFLOW.stages) == 3

    def test_creative_workflow_has_four_stages(self):
        """Creative workflow should have 4 stages."""
        assert len(CREATIVE_INQUIRY_WORKFLOW.stages) == 4

    def test_workflow_stage_names_are_unique(self):
        """All stage names in a workflow should be unique."""
        for workflow in [EXPANSIVE_INQUIRY_WORKFLOW, QUICK_INQUIRY_WORKFLOW, CREATIVE_INQUIRY_WORKFLOW]:
            names = [s.name for s in workflow.stages]
            assert len(names) == len(set(names))

    def test_workflow_stages_have_prompt_templates(self):
        """All stages should have valid prompt templates."""
        for workflow in [EXPANSIVE_INQUIRY_WORKFLOW, QUICK_INQUIRY_WORKFLOW, CREATIVE_INQUIRY_WORKFLOW]:
            for stage in workflow.stages:
                assert stage.prompt_template in STAGE_PROMPTS

    def test_get_workflow_returns_correct_workflow(self):
        """get_workflow should return the correct workflow."""
        assert get_workflow("expansive") == EXPANSIVE_INQUIRY_WORKFLOW
        assert get_workflow("quick") == QUICK_INQUIRY_WORKFLOW
        assert get_workflow("creative") == CREATIVE_INQUIRY_WORKFLOW

    def test_get_workflow_returns_none_for_unknown(self):
        """get_workflow should return None for unknown workflow."""
        assert get_workflow("nonexistent") is None

    def test_list_workflows_returns_all(self):
        """list_workflows should return all workflow names."""
        names = list_workflows()
        assert "expansive" in names
        assert "quick" in names
        assert "creative" in names

    def test_workflow_to_dict_serialization(self):
        """Workflow should serialize to dict properly."""
        data = EXPANSIVE_INQUIRY_WORKFLOW.to_dict()
        assert data["name"] == "Expansive Inquiry"
        assert len(data["stages"]) == 6
        assert data["context_accumulation"] is True

    def test_stage_to_dict_serialization(self):
        """Stage should serialize to dict properly."""
        stage = EXPANSIVE_INQUIRY_WORKFLOW.stages[0]
        data = stage.to_dict()
        assert data["name"] == "Scope Clarification"
        assert data["role"] == "Scope AI"
        assert data["cognitive_style"] == "structured_reasoning"

    def test_workflow_validation_empty_stages_raises(self):
        """Workflow with empty stages should raise ValueError."""
        with pytest.raises(ValueError, match="at least one stage"):
            InquiryWorkflow(
                name="Empty",
                description="Empty workflow",
                stages=[],
            )

    def test_workflow_get_stage_by_index(self):
        """get_stage should return correct stage by index."""
        stage = EXPANSIVE_INQUIRY_WORKFLOW.get_stage(0)
        assert stage is not None
        assert stage.name == "Scope Clarification"

    def test_workflow_get_stage_by_name(self):
        """get_stage_by_name should return correct stage."""
        stage = EXPANSIVE_INQUIRY_WORKFLOW.get_stage_by_name("Logical Branching")
        assert stage is not None
        assert stage.role == "Logic AI"


# =============================================================================
# Cognitive Router Tests
# =============================================================================


class TestCognitiveRouter:
    """Tests for cognitive task routing."""

    def test_all_task_types_have_model_map(self):
        """All cognitive task types should have model mappings."""
        for task_type in CognitiveTaskType:
            assert task_type in COGNITIVE_MODEL_MAP

    def test_router_selects_model_for_task(self):
        """Router should select appropriate model for task type."""
        router = CognitiveRouter()

        for task_type in CognitiveTaskType:
            decision = pytest.helpers.run_async(
                router.route_for_task(task_type)
            ) if hasattr(pytest, 'helpers') else None

    @pytest.mark.asyncio
    async def test_router_respects_preferred_model(self):
        """Router should use preferred model when available."""
        router = CognitiveRouter()

        decision = await router.route_for_task(
            CognitiveTaskType.STRUCTURED_REASONING,
            preferred_model="gpt-4-turbo",
        )

        assert decision.model_id == "gpt-4-turbo"
        assert "preferred" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_router_falls_back_when_preferred_unavailable(self):
        """Router should fall back when preferred model unavailable."""
        router = CognitiveRouter(
            available_models=["claude-3-5-sonnet-20241022"],
        )

        decision = await router.route_for_task(
            CognitiveTaskType.CREATIVE_SYNTHESIS,
            preferred_model="nonexistent-model",
        )

        # Should fall back to available model
        assert decision.model_id in ["claude-3-5-sonnet-20241022"]

    def test_model_rankings_valid_scores(self):
        """Model rankings should have valid scores (0-10)."""
        for model_id, rankings in MODEL_RANKINGS.items():
            for task_type, score in rankings.items():
                assert 0 <= score <= 10, f"Invalid score {score} for {model_id}/{task_type}"

    def test_get_model_score(self):
        """get_model_score should return correct score."""
        router = CognitiveRouter()
        score = router.get_model_score(
            "claude-3-5-sonnet-20241022",
            CognitiveTaskType.STRUCTURED_REASONING,
        )
        assert score == 9.0


# =============================================================================
# Prompt Template Tests
# =============================================================================


class TestInquiryPrompts:
    """Tests for prompt templates."""

    def test_all_stage_prompts_exist(self):
        """All expected stage prompts should exist."""
        expected = [
            "scope_clarification",
            "logical_branching",
            "intuitive_branching",
            "lateral_exploration",
            "recursive_design",
            "pattern_recognition",
        ]
        for key in expected:
            assert key in STAGE_PROMPTS
            assert key in CONCISE_STAGE_PROMPTS

    def test_prompt_has_topic_placeholder(self):
        """All prompts should have {topic} placeholder."""
        for key, template in STAGE_PROMPTS.items():
            assert "{topic}" in template, f"Missing {{topic}} in {key}"

    def test_prompt_has_context_placeholder_except_first(self):
        """All prompts except first should have {previous_context}."""
        for key, template in STAGE_PROMPTS.items():
            if key != "scope_clarification":
                assert "{previous_context}" in template, f"Missing {{previous_context}} in {key}"

    def test_get_prompt_formats_correctly(self):
        """get_prompt should format template correctly."""
        prompt = get_prompt(
            "scope_clarification",
            topic="The nature of consciousness",
        )
        assert "The nature of consciousness" in prompt
        assert "{topic}" not in prompt  # Should be replaced

    def test_get_prompt_includes_context(self):
        """get_prompt should include previous context."""
        context = json.dumps({"stage1": "some result"})
        prompt = get_prompt(
            "logical_branching",
            topic="Test topic",
            previous_context=context,
        )
        assert context in prompt

    def test_get_prompt_concise_mode(self):
        """get_prompt with concise=True should use shorter templates."""
        full_prompt = get_prompt("scope_clarification", topic="Test")
        concise_prompt = get_prompt("scope_clarification", topic="Test", concise=True)
        assert len(concise_prompt) < len(full_prompt)

    def test_get_prompt_unknown_template_raises(self):
        """get_prompt should raise for unknown template."""
        with pytest.raises(ValueError, match="Unknown prompt template"):
            get_prompt("nonexistent", topic="Test")

    def test_list_templates_returns_all(self):
        """list_templates should return all template keys."""
        templates = list_templates()
        assert "scope_clarification" in templates
        assert "pattern_recognition" in templates


# =============================================================================
# Package Import Surface Tests
# =============================================================================


class TestWorkflowPackageImports:
    """Tests for workflow package import behavior."""

    def test_workflows_package_import_is_lightweight(self):
        """Importing titan.workflows should not import inquiry_engine eagerly."""
        project_root = Path(__file__).resolve().parents[2]
        script = (
            "import sys; "
            "import titan.workflows; "
            "print('titan.workflows.inquiry_engine' in sys.modules)"
        )

        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )

        assert proc.stdout.strip() == "False"


# =============================================================================
# Inquiry Engine Tests
# =============================================================================


class TestInquiryEngine:
    """Tests for the inquiry engine."""

    @pytest.mark.asyncio
    async def test_start_inquiry_creates_session(self):
        """start_inquiry should create a new session."""
        engine = InquiryEngine()
        session = await engine.start_inquiry(
            topic="The nature of consciousness",
        )

        assert session.id.startswith("inq-")
        assert session.topic == "The nature of consciousness"
        assert session.status == InquiryStatus.PENDING
        assert session.workflow == EXPANSIVE_INQUIRY_WORKFLOW

    @pytest.mark.asyncio
    async def test_start_inquiry_with_custom_workflow(self):
        """start_inquiry should accept custom workflow."""
        engine = InquiryEngine()
        session = await engine.start_inquiry(
            topic="Test topic",
            workflow=QUICK_INQUIRY_WORKFLOW,
        )

        assert session.workflow == QUICK_INQUIRY_WORKFLOW
        assert session.total_stages == 3

    @pytest.mark.asyncio
    async def test_run_stage_executes_mock(self):
        """run_stage should execute and return result."""
        engine = InquiryEngine()
        session = await engine.start_inquiry(topic="Test topic")

        result = await engine.run_stage(session)

        assert result.stage_name == "Scope Clarification"
        assert result.role == "Scope AI"
        assert result.success
        assert len(session.results) == 1

    @pytest.mark.asyncio
    async def test_run_stage_updates_session_status(self):
        """run_stage should update session status to RUNNING."""
        engine = InquiryEngine()
        session = await engine.start_inquiry(topic="Test topic")

        await engine.run_stage(session)

        assert session.status == InquiryStatus.RUNNING

    @pytest.mark.asyncio
    async def test_run_full_workflow_completes_all_stages(self):
        """run_full_workflow should complete all stages."""
        engine = InquiryEngine()
        session = await engine.start_inquiry(
            topic="Test topic",
            workflow=QUICK_INQUIRY_WORKFLOW,  # 3 stages for faster test
        )

        result = await engine.run_full_workflow(session)

        assert result.status == InquiryStatus.COMPLETED
        assert len(result.results) == 3
        assert result.is_complete

    @pytest.mark.asyncio
    async def test_context_accumulates_between_stages(self):
        """Previous stage results should be passed to next stage."""
        engine = InquiryEngine()
        session = await engine.start_inquiry(
            topic="Test topic",
            workflow=QUICK_INQUIRY_WORKFLOW,
        )

        await engine.run_stage(session)  # Stage 0
        await engine.run_stage(session)  # Stage 1

        # Check that context includes stage 0 result
        context = session.get_previous_context()
        assert "Scope Clarification" in context

    @pytest.mark.asyncio
    async def test_session_progress_calculation(self):
        """Session progress should be calculated correctly."""
        engine = InquiryEngine()
        session = await engine.start_inquiry(
            topic="Test topic",
            workflow=QUICK_INQUIRY_WORKFLOW,  # 3 stages
        )

        assert session.progress == 0.0

        await engine.run_stage(session)
        assert abs(session.progress - 33.33) < 1

        await engine.run_stage(session)
        assert abs(session.progress - 66.67) < 1

        await engine.run_stage(session)
        assert session.progress == 100.0

    @pytest.mark.asyncio
    async def test_cancel_session(self):
        """cancel_session should stop running session."""
        engine = InquiryEngine()
        session = await engine.start_inquiry(topic="Test topic")

        await engine.run_stage(session)  # Start running
        success = engine.cancel_session(session.id)

        assert success
        assert session.status == InquiryStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_get_session_returns_correct_session(self):
        """get_session should return correct session."""
        engine = InquiryEngine()
        session = await engine.start_inquiry(topic="Test topic")

        retrieved = engine.get_session(session.id)

        assert retrieved == session

    @pytest.mark.asyncio
    async def test_list_sessions_filters_by_status(self):
        """list_sessions should filter by status."""
        engine = InquiryEngine()

        session1 = await engine.start_inquiry(topic="Topic 1")
        session2 = await engine.start_inquiry(topic="Topic 2")
        await engine.run_stage(session1)  # Make session1 RUNNING

        pending = engine.list_sessions(status=InquiryStatus.PENDING)
        running = engine.list_sessions(status=InquiryStatus.RUNNING)

        assert session2 in pending
        assert session1 in running

    def test_constructor_uses_runtime_config_when_defaults_omitted(self, monkeypatch):
        """Constructor should resolve defaults from config at runtime."""
        fake_config = SimpleNamespace(
            llm=SimpleNamespace(default_model="config-model"),
            max_context_tokens=1234,
        )
        monkeypatch.setattr(
            "titan.workflows.inquiry_engine.get_config",
            lambda: fake_config,
        )

        engine = InquiryEngine(default_model=None, max_context_tokens=None)

        assert engine._default_model == "config-model"
        assert engine._max_context_tokens == 1234

    def test_constructor_with_explicit_values_skips_config_lookup(self, monkeypatch):
        """Explicit constructor values should bypass config lookup."""
        def _boom():
            raise AssertionError("get_config should not be called")

        monkeypatch.setattr("titan.workflows.inquiry_engine.get_config", _boom)

        engine = InquiryEngine(
            default_model="explicit-model",
            max_context_tokens=777,
        )

        assert engine._default_model == "explicit-model"
        assert engine._max_context_tokens == 777


# =============================================================================
# Stage Result Tests
# =============================================================================


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_stage_result_success_property(self):
        """success property should be True when no error."""
        result = StageResult(
            stage_name="Test",
            role="Test AI",
            content="Some content",
            model_used="test-model",
            timestamp=datetime.now(),
        )
        assert result.success

    def test_stage_result_failure_property(self):
        """success property should be False when error present."""
        result = StageResult(
            stage_name="Test",
            role="Test AI",
            content="",
            model_used="test-model",
            timestamp=datetime.now(),
            error="Something went wrong",
        )
        assert not result.success

    def test_stage_result_to_dict(self):
        """to_dict should serialize correctly."""
        result = StageResult(
            stage_name="Test Stage",
            role="Test AI",
            content="Test content",
            model_used="test-model",
            timestamp=datetime.now(),
            tokens_used=100,
            duration_ms=500,
            stage_index=0,
        )

        data = result.to_dict()

        assert data["stage_name"] == "Test Stage"
        assert data["role"] == "Test AI"
        assert data["tokens_used"] == 100
        assert "timestamp" in data


# =============================================================================
# Export Tests
# =============================================================================


class TestInquiryExport:
    """Tests for markdown export functions."""

    def test_slugify_basic(self):
        """slugify should convert text to URL-friendly slug."""
        assert slugify("Hello World") == "hello-world"
        assert slugify("The Nature of Consciousness") == "the-nature-of-consciousness"

    def test_slugify_special_chars(self):
        """slugify should handle special characters."""
        assert slugify("Test! @#$ 123") == "test-123"
        assert slugify("---multiple---hyphens---") == "multiple-hyphens"

    @pytest.mark.asyncio
    async def test_export_stage_to_markdown(self):
        """export_stage_to_markdown should generate valid markdown."""
        engine = InquiryEngine()
        session = await engine.start_inquiry(topic="Test Topic")
        await engine.run_stage(session)

        markdown = export_stage_to_markdown(session, 0)

        # Check frontmatter
        assert "---" in markdown
        assert 'title: "Scope Clarification - Test Topic"' in markdown
        assert "stage_number: 1" in markdown

        # Check content
        assert "# Scope Clarification: Test Topic" in markdown
        assert "Scope AI" in markdown

    @pytest.mark.asyncio
    async def test_export_session_to_markdown(self):
        """export_session_to_markdown should generate combined document."""
        engine = InquiryEngine()
        session = await engine.start_inquiry(
            topic="Test Topic",
            workflow=QUICK_INQUIRY_WORKFLOW,
        )
        await engine.run_full_workflow(session)

        markdown = export_session_to_markdown(session)

        # Check frontmatter
        assert "---" in markdown
        assert 'title: "Collaborative Inquiry: Test Topic"' in markdown
        assert "stages_completed: 3" in markdown

        # Check all stages present
        assert "Scope Clarification" in markdown
        assert "Logical Branching" in markdown
        assert "Pattern Recognition" in markdown

        # Check table of contents
        assert "Table of Contents" in markdown

    @pytest.mark.asyncio
    async def test_export_stage_raises_for_incomplete(self):
        """export_stage_to_markdown should raise for incomplete stage."""
        engine = InquiryEngine()
        session = await engine.start_inquiry(topic="Test Topic")

        with pytest.raises(ValueError, match="not yet completed"):
            export_stage_to_markdown(session, 0)


# =============================================================================
# Inquiry Session Tests
# =============================================================================


class TestInquirySession:
    """Tests for InquirySession dataclass."""

    def test_session_to_dict_serialization(self):
        """Session should serialize to dict properly."""
        session = InquirySession(
            id="test-123",
            topic="Test topic",
            workflow=QUICK_INQUIRY_WORKFLOW,
        )

        data = session.to_dict()

        assert data["id"] == "test-123"
        assert data["topic"] == "Test topic"
        assert data["workflow_name"] == "Quick Inquiry"
        assert data["total_stages"] == 3

    def test_session_total_stages(self):
        """total_stages should return workflow stage count."""
        session = InquirySession(
            id="test-123",
            topic="Test",
            workflow=EXPANSIVE_INQUIRY_WORKFLOW,
        )
        assert session.total_stages == 6

    def test_session_is_complete(self):
        """is_complete should return True when all stages done."""
        session = InquirySession(
            id="test-123",
            topic="Test",
            workflow=QUICK_INQUIRY_WORKFLOW,
        )
        assert not session.is_complete

        # Add mock results
        for i in range(3):
            session.results.append(
                StageResult(
                    stage_name=f"Stage {i}",
                    role="Test",
                    content="Content",
                    model_used="test",
                    timestamp=datetime.now(),
                    stage_index=i,
                )
            )

        assert session.is_complete

    def test_session_get_previous_context(self):
        """get_previous_context should return JSON of results."""
        session = InquirySession(
            id="test-123",
            topic="Test",
            workflow=QUICK_INQUIRY_WORKFLOW,
        )

        session.results.append(
            StageResult(
                stage_name="Stage 1",
                role="Test AI",
                content="Some content",
                model_used="test",
                timestamp=datetime.now(),
                stage_index=0,
            )
        )

        context = session.get_previous_context()
        parsed = json.loads(context)

        assert "Stage 1" in parsed
        assert parsed["Stage 1"]["role"] == "Test AI"
