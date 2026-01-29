"""
Tests for Model API Routes.

Tests the model signatures and cognitive routing API endpoints.
"""

from __future__ import annotations

import pytest

from titan.workflows.cognitive_router import CognitiveTaskType, MODEL_RANKINGS


# =============================================================================
# Model Signatures Tests
# =============================================================================


class TestModelSignatures:
    """Tests for /api/models/signatures endpoint."""

    @pytest.mark.asyncio
    async def test_get_signatures(self) -> None:
        """Test getting all model signatures."""
        from titan.api.models_routes import get_model_signatures

        data = await get_model_signatures()

        assert "dimensions" in data
        assert "models" in data
        assert len(data["dimensions"]) == 8  # 8 cognitive task types
        assert len(data["models"]) >= 1

    @pytest.mark.asyncio
    async def test_signatures_dimensions(self) -> None:
        """Test that all expected dimensions are present."""
        from titan.api.models_routes import get_model_signatures

        data = await get_model_signatures()

        expected_dimensions = [
            "structured_reasoning",
            "creative_synthesis",
            "mathematical_analysis",
            "cross_domain",
            "meta_analysis",
            "pattern_recognition",
            "code_generation",
            "research_synthesis",
        ]

        for dim in expected_dimensions:
            assert dim in data["dimensions"]

    @pytest.mark.asyncio
    async def test_signatures_model_scores(self) -> None:
        """Test that model scores are within valid range."""
        from titan.api.models_routes import get_model_signatures

        data = await get_model_signatures()

        for model_id, scores in data["models"].items():
            for dimension, score in scores.items():
                assert 0 <= score <= 10, f"Score {score} for {model_id}/{dimension} out of range"


class TestModelSignatureById:
    """Tests for /api/models/signatures/{model_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_known_model(self) -> None:
        """Test getting signature for a known model."""
        from titan.api.models_routes import get_model_signature

        data = await get_model_signature("claude-3-5-sonnet-20241022")

        assert data["model_id"] == "claude-3-5-sonnet-20241022"
        assert data["found"] is True
        assert "scores" in data
        assert "dimensions" in data

    @pytest.mark.asyncio
    async def test_get_unknown_model(self) -> None:
        """Test getting signature for an unknown model."""
        from titan.api.models_routes import get_model_signature

        data = await get_model_signature("unknown-model-xyz")

        assert data["model_id"] == "unknown-model-xyz"
        assert data["found"] is False
        assert data["scores"] == {}


# =============================================================================
# Model List Tests
# =============================================================================


class TestModelList:
    """Tests for /api/models/list endpoint."""

    @pytest.mark.asyncio
    async def test_list_models(self) -> None:
        """Test listing all models."""
        from titan.api.models_routes import list_models

        data = await list_models()

        assert "models" in data
        assert "total" in data
        assert data["total"] >= 1

    @pytest.mark.asyncio
    async def test_model_list_structure(self) -> None:
        """Test model list item structure."""
        from titan.api.models_routes import list_models

        data = await list_models()

        for model in data["models"]:
            assert "model_id" in model
            assert "top_strengths" in model
            assert "average_score" in model
            assert len(model["top_strengths"]) <= 3

    @pytest.mark.asyncio
    async def test_models_sorted_by_average(self) -> None:
        """Test that models are sorted by average score descending."""
        from titan.api.models_routes import list_models

        data = await list_models()

        scores = [m["average_score"] for m in data["models"]]
        assert scores == sorted(scores, reverse=True)


# =============================================================================
# Task Preferences Tests
# =============================================================================


class TestTaskPreferences:
    """Tests for /api/models/preferences endpoint."""

    @pytest.mark.asyncio
    async def test_get_preferences(self) -> None:
        """Test getting task type preferences."""
        from titan.api.models_routes import get_task_preferences

        data = await get_task_preferences()

        assert len(data) == 8  # 8 cognitive task types

    @pytest.mark.asyncio
    async def test_preference_structure(self) -> None:
        """Test preference structure for each task type."""
        from titan.api.models_routes import get_task_preferences

        data = await get_task_preferences()

        for task_type, prefs in data.items():
            assert "preferred" in prefs
            assert "alternatives" in prefs
            assert isinstance(prefs["alternatives"], list)


# =============================================================================
# Model Comparison Tests
# =============================================================================


class TestModelComparison:
    """Tests for /api/models/compare endpoint."""

    @pytest.mark.asyncio
    async def test_compare_models(self) -> None:
        """Test comparing two models."""
        from titan.api.models_routes import compare_models

        data = await compare_models(
            model_a="claude-3-5-sonnet-20241022",
            model_b="gpt-4-turbo",
        )

        assert data["model_a"] == "claude-3-5-sonnet-20241022"
        assert data["model_b"] == "gpt-4-turbo"
        assert "dimensions" in data
        assert "summary" in data

    @pytest.mark.asyncio
    async def test_comparison_dimensions(self) -> None:
        """Test comparison includes all dimensions."""
        from titan.api.models_routes import compare_models

        data = await compare_models(
            model_a="claude-3-5-sonnet-20241022",
            model_b="gpt-4-turbo",
        )

        assert len(data["dimensions"]) == 8

        for dim in data["dimensions"]:
            assert "dimension" in dim
            assert "model_a_score" in dim
            assert "model_b_score" in dim
            assert "difference" in dim
            assert "winner" in dim

    @pytest.mark.asyncio
    async def test_comparison_summary(self) -> None:
        """Test comparison summary structure."""
        from titan.api.models_routes import compare_models

        data = await compare_models(
            model_a="claude-3-5-sonnet-20241022",
            model_b="gpt-4-turbo",
        )

        summary = data["summary"]
        assert "model_a_wins" in summary
        assert "model_b_wins" in summary
        assert "ties" in summary
        assert "overall_winner" in summary

        # Wins + ties should equal total dimensions
        assert summary["model_a_wins"] + summary["model_b_wins"] + summary["ties"] == 8

    @pytest.mark.asyncio
    async def test_compare_unknown_model(self) -> None:
        """Test comparing with an unknown model."""
        from titan.api.models_routes import compare_models

        data = await compare_models(
            model_a="claude-3-5-sonnet-20241022",
            model_b="unknown-model",
        )

        # Unknown model should have 0 scores
        for dim in data["dimensions"]:
            assert dim["model_b_score"] == 0.0


# =============================================================================
# Routing Tests
# =============================================================================


class TestModelRouting:
    """Tests for /api/models/route endpoint."""

    @pytest.mark.asyncio
    async def test_route_for_task(self) -> None:
        """Test routing for a cognitive task type."""
        from titan.api.models_routes import route_for_task

        data = await route_for_task(task_type="structured_reasoning")

        assert "model_id" in data
        assert "task_type" in data
        assert "score" in data
        assert "reasoning" in data

    @pytest.mark.asyncio
    async def test_route_with_preferred_model(self) -> None:
        """Test routing with a preferred model override."""
        from titan.api.models_routes import route_for_task

        data = await route_for_task(
            task_type="creative_synthesis",
            preferred_model="gemini-pro",
        )

        assert data["model_id"] == "gemini-pro"

    @pytest.mark.asyncio
    async def test_route_invalid_task_type(self) -> None:
        """Test routing with invalid task type."""
        from titan.api.models_routes import route_for_task

        data = await route_for_task(task_type="invalid_task_type")

        assert "error" in data
        assert "valid_types" in data

    @pytest.mark.asyncio
    async def test_route_all_task_types(self) -> None:
        """Test routing for all valid task types."""
        from titan.api.models_routes import route_for_task

        task_types = [
            "structured_reasoning",
            "creative_synthesis",
            "mathematical_analysis",
            "cross_domain",
            "meta_analysis",
            "pattern_recognition",
            "code_generation",
            "research_synthesis",
        ]

        for task_type in task_types:
            data = await route_for_task(task_type=task_type)
            assert "model_id" in data
            assert data["task_type"] == task_type
