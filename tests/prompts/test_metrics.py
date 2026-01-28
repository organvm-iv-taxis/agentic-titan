"""
Tests for titan.prompts.metrics module.
"""

import pytest
from datetime import datetime, timedelta

from titan.prompts.metrics import (
    AggregatedMetrics,
    MetricAggregation,
    OptimizationRecommendation,
    PromptMetrics,
    PromptTracker,
    get_prompt_tracker,
)


class TestPromptMetrics:
    """Tests for PromptMetrics dataclass."""

    def test_total_tokens_calculation(self):
        """Test total tokens property."""
        metrics = PromptMetrics(
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert metrics.total_tokens == 150

    def test_efficiency_ratio_calculation(self):
        """Test efficiency ratio property."""
        metrics = PromptMetrics(
            prompt_tokens=100,
            completion_tokens=100,
            quality_score=0.8,
        )
        # efficiency = (0.8 * 1000) / 200 = 4.0
        assert metrics.efficiency_ratio == pytest.approx(4.0, rel=0.01)

    def test_efficiency_ratio_zero_tokens(self):
        """Test efficiency ratio with zero tokens."""
        metrics = PromptMetrics(
            prompt_tokens=0,
            completion_tokens=0,
            quality_score=0.8,
        )
        assert metrics.efficiency_ratio == 0.0

    def test_cost_efficiency_calculation(self):
        """Test cost efficiency property."""
        metrics = PromptMetrics(
            quality_score=0.8,
            cost_usd=0.01,
        )
        # cost_efficiency = 0.8 / 0.01 = 80.0
        assert metrics.cost_efficiency == pytest.approx(80.0, rel=0.01)

    def test_cost_efficiency_zero_cost(self):
        """Test cost efficiency with zero cost."""
        metrics = PromptMetrics(
            quality_score=0.8,
            cost_usd=0.0,
        )
        assert metrics.cost_efficiency == 0.0

    def test_to_dict(self):
        """Test serialization to dict."""
        metrics = PromptMetrics(
            stage_name="test_stage",
            model="claude-3-sonnet",
            prompt_tokens=100,
            completion_tokens=50,
            quality_score=0.75,
        )
        result = metrics.to_dict()

        assert result["stage_name"] == "test_stage"
        assert result["model"] == "claude-3-sonnet"
        assert result["total_tokens"] == 150
        assert result["efficiency_ratio"] > 0


class TestAggregatedMetrics:
    """Tests for AggregatedMetrics dataclass."""

    def test_default_values(self):
        """Test default aggregated metrics."""
        agg = AggregatedMetrics()
        assert agg.execution_count == 0
        assert agg.avg_quality_score == 0.0
        assert agg.total_cost_usd == 0.0


class TestOptimizationRecommendation:
    """Tests for OptimizationRecommendation dataclass."""

    def test_recommendation_creation(self):
        """Test creating a recommendation."""
        rec = OptimizationRecommendation(
            recommendation_type="low_efficiency",
            priority="high",
            description="Consider using concise prompts",
            expected_improvement="20% token reduction",
            affected_stages=["scope_clarification"],
        )
        assert rec.priority == "high"
        assert "scope_clarification" in rec.affected_stages


class TestPromptTracker:
    """Tests for PromptTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker for each test."""
        return PromptTracker(min_samples_for_stats=3)

    def test_record_metrics(self, tracker):
        """Test recording metrics."""
        metrics = tracker.record(
            stage_name="scope_clarification",
            model="claude-3-sonnet",
            prompt_tokens=100,
            completion_tokens=50,
            quality_score=0.8,
            latency_ms=500,
            cost_usd=0.005,
        )

        assert metrics.stage_name == "scope_clarification"
        assert metrics.model == "claude-3-sonnet"
        assert metrics.quality_score == 0.8

    def test_get_efficiency_ratio(self, tracker):
        """Test getting efficiency ratio."""
        # Record multiple metrics
        for i in range(5):
            tracker.record(
                stage_name="test_stage",
                model="claude-3-sonnet",
                prompt_tokens=100,
                completion_tokens=50,
                quality_score=0.7 + i * 0.05,
                latency_ms=500,
                cost_usd=0.005,
            )

        ratio = tracker.get_efficiency_ratio(stage_name="test_stage")
        assert ratio > 0

    def test_get_efficiency_ratio_no_data(self, tracker):
        """Test efficiency ratio with no data."""
        ratio = tracker.get_efficiency_ratio(stage_name="nonexistent")
        assert ratio == 0.0

    def test_get_aggregated_metrics(self, tracker):
        """Test aggregated metrics calculation."""
        # Record metrics for same stage
        for i in range(5):
            tracker.record(
                stage_name="test_stage",
                model="claude-3-sonnet",
                prompt_tokens=100 + i * 10,
                completion_tokens=50,
                quality_score=0.7 + i * 0.05,
                latency_ms=500 + i * 100,
                cost_usd=0.005,
            )

        agg = tracker.get_aggregated_metrics(stage_name="test_stage")

        assert agg.execution_count == 5
        assert agg.avg_quality_score > 0
        assert agg.total_cost_usd == pytest.approx(0.025, rel=0.01)

    def test_get_aggregated_metrics_no_data(self, tracker):
        """Test aggregated metrics with no data."""
        agg = tracker.get_aggregated_metrics(stage_name="nonexistent")
        assert agg.execution_count == 0

    def test_recommend_optimizations_insufficient_data(self, tracker):
        """Test recommendations with insufficient data."""
        recommendations = tracker.recommend_optimizations()

        assert len(recommendations) == 1
        assert recommendations[0].recommendation_type == "insufficient_data"

    def test_recommend_optimizations_low_efficiency(self, tracker):
        """Test recommendations for low efficiency."""
        # Record metrics with low efficiency
        for i in range(5):
            tracker.record(
                stage_name="test_stage",
                model="claude-3-sonnet",
                prompt_tokens=1000,  # High input
                completion_tokens=50,  # Low output
                quality_score=0.3,  # Low quality
                latency_ms=500,
                cost_usd=0.01,
            )

        recommendations = tracker.recommend_optimizations()

        # Should recommend improving efficiency
        types = [r.recommendation_type for r in recommendations]
        assert "low_efficiency_stage" in types or "prompt_bloat" in types

    def test_compare_variants(self, tracker):
        """Test comparing prompt variants."""
        # Record variant A
        for i in range(5):
            tracker.record(
                stage_name="test_stage",
                model="claude-3-sonnet",
                prompt_tokens=100,
                completion_tokens=50,
                quality_score=0.7,
                latency_ms=500,
                cost_usd=0.005,
                prompt_variant="variant_a",
            )

        # Record variant B (better)
        for i in range(5):
            tracker.record(
                stage_name="test_stage",
                model="claude-3-sonnet",
                prompt_tokens=80,
                completion_tokens=50,
                quality_score=0.9,
                latency_ms=400,
                cost_usd=0.004,
                prompt_variant="variant_b",
            )

        comparison = tracker.compare_variants("variant_a", "variant_b")

        assert comparison["quality_winner"] == "variant_b"
        assert comparison["variant_b"]["avg_quality"] > comparison["variant_a"]["avg_quality"]

    def test_compare_variants_insufficient_data(self, tracker):
        """Test comparison with insufficient data."""
        tracker.record(
            stage_name="test_stage",
            model="claude-3-sonnet",
            prompt_tokens=100,
            completion_tokens=50,
            quality_score=0.7,
            latency_ms=500,
            cost_usd=0.005,
            prompt_variant="variant_a",
        )

        comparison = tracker.compare_variants("variant_a", "variant_b")
        assert "error" in comparison

    def test_get_stage_summary(self, tracker):
        """Test stage summary."""
        # Record metrics for multiple stages
        for stage in ["scope_clarification", "logical_branching"]:
            for i in range(3):
                tracker.record(
                    stage_name=stage,
                    model="claude-3-sonnet",
                    prompt_tokens=100,
                    completion_tokens=50,
                    quality_score=0.8,
                    latency_ms=500,
                    cost_usd=0.005,
                )

        summary = tracker.get_stage_summary()

        assert "scope_clarification" in summary
        assert "logical_branching" in summary
        assert summary["scope_clarification"]["count"] == 3

    def test_prune_old_metrics(self, tracker):
        """Test pruning old metrics."""
        tracker = PromptTracker(retention_days=1)

        # Record a metric
        tracker.record(
            stage_name="test_stage",
            model="claude-3-sonnet",
            prompt_tokens=100,
            completion_tokens=50,
            quality_score=0.8,
            latency_ms=500,
            cost_usd=0.005,
        )

        # Manually set old timestamp
        tracker._metrics[0].timestamp = datetime.utcnow() - timedelta(days=2)

        pruned = tracker.prune_old_metrics()
        assert pruned == 1
        assert len(tracker._metrics) == 0

    def test_filter_by_model(self, tracker):
        """Test filtering metrics by model."""
        # Record for different models
        for model in ["claude-3-sonnet", "gpt-4o"]:
            for i in range(3):
                tracker.record(
                    stage_name="test_stage",
                    model=model,
                    prompt_tokens=100,
                    completion_tokens=50,
                    quality_score=0.8,
                    latency_ms=500,
                    cost_usd=0.005,
                )

        agg = tracker.get_aggregated_metrics(model="claude-3-sonnet")
        assert agg.execution_count == 3


class TestGetPromptTracker:
    """Tests for get_prompt_tracker singleton."""

    def test_returns_same_instance(self):
        """Test that get_prompt_tracker returns singleton."""
        tracker1 = get_prompt_tracker()
        tracker2 = get_prompt_tracker()
        assert tracker1 is tracker2

    def test_is_prompt_tracker(self):
        """Test that returned instance is PromptTracker."""
        tracker = get_prompt_tracker()
        assert isinstance(tracker, PromptTracker)
