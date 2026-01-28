"""
Titan Prompts - Prompt Effectiveness Metrics

Tracks and analyzes prompt performance to enable data-driven optimization.

Features:
- Per-prompt token and cost tracking
- Quality score correlation
- Efficiency ratio calculation
- Optimization recommendations

Based on research:
- Build evaluation datasets (30+ cases per agent)
- Measure prompt ROI (quality per token)
- A/B testing framework for prompt variants
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

logger = logging.getLogger("titan.prompts.metrics")


class MetricAggregation(str, Enum):
    """Aggregation methods for metrics."""

    MEAN = "mean"
    MEDIAN = "median"
    P95 = "p95"
    SUM = "sum"
    COUNT = "count"


@dataclass
class PromptMetrics:
    """Metrics for a single prompt execution."""

    id: UUID = field(default_factory=uuid4)
    prompt_id: str = ""
    stage_name: str = ""
    model: str = ""

    # Token counts
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    # Quality metrics
    quality_score: float = 0.0  # 0-1, from evaluator
    relevance_score: float = 0.0  # 0-1
    coherence_score: float = 0.0  # 0-1

    # Performance metrics
    latency_ms: int = 0
    cost_usd: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    prompt_variant: str = "default"  # For A/B testing
    adaptations_applied: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def efficiency_ratio(self) -> float:
        """Quality per 1000 tokens (higher is better)."""
        if self.total_tokens == 0:
            return 0.0
        return (self.quality_score * 1000) / self.total_tokens

    @property
    def cost_efficiency(self) -> float:
        """Quality per dollar spent (higher is better)."""
        if self.cost_usd == 0:
            return 0.0
        return self.quality_score / self.cost_usd

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "prompt_id": self.prompt_id,
            "stage_name": self.stage_name,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "quality_score": self.quality_score,
            "relevance_score": self.relevance_score,
            "coherence_score": self.coherence_score,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "efficiency_ratio": self.efficiency_ratio,
            "cost_efficiency": self.cost_efficiency,
            "timestamp": self.timestamp.isoformat(),
            "prompt_variant": self.prompt_variant,
            "adaptations_applied": self.adaptations_applied,
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for a stage or model."""

    stage_name: str = ""
    model: str = ""
    variant: str = ""

    # Counts
    execution_count: int = 0

    # Token statistics
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    total_tokens: int = 0

    # Quality statistics
    avg_quality_score: float = 0.0
    median_quality_score: float = 0.0
    quality_std_dev: float = 0.0

    # Efficiency statistics
    avg_efficiency_ratio: float = 0.0
    avg_cost_efficiency: float = 0.0

    # Cost statistics
    total_cost_usd: float = 0.0
    avg_cost_usd: float = 0.0

    # Performance statistics
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


@dataclass
class OptimizationRecommendation:
    """A recommendation for prompt optimization."""

    recommendation_type: str
    priority: str  # "high", "medium", "low"
    description: str
    expected_improvement: str
    affected_stages: list[str] = field(default_factory=list)
    metrics_basis: dict[str, Any] = field(default_factory=dict)


class PromptTracker:
    """
    Tracks prompt effectiveness and generates optimization recommendations.

    Features:
    - Records metrics for every prompt execution
    - Aggregates by stage, model, and variant
    - Calculates efficiency ratios
    - Generates data-driven recommendations
    """

    def __init__(
        self,
        retention_days: int = 30,
        min_samples_for_stats: int = 5,
    ) -> None:
        """
        Initialize prompt tracker.

        Args:
            retention_days: Days to retain metrics
            min_samples_for_stats: Minimum samples for statistical analysis
        """
        self.retention_days = retention_days
        self.min_samples_for_stats = min_samples_for_stats

        # Storage
        self._metrics: list[PromptMetrics] = []
        self._by_stage: dict[str, list[PromptMetrics]] = defaultdict(list)
        self._by_model: dict[str, list[PromptMetrics]] = defaultdict(list)
        self._by_variant: dict[str, list[PromptMetrics]] = defaultdict(list)

    def record(
        self,
        stage_name: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        quality_score: float,
        latency_ms: int,
        cost_usd: float,
        cached_tokens: int = 0,
        relevance_score: float = 0.0,
        coherence_score: float = 0.0,
        prompt_variant: str = "default",
        adaptations_applied: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PromptMetrics:
        """
        Record metrics for a prompt execution.

        Args:
            stage_name: Name of the inquiry stage
            model: Model used
            prompt_tokens: Input token count
            completion_tokens: Output token count
            quality_score: Quality score (0-1)
            latency_ms: Response latency in ms
            cost_usd: Cost in USD
            cached_tokens: Cached token count
            relevance_score: Relevance score (0-1)
            coherence_score: Coherence score (0-1)
            prompt_variant: Variant identifier for A/B testing
            adaptations_applied: List of adaptations applied
            metadata: Additional metadata

        Returns:
            Recorded PromptMetrics
        """
        metrics = PromptMetrics(
            prompt_id=f"{stage_name}_{prompt_variant}_{model}",
            stage_name=stage_name,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            quality_score=quality_score,
            relevance_score=relevance_score,
            coherence_score=coherence_score,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            prompt_variant=prompt_variant,
            adaptations_applied=adaptations_applied or [],
            metadata=metadata or {},
        )

        self._metrics.append(metrics)
        self._by_stage[stage_name].append(metrics)
        self._by_model[model].append(metrics)
        self._by_variant[prompt_variant].append(metrics)

        logger.debug(
            f"Recorded metrics: stage={stage_name}, model={model}, "
            f"quality={quality_score:.2f}, efficiency={metrics.efficiency_ratio:.2f}"
        )

        return metrics

    def get_efficiency_ratio(
        self,
        stage_name: str | None = None,
        model: str | None = None,
        variant: str | None = None,
    ) -> float:
        """
        Get average efficiency ratio for filters.

        Args:
            stage_name: Filter by stage
            model: Filter by model
            variant: Filter by variant

        Returns:
            Average efficiency ratio
        """
        metrics = self._filter_metrics(stage_name, model, variant)

        if not metrics:
            return 0.0

        ratios = [m.efficiency_ratio for m in metrics]
        return statistics.mean(ratios)

    def get_aggregated_metrics(
        self,
        stage_name: str | None = None,
        model: str | None = None,
        variant: str | None = None,
    ) -> AggregatedMetrics:
        """
        Get aggregated metrics for filters.

        Args:
            stage_name: Filter by stage
            model: Filter by model
            variant: Filter by variant

        Returns:
            AggregatedMetrics
        """
        metrics = self._filter_metrics(stage_name, model, variant)

        if not metrics:
            return AggregatedMetrics(
                stage_name=stage_name or "",
                model=model or "",
                variant=variant or "",
            )

        # Extract values
        prompt_tokens = [m.prompt_tokens for m in metrics]
        completion_tokens = [m.completion_tokens for m in metrics]
        quality_scores = [m.quality_score for m in metrics]
        efficiency_ratios = [m.efficiency_ratio for m in metrics]
        cost_efficiencies = [m.cost_efficiency for m in metrics if m.cost_efficiency > 0]
        costs = [m.cost_usd for m in metrics]
        latencies = [m.latency_ms for m in metrics]

        return AggregatedMetrics(
            stage_name=stage_name or "",
            model=model or "",
            variant=variant or "",
            execution_count=len(metrics),
            avg_prompt_tokens=statistics.mean(prompt_tokens),
            avg_completion_tokens=statistics.mean(completion_tokens),
            total_tokens=sum(m.total_tokens for m in metrics),
            avg_quality_score=statistics.mean(quality_scores),
            median_quality_score=statistics.median(quality_scores),
            quality_std_dev=statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0,
            avg_efficiency_ratio=statistics.mean(efficiency_ratios),
            avg_cost_efficiency=statistics.mean(cost_efficiencies) if cost_efficiencies else 0.0,
            total_cost_usd=sum(costs),
            avg_cost_usd=statistics.mean(costs),
            avg_latency_ms=statistics.mean(latencies),
            p95_latency_ms=self._percentile(latencies, 95),
        )

    def recommend_optimizations(
        self,
        min_samples: int | None = None,
    ) -> list[OptimizationRecommendation]:
        """
        Generate optimization recommendations based on metrics.

        Args:
            min_samples: Minimum samples required (uses default if None)

        Returns:
            List of recommendations
        """
        min_samples = min_samples or self.min_samples_for_stats
        recommendations = []

        # Check if we have enough data
        if len(self._metrics) < min_samples:
            return [OptimizationRecommendation(
                recommendation_type="insufficient_data",
                priority="medium",
                description=f"Need at least {min_samples} samples for analysis. Current: {len(self._metrics)}",
                expected_improvement="N/A",
            )]

        # Analyze by stage
        stage_recommendations = self._analyze_stages(min_samples)
        recommendations.extend(stage_recommendations)

        # Analyze by model
        model_recommendations = self._analyze_models(min_samples)
        recommendations.extend(model_recommendations)

        # Analyze by variant
        variant_recommendations = self._analyze_variants(min_samples)
        recommendations.extend(variant_recommendations)

        # Analyze token efficiency
        token_recommendations = self._analyze_token_efficiency()
        recommendations.extend(token_recommendations)

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 3))

        return recommendations

    def compare_variants(
        self,
        variant_a: str,
        variant_b: str,
        stage_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Compare two prompt variants.

        Args:
            variant_a: First variant
            variant_b: Second variant
            stage_name: Optional stage filter

        Returns:
            Comparison results
        """
        metrics_a = self._filter_metrics(stage_name, None, variant_a)
        metrics_b = self._filter_metrics(stage_name, None, variant_b)

        if not metrics_a or not metrics_b:
            return {
                "error": "Insufficient data for comparison",
                "variant_a_count": len(metrics_a),
                "variant_b_count": len(metrics_b),
            }

        agg_a = self.get_aggregated_metrics(stage_name, None, variant_a)
        agg_b = self.get_aggregated_metrics(stage_name, None, variant_b)

        return {
            "variant_a": {
                "name": variant_a,
                "count": agg_a.execution_count,
                "avg_quality": agg_a.avg_quality_score,
                "avg_efficiency": agg_a.avg_efficiency_ratio,
                "avg_cost": agg_a.avg_cost_usd,
            },
            "variant_b": {
                "name": variant_b,
                "count": agg_b.execution_count,
                "avg_quality": agg_b.avg_quality_score,
                "avg_efficiency": agg_b.avg_efficiency_ratio,
                "avg_cost": agg_b.avg_cost_usd,
            },
            "quality_winner": variant_a if agg_a.avg_quality_score > agg_b.avg_quality_score else variant_b,
            "efficiency_winner": variant_a if agg_a.avg_efficiency_ratio > agg_b.avg_efficiency_ratio else variant_b,
            "cost_winner": variant_a if agg_a.avg_cost_usd < agg_b.avg_cost_usd else variant_b,
            "quality_diff_pct": ((agg_a.avg_quality_score - agg_b.avg_quality_score) / agg_b.avg_quality_score * 100)
                if agg_b.avg_quality_score > 0 else 0,
            "efficiency_diff_pct": ((agg_a.avg_efficiency_ratio - agg_b.avg_efficiency_ratio) / agg_b.avg_efficiency_ratio * 100)
                if agg_b.avg_efficiency_ratio > 0 else 0,
        }

    def get_stage_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary metrics for all stages."""
        summary = {}
        for stage_name in self._by_stage:
            agg = self.get_aggregated_metrics(stage_name=stage_name)
            summary[stage_name] = {
                "count": agg.execution_count,
                "avg_quality": round(agg.avg_quality_score, 3),
                "avg_efficiency": round(agg.avg_efficiency_ratio, 2),
                "total_cost": round(agg.total_cost_usd, 4),
                "avg_tokens": round(agg.avg_prompt_tokens + agg.avg_completion_tokens, 0),
            }
        return summary

    def prune_old_metrics(self) -> int:
        """Remove metrics older than retention period."""
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        original_count = len(self._metrics)

        self._metrics = [m for m in self._metrics if m.timestamp >= cutoff]

        # Rebuild indexes
        self._by_stage.clear()
        self._by_model.clear()
        self._by_variant.clear()

        for m in self._metrics:
            self._by_stage[m.stage_name].append(m)
            self._by_model[m.model].append(m)
            self._by_variant[m.prompt_variant].append(m)

        pruned = original_count - len(self._metrics)
        if pruned > 0:
            logger.info(f"Pruned {pruned} old metrics")

        return pruned

    def _filter_metrics(
        self,
        stage_name: str | None,
        model: str | None,
        variant: str | None,
    ) -> list[PromptMetrics]:
        """Filter metrics by criteria."""
        metrics = self._metrics

        if stage_name:
            metrics = [m for m in metrics if m.stage_name == stage_name]
        if model:
            metrics = [m for m in metrics if m.model == model]
        if variant:
            metrics = [m for m in metrics if m.prompt_variant == variant]

        return metrics

    def _percentile(self, values: list[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def _analyze_stages(self, min_samples: int) -> list[OptimizationRecommendation]:
        """Analyze stages for optimization opportunities."""
        recommendations = []

        for stage_name, metrics in self._by_stage.items():
            if len(metrics) < min_samples:
                continue

            agg = self.get_aggregated_metrics(stage_name=stage_name)

            # Check for low efficiency
            if agg.avg_efficiency_ratio < 0.5 and agg.avg_quality_score > 0:
                recommendations.append(OptimizationRecommendation(
                    recommendation_type="low_efficiency_stage",
                    priority="high",
                    description=f"Stage '{stage_name}' has low efficiency ratio ({agg.avg_efficiency_ratio:.2f}). "
                                f"Consider using concise prompts or compressing context.",
                    expected_improvement="20-40% token reduction",
                    affected_stages=[stage_name],
                    metrics_basis={
                        "efficiency_ratio": agg.avg_efficiency_ratio,
                        "avg_tokens": agg.avg_prompt_tokens + agg.avg_completion_tokens,
                    },
                ))

            # Check for high quality variance
            if agg.quality_std_dev > 0.2:
                recommendations.append(OptimizationRecommendation(
                    recommendation_type="high_quality_variance",
                    priority="medium",
                    description=f"Stage '{stage_name}' has high quality variance (std={agg.quality_std_dev:.2f}). "
                                f"Add more examples or explicit instructions.",
                    expected_improvement="More consistent quality",
                    affected_stages=[stage_name],
                    metrics_basis={
                        "quality_std_dev": agg.quality_std_dev,
                        "avg_quality": agg.avg_quality_score,
                    },
                ))

        return recommendations

    def _analyze_models(self, min_samples: int) -> list[OptimizationRecommendation]:
        """Analyze model usage for optimization."""
        recommendations = []

        # Compare models by efficiency
        model_efficiencies = {}
        for model, metrics in self._by_model.items():
            if len(metrics) >= min_samples:
                agg = self.get_aggregated_metrics(model=model)
                model_efficiencies[model] = {
                    "efficiency": agg.avg_efficiency_ratio,
                    "quality": agg.avg_quality_score,
                    "cost": agg.avg_cost_usd,
                }

        if len(model_efficiencies) >= 2:
            # Find most efficient model
            best_model = max(model_efficiencies, key=lambda m: model_efficiencies[m]["efficiency"])
            worst_model = min(model_efficiencies, key=lambda m: model_efficiencies[m]["efficiency"])

            if model_efficiencies[best_model]["efficiency"] > model_efficiencies[worst_model]["efficiency"] * 1.5:
                recommendations.append(OptimizationRecommendation(
                    recommendation_type="model_efficiency_gap",
                    priority="medium",
                    description=f"'{best_model}' is significantly more efficient than '{worst_model}'. "
                                f"Consider routing more tasks to efficient models.",
                    expected_improvement="15-30% cost reduction",
                    metrics_basis={
                        "best_model": best_model,
                        "best_efficiency": model_efficiencies[best_model]["efficiency"],
                        "worst_model": worst_model,
                        "worst_efficiency": model_efficiencies[worst_model]["efficiency"],
                    },
                ))

        return recommendations

    def _analyze_variants(self, min_samples: int) -> list[OptimizationRecommendation]:
        """Analyze prompt variants for A/B testing insights."""
        recommendations = []

        if len(self._by_variant) < 2:
            return recommendations

        # Compare default vs other variants
        if "default" in self._by_variant and len(self._by_variant["default"]) >= min_samples:
            default_agg = self.get_aggregated_metrics(variant="default")

            for variant, metrics in self._by_variant.items():
                if variant == "default" or len(metrics) < min_samples:
                    continue

                variant_agg = self.get_aggregated_metrics(variant=variant)

                # Check if variant is significantly better
                quality_improvement = (variant_agg.avg_quality_score - default_agg.avg_quality_score) / default_agg.avg_quality_score if default_agg.avg_quality_score > 0 else 0
                efficiency_improvement = (variant_agg.avg_efficiency_ratio - default_agg.avg_efficiency_ratio) / default_agg.avg_efficiency_ratio if default_agg.avg_efficiency_ratio > 0 else 0

                if quality_improvement > 0.1 and efficiency_improvement > 0:
                    recommendations.append(OptimizationRecommendation(
                        recommendation_type="superior_variant",
                        priority="high",
                        description=f"Variant '{variant}' outperforms 'default' by {quality_improvement*100:.1f}% quality "
                                    f"and {efficiency_improvement*100:.1f}% efficiency. Consider making it default.",
                        expected_improvement=f"{quality_improvement*100:.0f}% quality improvement",
                        metrics_basis={
                            "variant": variant,
                            "quality_improvement": quality_improvement,
                            "efficiency_improvement": efficiency_improvement,
                        },
                    ))

        return recommendations

    def _analyze_token_efficiency(self) -> list[OptimizationRecommendation]:
        """Analyze overall token efficiency."""
        recommendations = []

        if not self._metrics:
            return recommendations

        # Calculate overall metrics
        total_tokens = sum(m.total_tokens for m in self._metrics)
        total_cached = sum(m.cached_tokens for m in self._metrics)
        cache_rate = total_cached / total_tokens if total_tokens > 0 else 0

        # Check cache utilization
        if cache_rate < 0.1 and total_tokens > 10000:
            recommendations.append(OptimizationRecommendation(
                recommendation_type="low_cache_utilization",
                priority="medium",
                description=f"Cache utilization is only {cache_rate*100:.1f}%. "
                            f"Enable prompt caching for repeated context.",
                expected_improvement="10-40% cost reduction",
                metrics_basis={
                    "cache_rate": cache_rate,
                    "total_tokens": total_tokens,
                },
            ))

        # Check for prompt bloat
        avg_prompt_tokens = statistics.mean(m.prompt_tokens for m in self._metrics)
        avg_completion_tokens = statistics.mean(m.completion_tokens for m in self._metrics)

        if avg_prompt_tokens > avg_completion_tokens * 5:
            recommendations.append(OptimizationRecommendation(
                recommendation_type="prompt_bloat",
                priority="high",
                description=f"Prompts are {avg_prompt_tokens/avg_completion_tokens:.1f}x larger than completions. "
                            f"Consider context compression or concise prompts.",
                expected_improvement="30-50% token reduction",
                metrics_basis={
                    "avg_prompt_tokens": avg_prompt_tokens,
                    "avg_completion_tokens": avg_completion_tokens,
                    "ratio": avg_prompt_tokens / avg_completion_tokens,
                },
            ))

        return recommendations


# Default instance
_default_tracker: PromptTracker | None = None


def get_prompt_tracker() -> PromptTracker:
    """Get the default prompt tracker instance."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = PromptTracker()
    return _default_tracker
