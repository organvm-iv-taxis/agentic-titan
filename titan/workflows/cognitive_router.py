"""
Titan Workflows - Cognitive Model Router

Routes requests to different LLM providers based on cognitive task type.
Different models excel at different types of thinking:

- Claude: Structured reasoning, ethical analysis, consistent narrative
- GPT-4: Creative synthesis, pattern matching, cross-domain connections
- Gemini: Mathematical reasoning, scientific analysis, structured data

This router selects the optimal model for each cognitive task type,
with fallbacks and budget-aware selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from titan.costs.router import CostAwareRouter

logger = logging.getLogger("titan.workflows.cognitive_router")


class CognitiveTaskType(StrEnum):
    """Types of cognitive tasks with different model requirements."""

    STRUCTURED_REASONING = "structured_reasoning"
    CREATIVE_SYNTHESIS = "creative_synthesis"
    MATHEMATICAL_ANALYSIS = "mathematical_analysis"
    CROSS_DOMAIN = "cross_domain"
    META_ANALYSIS = "meta_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    CODE_GENERATION = "code_generation"
    RESEARCH_SYNTHESIS = "research_synthesis"


# Model preferences per cognitive task type
# Based on the cognitive framework analysis from expand_AI_inquiry
COGNITIVE_MODEL_MAP: dict[CognitiveTaskType, list[str]] = {
    CognitiveTaskType.STRUCTURED_REASONING: [
        "claude-3-5-sonnet-20241022",  # Logical consistency
        "claude-3-opus-20240229",
        "gpt-4-turbo",
    ],
    CognitiveTaskType.CREATIVE_SYNTHESIS: [
        "gpt-4-turbo",  # Creative pattern matching
        "gpt-4o",
        "claude-3-5-sonnet-20241022",
    ],
    CognitiveTaskType.MATHEMATICAL_ANALYSIS: [
        "gemini-pro",  # Strong mathematical reasoning
        "claude-3-opus-20240229",
        "gpt-4-turbo",
    ],
    CognitiveTaskType.CROSS_DOMAIN: [
        "gpt-4-turbo",  # Broad knowledge synthesis
        "gpt-4o",
        "claude-3-5-sonnet-20241022",
    ],
    CognitiveTaskType.META_ANALYSIS: [
        "claude-3-5-sonnet-20241022",  # Consistency in self-reference
        "claude-3-opus-20240229",
        "gpt-4-turbo",
    ],
    CognitiveTaskType.PATTERN_RECOGNITION: [
        "claude-3-5-sonnet-20241022",  # Balanced across pattern types
        "gpt-4-turbo",
        "gemini-pro",
    ],
    CognitiveTaskType.CODE_GENERATION: [
        "gpt-4-turbo",  # Strong code adaptation
        "claude-3-5-sonnet-20241022",
        "gpt-4o",
    ],
    CognitiveTaskType.RESEARCH_SYNTHESIS: [
        "claude-3-opus-20240229",  # Coherent integration
        "claude-3-5-sonnet-20241022",
        "gpt-4-turbo",
    ],
}

# Model rankings per cognitive task (from the source analysis)
# Scores are 0-10
MODEL_RANKINGS: dict[str, dict[CognitiveTaskType, float]] = {
    "claude-3-5-sonnet-20241022": {
        CognitiveTaskType.STRUCTURED_REASONING: 9.0,
        CognitiveTaskType.CREATIVE_SYNTHESIS: 7.0,
        CognitiveTaskType.MATHEMATICAL_ANALYSIS: 7.0,
        CognitiveTaskType.CROSS_DOMAIN: 7.0,
        CognitiveTaskType.META_ANALYSIS: 9.0,
        CognitiveTaskType.PATTERN_RECOGNITION: 8.0,
        CognitiveTaskType.CODE_GENERATION: 8.0,
        CognitiveTaskType.RESEARCH_SYNTHESIS: 9.0,
    },
    "claude-3-opus-20240229": {
        CognitiveTaskType.STRUCTURED_REASONING: 9.5,
        CognitiveTaskType.CREATIVE_SYNTHESIS: 7.5,
        CognitiveTaskType.MATHEMATICAL_ANALYSIS: 7.5,
        CognitiveTaskType.CROSS_DOMAIN: 7.5,
        CognitiveTaskType.META_ANALYSIS: 9.5,
        CognitiveTaskType.PATTERN_RECOGNITION: 8.5,
        CognitiveTaskType.CODE_GENERATION: 8.0,
        CognitiveTaskType.RESEARCH_SYNTHESIS: 9.5,
    },
    "gpt-4-turbo": {
        CognitiveTaskType.STRUCTURED_REASONING: 7.0,
        CognitiveTaskType.CREATIVE_SYNTHESIS: 9.0,
        CognitiveTaskType.MATHEMATICAL_ANALYSIS: 6.0,
        CognitiveTaskType.CROSS_DOMAIN: 9.0,
        CognitiveTaskType.META_ANALYSIS: 6.0,
        CognitiveTaskType.PATTERN_RECOGNITION: 8.0,
        CognitiveTaskType.CODE_GENERATION: 9.0,
        CognitiveTaskType.RESEARCH_SYNTHESIS: 7.0,
    },
    "gpt-4o": {
        CognitiveTaskType.STRUCTURED_REASONING: 7.0,
        CognitiveTaskType.CREATIVE_SYNTHESIS: 8.5,
        CognitiveTaskType.MATHEMATICAL_ANALYSIS: 6.5,
        CognitiveTaskType.CROSS_DOMAIN: 8.5,
        CognitiveTaskType.META_ANALYSIS: 6.5,
        CognitiveTaskType.PATTERN_RECOGNITION: 8.0,
        CognitiveTaskType.CODE_GENERATION: 8.5,
        CognitiveTaskType.RESEARCH_SYNTHESIS: 7.0,
    },
    "gemini-pro": {
        CognitiveTaskType.STRUCTURED_REASONING: 6.0,
        CognitiveTaskType.CREATIVE_SYNTHESIS: 5.0,
        CognitiveTaskType.MATHEMATICAL_ANALYSIS: 9.0,
        CognitiveTaskType.CROSS_DOMAIN: 6.0,
        CognitiveTaskType.META_ANALYSIS: 7.0,
        CognitiveTaskType.PATTERN_RECOGNITION: 8.0,
        CognitiveTaskType.CODE_GENERATION: 7.0,
        CognitiveTaskType.RESEARCH_SYNTHESIS: 6.0,
    },
}


@dataclass
class CognitiveRoutingDecision:
    """Result of cognitive routing decision."""

    model_id: str
    task_type: CognitiveTaskType
    score: float
    reasoning: str
    alternatives: list[str] = field(default_factory=list)
    fallback_used: bool = False


class CognitiveRouter:
    """
    Routes cognitive tasks to optimal models.

    Selects the best model for a given cognitive task type, considering:
    - Task type requirements
    - Model availability
    - Cost constraints (if cost router provided)
    - Explicit preferences
    """

    def __init__(
        self,
        cost_router: CostAwareRouter | None = None,
        available_models: list[str] | None = None,
        default_model: str = "claude-3-5-sonnet-20241022",
    ) -> None:
        """
        Initialize cognitive router.

        Args:
            cost_router: Optional cost-aware router for budget constraints
            available_models: List of available model IDs (if not all are available)
            default_model: Fallback model when no preference matches
        """
        self._cost_router = cost_router
        self._available_models = set(available_models) if available_models else None
        self._default_model = default_model

    async def route_for_task(
        self,
        task_type: CognitiveTaskType,
        *,
        preferred_model: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> CognitiveRoutingDecision:
        """
        Select the best available model for a cognitive task type.

        Args:
            task_type: Type of cognitive task
            preferred_model: Explicit model preference (overrides routing)
            session_id: Session ID for cost tracking
            agent_id: Agent ID for cost tracking
            context: Additional context for routing decisions

        Returns:
            CognitiveRoutingDecision with selected model and reasoning
        """
        # If explicit preference provided, use it if available
        if preferred_model:
            if self._is_model_available(preferred_model):
                return CognitiveRoutingDecision(
                    model_id=preferred_model,
                    task_type=task_type,
                    score=10.0,  # Max score for explicit preference
                    reasoning=f"Using explicitly preferred model: {preferred_model}",
                    alternatives=self._get_alternatives(task_type, exclude=preferred_model),
                )
            else:
                logger.warning(
                    f"Preferred model {preferred_model} not available, falling back to routing"
                )

        # Get ranked models for this task type
        candidates = COGNITIVE_MODEL_MAP.get(task_type, [self._default_model])
        available_candidates = [m for m in candidates if self._is_model_available(m)]

        if not available_candidates:
            # Fallback to default
            return CognitiveRoutingDecision(
                model_id=self._default_model,
                task_type=task_type,
                score=5.0,
                reasoning=f"No preferred models available for {task_type.value}, using default",
                alternatives=[],
                fallback_used=True,
            )

        # Select best model
        selected = available_candidates[0]
        score = MODEL_RANKINGS.get(selected, {}).get(task_type, 7.0)
        alternatives = available_candidates[1:3]  # Top 3 alternatives

        reasoning = (
            f"Selected {selected} for {task_type.value} "
            f"(score: {score}/10). "
            f"Model excels at {self._get_model_strengths(selected, task_type)}."
        )

        return CognitiveRoutingDecision(
            model_id=selected,
            task_type=task_type,
            score=score,
            reasoning=reasoning,
            alternatives=alternatives,
        )

    def get_model_score(
        self,
        model_id: str,
        task_type: CognitiveTaskType,
    ) -> float:
        """Get the score for a model on a specific task type."""
        return MODEL_RANKINGS.get(model_id, {}).get(task_type, 5.0)

    def get_best_model_for_task(self, task_type: CognitiveTaskType) -> str:
        """Get the best model for a task type without availability checks."""
        candidates = COGNITIVE_MODEL_MAP.get(task_type, [])
        return candidates[0] if candidates else self._default_model

    def _is_model_available(self, model_id: str) -> bool:
        """Check if a model is available."""
        if self._available_models is None:
            return True  # Assume all models available if not specified
        return model_id in self._available_models

    def _get_alternatives(
        self,
        task_type: CognitiveTaskType,
        exclude: str | None = None,
    ) -> list[str]:
        """Get alternative models for a task type."""
        candidates = COGNITIVE_MODEL_MAP.get(task_type, [])
        available = [m for m in candidates if self._is_model_available(m) and m != exclude]
        return available[:3]

    def _get_model_strengths(
        self,
        model_id: str,
        task_type: CognitiveTaskType,
    ) -> str:
        """Get a description of model strengths for this task type."""
        strengths_map = {
            "claude-3-5-sonnet-20241022": {
                CognitiveTaskType.STRUCTURED_REASONING: (
                    "logical consistency and structured analysis"
                ),
                CognitiveTaskType.META_ANALYSIS: "consistency in self-referential thinking",
                CognitiveTaskType.PATTERN_RECOGNITION: "balanced pattern recognition",
                CognitiveTaskType.RESEARCH_SYNTHESIS: "coherent integration of diverse sources",
            },
            "claude-3-opus-20240229": {
                CognitiveTaskType.STRUCTURED_REASONING: "deep logical reasoning",
                CognitiveTaskType.META_ANALYSIS: "nuanced self-reflection",
                CognitiveTaskType.RESEARCH_SYNTHESIS: "comprehensive synthesis",
            },
            "gpt-4-turbo": {
                CognitiveTaskType.CREATIVE_SYNTHESIS: "creative ideation and novel connections",
                CognitiveTaskType.CROSS_DOMAIN: "broad knowledge synthesis",
                CognitiveTaskType.CODE_GENERATION: "flexible code adaptation",
            },
            "gpt-4o": {
                CognitiveTaskType.CREATIVE_SYNTHESIS: "rapid creative exploration",
                CognitiveTaskType.CROSS_DOMAIN: "pattern matching across domains",
            },
            "gemini-pro": {
                CognitiveTaskType.MATHEMATICAL_ANALYSIS: "precise mathematical reasoning",
                CognitiveTaskType.PATTERN_RECOGNITION: "structured pattern detection",
            },
        }

        model_strengths = strengths_map.get(model_id, {})
        return model_strengths.get(task_type, "general capability")


# Singleton instance
_default_cognitive_router: CognitiveRouter | None = None


def get_cognitive_router() -> CognitiveRouter:
    """Get the default cognitive router instance."""
    global _default_cognitive_router
    if _default_cognitive_router is None:
        _default_cognitive_router = CognitiveRouter()
    return _default_cognitive_router
