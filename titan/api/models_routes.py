"""
Titan API - Model Routes

Provides endpoints for model information and cognitive signatures.
Enables visualization of model strengths across cognitive task types.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from titan.workflows.cognitive_router import (
    CognitiveTaskType,
    MODEL_RANKINGS,
    COGNITIVE_MODEL_MAP,
    get_cognitive_router,
)

logger = logging.getLogger("titan.api.models")

models_router = APIRouter(prefix="/models", tags=["models"])


@models_router.get("/signatures")
async def get_model_signatures() -> dict[str, Any]:
    """
    Get cognitive task signatures for all models.

    Returns model rankings across all cognitive dimensions,
    suitable for radar chart visualization.

    Returns:
        Dictionary with dimensions and model rankings
    """
    return {
        "dimensions": [t.value for t in CognitiveTaskType],
        "models": {
            model_id: {
                task_type.value: score
                for task_type, score in scores.items()
            }
            for model_id, scores in MODEL_RANKINGS.items()
        },
    }


@models_router.get("/signatures/{model_id}")
async def get_model_signature(model_id: str) -> dict[str, Any]:
    """
    Get cognitive signature for a specific model.

    Args:
        model_id: The model identifier

    Returns:
        Model's cognitive task scores
    """
    if model_id not in MODEL_RANKINGS:
        return {
            "model_id": model_id,
            "found": False,
            "scores": {},
        }

    scores = MODEL_RANKINGS[model_id]
    return {
        "model_id": model_id,
        "found": True,
        "scores": {
            task_type.value: score
            for task_type, score in scores.items()
        },
        "dimensions": [t.value for t in CognitiveTaskType],
    }


@models_router.get("/list")
async def list_models() -> dict[str, Any]:
    """
    List all available models with their primary strengths.

    Returns:
        Dictionary with model list and metadata
    """
    models = []
    for model_id, scores in MODEL_RANKINGS.items():
        # Find top 3 strengths
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_strengths = [
            {"task": task_type.value, "score": score}
            for task_type, score in sorted_scores[:3]
        ]

        models.append({
            "model_id": model_id,
            "top_strengths": top_strengths,
            "average_score": sum(scores.values()) / len(scores),
        })

    # Sort by average score
    models.sort(key=lambda x: x["average_score"], reverse=True)

    return {
        "models": models,
        "total": len(models),
    }


@models_router.get("/preferences")
async def get_task_preferences() -> dict[str, Any]:
    """
    Get model preferences per cognitive task type.

    Returns:
        Dictionary mapping task types to preferred models
    """
    preferences = {}
    for task_type, models in COGNITIVE_MODEL_MAP.items():
        preferences[task_type.value] = {
            "preferred": models[0] if models else None,
            "alternatives": models[1:] if len(models) > 1 else [],
        }

    return preferences


@models_router.get("/compare")
async def compare_models(
    model_a: str,
    model_b: str,
) -> dict[str, Any]:
    """
    Compare two models across all cognitive dimensions.

    Args:
        model_a: First model identifier
        model_b: Second model identifier

    Returns:
        Comparison with differences per dimension
    """
    scores_a = MODEL_RANKINGS.get(model_a, {})
    scores_b = MODEL_RANKINGS.get(model_b, {})

    comparison = []
    for task_type in CognitiveTaskType:
        score_a = scores_a.get(task_type, 0.0)
        score_b = scores_b.get(task_type, 0.0)
        diff = score_a - score_b

        comparison.append({
            "dimension": task_type.value,
            "model_a_score": score_a,
            "model_b_score": score_b,
            "difference": diff,
            "winner": model_a if diff > 0 else (model_b if diff < 0 else "tie"),
        })

    # Determine overall winner by count
    a_wins = sum(1 for c in comparison if c["winner"] == model_a)
    b_wins = sum(1 for c in comparison if c["winner"] == model_b)

    return {
        "model_a": model_a,
        "model_b": model_b,
        "dimensions": comparison,
        "summary": {
            "model_a_wins": a_wins,
            "model_b_wins": b_wins,
            "ties": len(comparison) - a_wins - b_wins,
            "overall_winner": model_a if a_wins > b_wins else (model_b if b_wins > a_wins else "tie"),
        },
    }


@models_router.post("/route")
async def route_for_task(
    task_type: str,
    preferred_model: str | None = None,
) -> dict[str, Any]:
    """
    Get model routing recommendation for a cognitive task type.

    Args:
        task_type: The cognitive task type value
        preferred_model: Optional preferred model override

    Returns:
        Routing decision with model and reasoning
    """
    try:
        cognitive_type = CognitiveTaskType(task_type)
    except ValueError:
        return {
            "error": f"Invalid task type: {task_type}",
            "valid_types": [t.value for t in CognitiveTaskType],
        }

    router = get_cognitive_router()
    decision = await router.route_for_task(
        cognitive_type,
        preferred_model=preferred_model,
    )

    return {
        "model_id": decision.model_id,
        "task_type": decision.task_type.value,
        "score": decision.score,
        "reasoning": decision.reasoning,
        "alternatives": decision.alternatives,
        "fallback_used": decision.fallback_used,
    }
