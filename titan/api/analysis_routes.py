"""
Titan API - Analysis Routes

Endpoints for contradiction detection and dialectic synthesis
of inquiry session results.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from titan.api.typing_helpers import BaseModel, Field, typed_get, typed_post

logger = logging.getLogger("titan.api.analysis")

router = APIRouter(prefix="/analysis", tags=["analysis"])


# =============================================================================
# Request/Response Models
# =============================================================================


class ContradictionDetectRequest(BaseModel):
    """Request to detect contradictions in content."""

    content_pairs: list[dict[str, str]] = Field(
        ...,
        description="List of content pairs to analyze. Each pair has 'text_a' and 'text_b'",
    )
    sensitivity: str = Field(
        default="medium",
        description="Detection sensitivity: low, medium, high",
    )
    use_llm: bool = Field(
        default=False,
        description="Whether to use LLM for semantic analysis",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "content_pairs": [
                    {
                        "text_a": "The market is growing rapidly",
                        "text_b": "Market growth has stagnated",
                    }
                ],
                "sensitivity": "medium",
                "use_llm": False,
            }
        }
    }


class ContradictionResult(BaseModel):
    """A detected contradiction."""

    pair_index: int = Field(..., description="Index of the content pair")
    contradiction_type: str = Field(..., description="Type of contradiction detected")
    severity: str = Field(..., description="Severity: low, medium, high, critical")
    confidence: float = Field(..., description="Detection confidence 0-1")
    description: str = Field(..., description="Description of the contradiction")
    text_a_excerpt: str = Field(..., description="Relevant excerpt from text_a")
    text_b_excerpt: str = Field(..., description="Relevant excerpt from text_b")


class ContradictionDetectResponse(BaseModel):
    """Response from contradiction detection."""

    contradictions: list[ContradictionResult] = Field(..., description="Detected contradictions")
    total_pairs_analyzed: int = Field(..., description="Number of pairs analyzed")
    processing_method: str = Field(..., description="heuristic or llm")


class DialecticSynthesizeRequest(BaseModel):
    """Request to synthesize dialectically."""

    thesis: str = Field(..., description="The thesis statement or content")
    antithesis: str = Field(..., description="The antithesis statement or content")
    strategy: str = Field(
        default="auto",
        description=(
            "Synthesis strategy: auto, higher_order, integration, contextual, "
            "complementary, temporal, perspectival"
        ),
    )
    use_llm: bool = Field(
        default=False,
        description="Whether to use LLM for synthesis",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "thesis": "Technology improves quality of life",
                "antithesis": "Technology causes social isolation",
                "strategy": "auto",
                "use_llm": False,
            }
        }
    }


class DialecticSynthesisResult(BaseModel):
    """Result of dialectic synthesis."""

    synthesis: str = Field(..., description="The synthesized resolution")
    strategy_used: str = Field(..., description="Strategy that was applied")
    confidence: float = Field(..., description="Synthesis confidence 0-1")
    reasoning: str = Field(..., description="Explanation of the synthesis approach")


class SessionContradictionsResponse(BaseModel):
    """Response for session contradiction analysis."""

    session_id: str = Field(..., description="The inquiry session ID")
    stage_pairs_analyzed: int = Field(..., description="Number of stage pairs analyzed")
    contradictions: list[dict[str, Any]] = Field(..., description="Contradictions between stages")
    dialectic_suggestions: list[dict[str, Any]] = Field(
        ..., description="Suggested dialectic resolutions"
    )


# =============================================================================
# Endpoints
# =============================================================================


@typed_post(router, "/contradictions/detect", response_model=ContradictionDetectResponse)
async def detect_contradictions(request: ContradictionDetectRequest) -> ContradictionDetectResponse:
    """
    Detect contradictions in content pairs.

    Analyzes pairs of text for logical, semantic, value, temporal,
    quantitative, epistemic, and contextual contradictions.
    """
    try:
        from titan.analysis.contradictions import ContradictionPair
        from titan.analysis.detector import ContradictionDetector, DetectorConfig

        # Create detector with config
        config = DetectorConfig(
            sensitivity=request.sensitivity,
            use_llm=request.use_llm,
        )
        detector = ContradictionDetector(config)

        # Convert to ContradictionPairs
        pairs = [
            ContradictionPair(
                id=f"pair_{i}",
                text_a=p["text_a"],
                text_b=p["text_b"],
                source_a=p.get("source_a", "input_a"),
                source_b=p.get("source_b", "input_b"),
            )
            for i, p in enumerate(request.content_pairs)
        ]

        # Detect contradictions
        report = await detector.analyze(pairs)

        # Convert to response format
        results = []
        for c in report.contradictions:
            results.append(
                ContradictionResult(
                    pair_index=int(c.pair_id.split("_")[1]) if c.pair_id.startswith("pair_") else 0,
                    contradiction_type=c.contradiction_type.value,
                    severity=c.severity.value,
                    confidence=c.confidence,
                    description=c.description,
                    text_a_excerpt=c.text_a_excerpt[:200],
                    text_b_excerpt=c.text_b_excerpt[:200],
                )
            )

        return ContradictionDetectResponse(
            contradictions=results,
            total_pairs_analyzed=len(pairs),
            processing_method="llm" if request.use_llm else "heuristic",
        )

    except ImportError:
        # Analysis module not available, return mock response
        logger.warning("Contradiction analysis module not available")
        return ContradictionDetectResponse(
            contradictions=[],
            total_pairs_analyzed=len(request.content_pairs),
            processing_method="unavailable",
        )
    except Exception as e:
        logger.error(f"Contradiction detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@typed_post(router, "/dialectic/synthesize", response_model=DialecticSynthesisResult)
async def synthesize_dialectic(request: DialecticSynthesizeRequest) -> DialecticSynthesisResult:
    """
    Synthesize a resolution from thesis and antithesis.

    Uses dialectic synthesis strategies to find higher-order truths
    or complementary perspectives.
    """
    try:
        from titan.analysis.contradictions import (
            Contradiction,
            ContradictionSeverity,
            ContradictionType,
        )
        from titan.analysis.dialectic import DialecticSynthesizer, SynthesisStrategy

        synthesizer = DialecticSynthesizer(use_llm=request.use_llm)

        # Create a contradiction object
        contradiction = Contradiction(
            pair_id="synthesis_request",
            contradiction_type=ContradictionType.SEMANTIC,
            severity=ContradictionSeverity.MEDIUM,
            confidence=1.0,
            description="User-provided thesis/antithesis pair",
            text_a_excerpt=request.thesis[:500],
            text_b_excerpt=request.antithesis[:500],
        )

        # Determine strategy
        if request.strategy == "auto":
            strategy = synthesizer._determine_strategy(contradiction)
        else:
            try:
                strategy = SynthesisStrategy(request.strategy)
            except ValueError:
                strategy = SynthesisStrategy.INTEGRATION

        # Perform synthesis
        result = await synthesizer.synthesize(
            contradiction,
            thesis=request.thesis,
            antithesis=request.antithesis,
            strategy=strategy,
        )

        return DialecticSynthesisResult(
            synthesis=result.synthesis,
            strategy_used=result.strategy.value,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )

    except ImportError:
        # Dialectic module not available, return heuristic response
        logger.warning("Dialectic synthesis module not available")
        return DialecticSynthesisResult(
            synthesis=(
                f"While '{request.thesis[:50]}...' and '{request.antithesis[:50]}...' "
                "appear contradictory, they may represent complementary perspectives "
                "on a complex issue."
            ),
            strategy_used="heuristic_fallback",
            confidence=0.3,
            reasoning="Dialectic synthesis module not available; using basic heuristic response",
        )
    except Exception as e:
        logger.error(f"Dialectic synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@typed_get(
    router,
    "/inquiry/{session_id}/contradictions",
    response_model=SessionContradictionsResponse,
)
async def get_inquiry_contradictions(session_id: str) -> SessionContradictionsResponse:
    """
    Analyze an inquiry session for contradictions between stages.

    Compares results from different inquiry stages to identify
    potential contradictions and suggest dialectic resolutions.
    """
    try:
        from titan.workflows.inquiry_engine import get_inquiry_engine

        engine = get_inquiry_engine()
        session = engine.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        if len(session.results) < 2:
            return SessionContradictionsResponse(
                session_id=session_id,
                stage_pairs_analyzed=0,
                contradictions=[],
                dialectic_suggestions=[],
            )

        # Analyze pairs of stages
        contradictions = []
        suggestions = []
        pairs_analyzed = 0

        try:
            from titan.analysis.contradictions import ContradictionPair
            from titan.analysis.detector import ContradictionDetector, DetectorConfig
            from titan.analysis.dialectic import DialecticSynthesizer

            detector = ContradictionDetector(DetectorConfig(sensitivity="medium"))
            synthesizer = DialecticSynthesizer(use_llm=False)

            # Compare each pair of stages
            for i, result_a in enumerate(session.results[:-1]):
                for j, result_b in enumerate(session.results[i + 1 :], i + 1):
                    pairs_analyzed += 1

                    pair = ContradictionPair(
                        id=f"stage_{i}_{j}",
                        text_a=result_a.content,
                        text_b=result_b.content,
                        source_a=result_a.stage_name,
                        source_b=result_b.stage_name,
                    )

                    # Detect contradictions
                    report = await detector.analyze([pair])

                    for c in report.contradictions:
                        contradictions.append(
                            {
                                "stage_a": result_a.stage_name,
                                "stage_b": result_b.stage_name,
                                "type": c.contradiction_type.value,
                                "severity": c.severity.value,
                                "description": c.description,
                            }
                        )

                        # Suggest resolution
                        synthesis = await synthesizer.synthesize(
                            c,
                            thesis=result_a.content[:1000],
                            antithesis=result_b.content[:1000],
                        )
                        suggestions.append(
                            {
                                "contradiction_index": len(contradictions) - 1,
                                "synthesis": synthesis.synthesis,
                                "strategy": synthesis.strategy.value,
                            }
                        )

        except ImportError:
            logger.warning("Analysis modules not available for session analysis")

        return SessionContradictionsResponse(
            session_id=session_id,
            stage_pairs_analyzed=pairs_analyzed,
            contradictions=contradictions,
            dialectic_suggestions=suggestions,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session contradiction analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
