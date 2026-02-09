"""
Titan Workflows - Quality Gates

Automated quality checks and validation logic for inquiry workflows.
Includes the Dialectic AI gate for contradiction detection.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from titan.core.config import get_config
from titan.metrics import get_metrics

if TYPE_CHECKING:
    from titan.workflows.inquiry_engine import InquirySession

logger = logging.getLogger("titan.workflows.quality_gates")


@dataclass
class QualityGateResult:
    """Result of a quality gate evaluation."""

    passed: bool
    score: float
    issues: list[str]
    metadata: dict[str, Any]


class DialecticGate:
    """
    Dialectic AI Quality Gate.

    Analyzes inquiry stages for logical contradictions, tensions, and
    inconsistencies between different cognitive perspectives.
    """

    def __init__(
        self,
        llm_caller: Callable[[str, str], Awaitable[str]] | None = None,
        default_model: str = "claude-3-5-sonnet-20241022",
    ) -> None:
        self.llm_caller = llm_caller
        self.default_model = default_model

    async def evaluate(self, session: InquirySession) -> QualityGateResult:
        """
        Evaluate the session for dialectic friction.

        Args:
            session: The inquiry session to evaluate.

        Returns:
            QualityGateResult with findings.
        """
        if len(session.results) < 2:
            return QualityGateResult(
                passed=True,
                score=1.0,
                issues=[],
                metadata={"reason": "Insufficient stages for dialectic analysis"},
            )

        # Prepare context for analysis
        stages_content = [
            f"Stage: {result.stage_name} ({result.role})\nContent: {result.content[:1000]}..."
            for result in session.results
        ]

        prompt = (
            "You are a Dialectic AI specialized in detecting contradictions and logical tensions.\n"
            "Analyze the following inquiry stages for internal contradictions, divergent "
            "perspectives, or logical friction.\n\n"
            "Context:\n"
            f"{chr(10).join(stages_content)}\n\n"
            "Task:\n"
            "1. Identify any direct contradictions between stages.\n"
            "2. Highlight productive tensions (where perspectives disagree in a useful way).\n"
            "3. Flag any logical fallacies or inconsistencies.\n\n"
            "Return a JSON object with:\n"
            '- "contradictions": list of strings\n'
            '- "tensions": list of strings\n'
            '- "friction_score": float 0.0 to 1.0 (0 = coherent, 1 = chaotic)\n'
        )

        try:
            if self.llm_caller:
                response = await self.llm_caller(prompt, self.default_model)
                data = self._parse_response(response)
            else:
                data = self._mock_analysis(session)

            contradictions = self._coerce_string_list(data.get("contradictions"))
            tensions = self._coerce_string_list(data.get("tensions"))
            friction_score = self._coerce_float(data.get("friction_score"), default=0.0)

            # Record metric
            if friction_score > 0.3:
                get_metrics().record_dialectic_friction(session.id)

            return QualityGateResult(
                passed=friction_score < get_config().dialectic_friction_threshold,
                score=1.0 - friction_score,
                issues=contradictions + tensions,
                metadata=data,
            )

        except Exception as e:
            logger.error(f"Dialectic gate failed: {e}")
            return QualityGateResult(False, 0.0, [str(e)], {})

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse LLM output JSON into normalized metadata."""
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return {"contradictions": [], "tensions": [], "friction_score": 0.0}

        if isinstance(parsed, dict):
            return parsed
        return {"contradictions": [], "tensions": [], "friction_score": 0.0}

    @staticmethod
    def _coerce_string_list(value: Any) -> list[str]:
        """Normalize unknown list-like values to list[str]."""
        if not isinstance(value, list):
            return []
        return [str(item) for item in value]

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        """Normalize unknown values to float with safe fallback."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _mock_analysis(self, session: InquirySession) -> dict[str, Any]:
        """Simple keyword-based mock analysis."""
        text = " ".join(result.content.lower() for result in session.results)
        contradictions: list[str] = []
        if "however" in text and "but" in text:
            contradictions.append("Potential tension detected via linguistic markers.")

        return {
            "contradictions": contradictions,
            "tensions": ["Perspective divergence between Logic and Mythos"],
            "friction_score": 0.4 if contradictions else 0.1,
        }
