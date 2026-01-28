"""
Titan Batch - Cross-Session Synthesizer

Generates unified summaries across completed sessions in a batch.
Identifies common themes, patterns, and insights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from titan.batch.artifact_store import get_artifact_store
from titan.batch.models import BatchJob, QueuedSession, SessionArtifact

if TYPE_CHECKING:
    from hive.memory import HiveMind

logger = logging.getLogger("titan.batch.synthesizer")


# =============================================================================
# Synthesis Models
# =============================================================================

@dataclass
class SynthesisResult:
    """Result of batch synthesis."""

    batch_id: str
    summary: str
    themes: list[str]
    key_insights: list[str]
    cross_references: list[dict[str, Any]]
    artifact_uri: str | None = None
    tokens_used: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_id": self.batch_id,
            "summary": self.summary,
            "themes": self.themes,
            "key_insights": self.key_insights,
            "cross_references": self.cross_references,
            "artifact_uri": self.artifact_uri,
            "tokens_used": self.tokens_used,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


# =============================================================================
# Batch Synthesizer
# =============================================================================

class BatchSynthesizer:
    """
    Generates unified summaries across batch sessions.

    Features:
    - Theme extraction across topics
    - Cross-reference identification
    - Unified insight generation
    - Export to markdown
    """

    def __init__(
        self,
        llm_caller: Callable[[str, str], Any] | None = None,
        default_model: str = "claude-3-5-sonnet-20241022",
        max_context_tokens: int = 100000,
    ) -> None:
        """
        Initialize the synthesizer.

        Args:
            llm_caller: Function to call LLM
            default_model: Model for synthesis
            max_context_tokens: Maximum context tokens
        """
        self._llm_caller = llm_caller
        self._default_model = default_model
        self._max_context_tokens = max_context_tokens
        self._artifact_store = get_artifact_store()

        logger.info("Batch synthesizer initialized")

    async def synthesize_batch(
        self,
        batch_id: str,
        batch: BatchJob | None = None,
    ) -> dict[str, Any]:
        """
        Generate synthesis for a completed batch.

        Args:
            batch_id: Batch job ID
            batch: Optional BatchJob (to avoid re-fetching)

        Returns:
            Dictionary with synthesis results
        """
        logger.info(f"Synthesizing batch {batch_id}")

        # Get artifacts
        artifacts = await self._artifact_store.list_artifacts(batch_id)
        if not artifacts:
            return {
                "batch_id": batch_id,
                "error": "No artifacts found for batch",
            }

        # Load artifact contents
        contents = await self._load_artifacts(artifacts)

        # Extract summaries from each session
        session_summaries = self._extract_session_summaries(contents)

        # Generate synthesis
        result = await self._generate_synthesis(
            batch_id=batch_id,
            session_summaries=session_summaries,
            artifacts=artifacts,
        )

        # Store synthesis as artifact
        if result.summary:
            synthesis_md = self._format_synthesis_markdown(result, artifacts)
            artifact_uri = await self._artifact_store.save_artifact(
                batch_id=batch_id,
                session_id="synthesis",
                content=synthesis_md.encode("utf-8"),
                format="markdown",
                metadata={
                    "type": "synthesis",
                    "session_count": len(artifacts),
                    "themes": result.themes,
                },
            )
            result.artifact_uri = artifact_uri

        logger.info(
            f"Synthesis complete: {len(result.themes)} themes, "
            f"{len(result.key_insights)} insights"
        )

        return result.to_dict()

    async def _load_artifacts(
        self,
        artifacts: list[SessionArtifact],
    ) -> dict[str, str]:
        """Load content from all artifacts."""
        contents = {}

        for artifact in artifacts:
            try:
                content = await self._artifact_store.get_artifact(artifact.artifact_uri)
                contents[str(artifact.session_id)] = content.decode("utf-8")
            except Exception as e:
                logger.warning(
                    f"Failed to load artifact {artifact.session_id}: {e}"
                )

        return contents

    def _extract_session_summaries(
        self,
        contents: dict[str, str],
    ) -> list[dict[str, Any]]:
        """
        Extract key information from each session's content.

        Parses markdown to extract topic, insights, and patterns.
        """
        summaries = []

        for session_id, content in contents.items():
            summary = self._parse_session_content(session_id, content)
            summaries.append(summary)

        return summaries

    def _parse_session_content(
        self,
        session_id: str,
        content: str,
    ) -> dict[str, Any]:
        """Parse session markdown content for key information."""
        lines = content.split("\n")
        summary = {
            "session_id": session_id,
            "topic": "",
            "workflow": "",
            "key_points": [],
            "stages": [],
        }

        current_stage = None
        in_results = False

        for line in lines:
            line = line.strip()

            # Extract topic from title
            if line.startswith("# ") and ":" in line:
                parts = line[2:].split(":", 1)
                if len(parts) == 2:
                    summary["topic"] = parts[1].strip()

            # Extract workflow
            if line.startswith("**Workflow:**"):
                summary["workflow"] = line.split(":", 1)[1].strip()

            # Detect stage sections
            if line.startswith("## ") and ":" in line:
                # New stage section
                if current_stage and current_stage.get("content"):
                    summary["stages"].append(current_stage)
                stage_name = line[3:].split(":")[0].strip()
                # Remove emoji if present
                if stage_name and not stage_name[0].isalnum():
                    stage_name = stage_name[1:].strip()
                current_stage = {
                    "name": stage_name,
                    "content": "",
                }
                in_results = False

            # Detect results section
            if line == "### Results":
                in_results = True
                continue

            # Capture content under results
            if in_results and current_stage and line:
                current_stage["content"] += line + "\n"

            # Extract key points from bullet lists
            if line.startswith("- ") and len(line) > 20:
                point = line[2:].strip()
                if point and len(point) < 500:
                    summary["key_points"].append(point)

        # Add last stage
        if current_stage and current_stage.get("content"):
            summary["stages"].append(current_stage)

        return summary

    async def _generate_synthesis(
        self,
        batch_id: str,
        session_summaries: list[dict[str, Any]],
        artifacts: list[SessionArtifact],
    ) -> SynthesisResult:
        """Generate unified synthesis using LLM."""
        # Build context from summaries
        context = self._build_synthesis_context(session_summaries)

        if self._llm_caller:
            # Use LLM for synthesis
            prompt = self._build_synthesis_prompt(context)
            response = await self._llm_caller(prompt, self._default_model)
            return self._parse_synthesis_response(batch_id, response, session_summaries)
        else:
            # Simple extraction without LLM
            return self._extract_synthesis(batch_id, session_summaries)

    def _build_synthesis_context(
        self,
        summaries: list[dict[str, Any]],
    ) -> str:
        """Build context string from session summaries."""
        context_parts = []

        for summary in summaries:
            part = f"""
## Topic: {summary['topic']}

### Key Points:
"""
            for point in summary.get("key_points", [])[:10]:  # Limit points
                part += f"- {point}\n"

            if summary.get("stages"):
                part += "\n### Stage Insights:\n"
                for stage in summary["stages"][:3]:  # Limit stages
                    content = stage.get("content", "")[:500]  # Truncate
                    part += f"\n**{stage['name']}:** {content}\n"

            context_parts.append(part)

        return "\n---\n".join(context_parts)

    def _build_synthesis_prompt(self, context: str) -> str:
        """Build prompt for synthesis generation."""
        return f"""Analyze the following research session results and provide a unified synthesis.

<sessions>
{context}
</sessions>

Generate a synthesis with:
1. **Summary**: A 2-3 paragraph overview of findings across all topics
2. **Common Themes**: 3-5 themes that appear across multiple sessions
3. **Key Insights**: 5-10 actionable insights from the combined research
4. **Cross-References**: Connections between topics that weren't explicitly stated

Format your response as:
## Summary
[summary text]

## Themes
- [theme 1]
- [theme 2]
...

## Key Insights
1. [insight 1]
2. [insight 2]
...

## Cross-References
- [connection 1 between topic A and topic B]
- [connection 2]
...
"""

    def _parse_synthesis_response(
        self,
        batch_id: str,
        response: str,
        summaries: list[dict[str, Any]],
    ) -> SynthesisResult:
        """Parse LLM synthesis response."""
        sections = {
            "summary": "",
            "themes": [],
            "insights": [],
            "cross_refs": [],
        }

        current_section = None
        lines = response.split("\n")

        for line in lines:
            line = line.strip()

            if line.lower().startswith("## summary"):
                current_section = "summary"
            elif line.lower().startswith("## themes") or line.lower().startswith("## common themes"):
                current_section = "themes"
            elif line.lower().startswith("## key insights") or line.lower().startswith("## insights"):
                current_section = "insights"
            elif line.lower().startswith("## cross"):
                current_section = "cross_refs"
            elif line and current_section:
                if current_section == "summary":
                    sections["summary"] += line + "\n"
                elif line.startswith("- ") or line.startswith("* "):
                    item = line[2:].strip()
                    if current_section == "themes":
                        sections["themes"].append(item)
                    elif current_section == "cross_refs":
                        sections["cross_refs"].append({"connection": item})
                elif line[0].isdigit() and "." in line[:3]:
                    item = line.split(".", 1)[1].strip()
                    if current_section == "insights":
                        sections["insights"].append(item)

        return SynthesisResult(
            batch_id=batch_id,
            summary=sections["summary"].strip(),
            themes=sections["themes"],
            key_insights=sections["insights"],
            cross_references=sections["cross_refs"],
            metadata={
                "session_count": len(summaries),
                "topics": [s["topic"] for s in summaries],
            },
        )

    def _extract_synthesis(
        self,
        batch_id: str,
        summaries: list[dict[str, Any]],
    ) -> SynthesisResult:
        """Extract synthesis without LLM (simple aggregation)."""
        # Collect all topics
        topics = [s["topic"] for s in summaries if s.get("topic")]

        # Collect all key points
        all_points = []
        for summary in summaries:
            all_points.extend(summary.get("key_points", []))

        # Find common words/themes (simple frequency analysis)
        word_freq: dict[str, int] = {}
        for point in all_points:
            words = point.lower().split()
            for word in words:
                if len(word) > 5:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1

        # Top themes based on word frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        themes = [word for word, count in sorted_words[:5] if count > 1]

        # Build summary
        summary = f"This batch explored {len(topics)} topics: {', '.join(topics[:5])}"
        if len(topics) > 5:
            summary += f" and {len(topics) - 5} more"
        summary += f". A total of {len(all_points)} key insights were identified."

        return SynthesisResult(
            batch_id=batch_id,
            summary=summary,
            themes=themes,
            key_insights=all_points[:10],  # Top 10 points
            cross_references=[],
            metadata={
                "session_count": len(summaries),
                "topics": topics,
                "extraction_method": "simple",
            },
        )

    def _format_synthesis_markdown(
        self,
        result: SynthesisResult,
        artifacts: list[SessionArtifact],
    ) -> str:
        """Format synthesis as markdown document."""
        topics = result.metadata.get("topics", [])

        md = f"""---
title: "Batch Synthesis"
batch_id: "{result.batch_id}"
session_count: {len(artifacts)}
created_date: "{result.created_at.isoformat()}"
themes:
"""
        for theme in result.themes:
            md += f"  - {theme}\n"
        md += f"""---

# Batch Research Synthesis

**Batch ID:** `{result.batch_id}`
**Sessions:** {len(artifacts)}
**Generated:** {result.created_at.strftime("%B %d, %Y %H:%M")}

## Topics Explored

"""
        for topic in topics:
            md += f"- {topic}\n"

        md += f"""
## Summary

{result.summary}

## Common Themes

"""
        for i, theme in enumerate(result.themes, 1):
            md += f"{i}. {theme}\n"

        md += """
## Key Insights

"""
        for i, insight in enumerate(result.key_insights, 1):
            md += f"{i}. {insight}\n"

        if result.cross_references:
            md += """
## Cross-Topic Connections

"""
            for ref in result.cross_references:
                connection = ref.get("connection", str(ref))
                md += f"- {connection}\n"

        md += """
---

*This synthesis was generated by the Titan Batch Research Pipeline.*
"""

        return md


# =============================================================================
# Factory Functions
# =============================================================================

_default_synthesizer: BatchSynthesizer | None = None


def get_batch_synthesizer() -> BatchSynthesizer:
    """Get the default batch synthesizer."""
    global _default_synthesizer
    if _default_synthesizer is None:
        _default_synthesizer = BatchSynthesizer()
    return _default_synthesizer
