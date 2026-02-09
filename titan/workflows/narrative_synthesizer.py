"""
Titan Workflows - Narrative Synthesizer

Transforms multi-stage inquiry results into cohesive narrative outputs.
Supports multiple narrative styles and voice preservation options.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from titan.workflows.inquiry_engine import InquirySession

logger = logging.getLogger("titan.workflows.narrative")


class NarrativeStyle(StrEnum):
    """Available narrative styles."""

    ACADEMIC = "academic"  # Formal, citation-style
    JOURNALISTIC = "journalistic"  # News-style, objective
    CONVERSATIONAL = "conversational"  # Friendly, accessible
    POETIC = "poetic"  # Metaphorical, evocative
    EXECUTIVE = "executive"  # Concise, action-oriented
    TECHNICAL = "technical"  # Precise, detailed


class TargetLength(StrEnum):
    """Target length for narrative output."""

    BRIEF = "brief"  # 1-2 paragraphs
    MEDIUM = "medium"  # 3-5 paragraphs
    COMPREHENSIVE = "comprehensive"  # Full exploration


@dataclass
class NarrativeConfig:
    """
    Configuration for narrative synthesis.

    Attributes:
        style: Narrative writing style
        target_length: Desired output length
        preserve_stage_voices: Keep distinct AI personas in narrative
        highlight_contradictions: Explicitly call out contradictions
        include_methodology: Describe the inquiry process
        use_transitions: Add smooth transitions between sections
        llm_caller: Optional LLM function for enhanced synthesis
    """

    style: NarrativeStyle = NarrativeStyle.CONVERSATIONAL
    target_length: TargetLength = TargetLength.MEDIUM
    preserve_stage_voices: bool = True
    highlight_contradictions: bool = True
    include_methodology: bool = False
    use_transitions: bool = True
    llm_caller: Callable[[str, str], Awaitable[str]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "style": self.style.value,
            "target_length": self.target_length.value,
            "preserve_stage_voices": self.preserve_stage_voices,
            "highlight_contradictions": self.highlight_contradictions,
            "include_methodology": self.include_methodology,
            "use_transitions": self.use_transitions,
        }


@dataclass
class NarrativeSection:
    """A section of the synthesized narrative."""

    title: str
    content: str
    source_stage: str | None = None
    voice: str | None = None  # Preserved voice/persona
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NarrativeSynthesis:
    """
    The result of narrative synthesis.

    Contains the full narrative and metadata about the synthesis process.
    """

    title: str
    abstract: str
    sections: list[NarrativeSection]
    full_text: str
    config: NarrativeConfig
    session_id: str
    topic: str
    created_at: datetime = field(default_factory=datetime.now)
    word_count: int = 0
    stage_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "abstract": self.abstract,
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "source_stage": s.source_stage,
                    "voice": s.voice,
                }
                for s in self.sections
            ],
            "full_text": self.full_text,
            "config": self.config.to_dict(),
            "session_id": self.session_id,
            "topic": self.topic,
            "created_at": self.created_at.isoformat(),
            "word_count": self.word_count,
            "stage_count": self.stage_count,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = [
            f"# {self.title}",
            "",
            f"*{self.abstract}*",
            "",
        ]

        for section in self.sections:
            lines.append(f"## {section.title}")
            if section.voice:
                lines.append(f"*Voice: {section.voice}*")
            lines.append("")
            lines.append(section.content)
            lines.append("")

        lines.extend(
            [
                "---",
                f"*Topic: {self.topic}*",
                f"*Session: {self.session_id}*",
                f"*Generated: {self.created_at.isoformat()}*",
            ]
        )

        return "\n".join(lines)


class NarrativeSynthesizer:
    """
    Synthesizes multi-stage inquiry results into cohesive narratives.

    Transforms fragmented stage outputs into unified prose while
    optionally preserving individual stage voices and highlighting
    tensions between perspectives.
    """

    # Style-specific templates
    STYLE_INTRODUCTIONS = {
        NarrativeStyle.ACADEMIC: (
            "This analysis examines {topic} through multiple methodological lenses."
        ),
        NarrativeStyle.JOURNALISTIC: (
            "An investigation into {topic} reveals multiple dimensions of understanding."
        ),
        NarrativeStyle.CONVERSATIONAL: "Let's explore {topic} from several different angles.",
        NarrativeStyle.POETIC: (
            "The nature of {topic} unfolds through layers of meaning and perspective."
        ),
        NarrativeStyle.EXECUTIVE: "Key findings on {topic}:",
        NarrativeStyle.TECHNICAL: (
            "Technical analysis of {topic} across {stage_count} analytical dimensions."
        ),
    }

    STYLE_TRANSITIONS = {
        NarrativeStyle.ACADEMIC: [
            "Furthermore,",
            "Building on this analysis,",
            "From a complementary perspective,",
            "Extending this framework,",
        ],
        NarrativeStyle.JOURNALISTIC: [
            "Meanwhile,",
            "Another angle emerges:",
            "Sources also indicate that",
            "A different perspective suggests",
        ],
        NarrativeStyle.CONVERSATIONAL: [
            "But here's another way to look at it:",
            "Now, consider this:",
            "Interestingly,",
            "Here's where it gets interesting:",
        ],
        NarrativeStyle.POETIC: [
            "And yet, beneath the surface,",
            "Like light through a prism,",
            "Weaving through these threads,",
            "In the spaces between,",
        ],
        NarrativeStyle.EXECUTIVE: [
            "Additionally:",
            "Key point:",
            "Furthermore:",
            "Note:",
        ],
        NarrativeStyle.TECHNICAL: [
            "Additionally,",
            "In the subsequent analysis,",
            "Cross-referencing with",
            "Extending this to",
        ],
    }

    def __init__(
        self,
        default_config: NarrativeConfig | None = None,
    ) -> None:
        """Initialize the narrative synthesizer."""
        self._default_config = default_config or NarrativeConfig()

    async def synthesize(
        self,
        session: InquirySession,
        config: NarrativeConfig | None = None,
    ) -> NarrativeSynthesis:
        """
        Synthesize inquiry results into a narrative.

        Args:
            session: The inquiry session to synthesize
            config: Synthesis configuration (uses defaults if not provided)

        Returns:
            NarrativeSynthesis with the generated narrative
        """
        cfg = config or self._default_config

        logger.info(f"Synthesizing narrative for session {session.id} (style: {cfg.style.value})")

        # Generate title
        title = self._generate_title(session.topic, cfg.style)

        # Generate abstract
        abstract = self._generate_abstract(session, cfg)

        # Generate sections
        sections = await self._generate_sections(session, cfg)

        # Assemble full text
        full_text = self._assemble_full_text(title, abstract, sections, cfg)

        # Calculate word count
        word_count = len(full_text.split())

        return NarrativeSynthesis(
            title=title,
            abstract=abstract,
            sections=sections,
            full_text=full_text,
            config=cfg,
            session_id=session.id,
            topic=session.topic,
            word_count=word_count,
            stage_count=len(session.results),
            metadata={
                "workflow": session.workflow.name,
                "status": session.status.value,
            },
        )

    def _generate_title(self, topic: str, style: NarrativeStyle) -> str:
        """Generate a title appropriate to the style."""
        # Truncate and clean topic
        clean_topic = topic[:100].strip()

        title_templates = {
            NarrativeStyle.ACADEMIC: f"A Multi-Perspective Analysis of {clean_topic}",
            NarrativeStyle.JOURNALISTIC: f"Inside {clean_topic}: A Comprehensive Investigation",
            NarrativeStyle.CONVERSATIONAL: f"Understanding {clean_topic}",
            NarrativeStyle.POETIC: f"Reflections on {clean_topic}",
            NarrativeStyle.EXECUTIVE: f"Executive Brief: {clean_topic}",
            NarrativeStyle.TECHNICAL: f"Technical Analysis: {clean_topic}",
        }

        return title_templates.get(style, f"Inquiry: {clean_topic}")

    def _generate_abstract(
        self,
        session: InquirySession,
        config: NarrativeConfig,
    ) -> str:
        """Generate an abstract/summary."""
        stage_count = len(session.results)
        stage_names = [r.stage_name for r in session.results]

        if config.style == NarrativeStyle.EXECUTIVE:
            return (
                f"This {stage_count}-stage inquiry examined '{session.topic}' through "
                f"the following lenses: {', '.join(stage_names)}."
            )
        elif config.style == NarrativeStyle.ACADEMIC:
            return (
                f"This analysis employs {stage_count} distinct methodological perspectives "
                f"to examine {session.topic}. The investigation proceeds through "
                f"{', '.join(stage_names[:-1])}, and {stage_names[-1]} "
                f"to construct a comprehensive understanding of the subject."
            )
        else:
            return (
                f"This exploration of '{session.topic}' brings together {stage_count} "
                f"different ways of thinking, each revealing unique insights."
            )

    async def _generate_sections(
        self,
        session: InquirySession,
        config: NarrativeConfig,
    ) -> list[NarrativeSection]:
        """Generate narrative sections from stage results."""
        sections = []

        # Optionally add methodology section
        if config.include_methodology:
            sections.append(
                NarrativeSection(
                    title="Methodology",
                    content=self._generate_methodology(session, config),
                )
            )

        # Generate section for each stage
        for i, result in enumerate(session.results):
            section_title = self._section_title_for_stage(result.stage_name, config.style)

            # Build content
            if config.preserve_stage_voices:
                content = self._format_with_voice(result, config)
            else:
                content = self._neutralize_voice(result.content, config)

            # Add transition if not first stage and transitions enabled
            if i > 0 and config.use_transitions:
                transition = self._get_transition(i, config.style)
                content = f"{transition} {content}"

            sections.append(
                NarrativeSection(
                    title=section_title,
                    content=content,
                    source_stage=result.stage_name,
                    voice=result.role if config.preserve_stage_voices else None,
                    metadata={
                        "model": result.model_used,
                        "stage_index": i,
                    },
                )
            )

        # Add synthesis/conclusion section
        if len(session.results) >= 2:
            synthesis = self._generate_synthesis_section(session, config)
            sections.append(synthesis)

        return sections

    def _generate_methodology(
        self,
        session: InquirySession,
        config: NarrativeConfig,
    ) -> str:
        """Generate methodology description."""
        stage_lines: list[str] = []
        for i, result in enumerate(session.results):
            stage = session.workflow.get_stage(i) if i < len(session.workflow.stages) else None
            description = stage.description if stage else ""
            stage_lines.append(f"- **{result.stage_name}** ({result.role}): {description}")
        stages_desc = "\n".join(stage_lines)

        return (
            f"This inquiry employed the {session.workflow.name} workflow, comprising "
            f"{len(session.results)} distinct cognitive stages:\n\n"
            f"{stages_desc}\n\n"
            "Each stage builds upon previous insights while bringing its unique "
            "analytical perspective."
        )

    def _section_title_for_stage(self, stage_name: str, style: NarrativeStyle) -> str:
        """Generate section title based on style."""
        if style == NarrativeStyle.ACADEMIC:
            return f"Analysis: {stage_name}"
        elif style == NarrativeStyle.EXECUTIVE:
            return stage_name
        else:
            return stage_name

    def _format_with_voice(self, result: Any, config: NarrativeConfig) -> str:
        """Format content preserving the stage's voice."""
        voice_intro = f"*[{result.role}]*\n\n" if config.preserve_stage_voices else ""
        return f"{voice_intro}{result.content}"

    def _neutralize_voice(self, content: str, config: NarrativeConfig) -> str:
        """Remove persona-specific language from content."""
        # Basic neutralization - could be enhanced with LLM
        replacements = [
            ("I believe", "It appears"),
            ("In my view", "From this perspective"),
            ("I would argue", "One could argue"),
            ("I see", "One observes"),
        ]

        result = content
        for old, new in replacements:
            result = result.replace(old, new)

        return result

    def _get_transition(self, stage_index: int, style: NarrativeStyle) -> str:
        """Get a transition phrase for the style."""
        transitions = self.STYLE_TRANSITIONS.get(style, [""])
        return transitions[stage_index % len(transitions)]

    def _generate_synthesis_section(
        self,
        session: InquirySession,
        config: NarrativeConfig,
    ) -> NarrativeSection:
        """Generate a synthesis/conclusion section."""
        # Basic heuristic synthesis
        key_points = []
        for result in session.results:
            # Extract first sentence or key point
            first_sentence = result.content.split(".")[0] + "."
            key_points.append(f"- {result.stage_name}: {first_sentence[:150]}")

        synthesis_content = f"""Bringing together the perspectives above:

{chr(10).join(key_points)}

These diverse viewpoints illuminate different facets of '{session.topic}',
suggesting that comprehensive understanding requires engaging with multiple cognitive modalities."""

        return NarrativeSection(
            title="Synthesis",
            content=synthesis_content,
            metadata={"type": "synthesis"},
        )

    def _assemble_full_text(
        self,
        title: str,
        abstract: str,
        sections: list[NarrativeSection],
        config: NarrativeConfig,
    ) -> str:
        """Assemble the full narrative text."""
        parts = [
            title,
            "",
            abstract,
            "",
        ]

        for section in sections:
            parts.extend(
                [
                    f"## {section.title}",
                    "",
                    section.content,
                    "",
                ]
            )

        return "\n".join(parts)


# =============================================================================
# Factory Functions
# =============================================================================

_synthesizer: NarrativeSynthesizer | None = None


def get_narrative_synthesizer() -> NarrativeSynthesizer:
    """Get the narrative synthesizer singleton."""
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = NarrativeSynthesizer()
    return _synthesizer


async def generate_narrative(
    session: InquirySession,
    style: str = "conversational",
    preserve_voices: bool = True,
) -> NarrativeSynthesis:
    """
    Convenience function to generate a narrative.

    Args:
        session: Inquiry session to narrate
        style: Narrative style name
        preserve_voices: Whether to preserve stage voices

    Returns:
        NarrativeSynthesis
    """
    synthesizer = get_narrative_synthesizer()

    try:
        narrative_style = NarrativeStyle(style)
    except ValueError:
        narrative_style = NarrativeStyle.CONVERSATIONAL

    config = NarrativeConfig(
        style=narrative_style,
        preserve_stage_voices=preserve_voices,
    )

    return await synthesizer.synthesize(session, config)
