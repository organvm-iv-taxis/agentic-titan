"""
Titan Workflows - Personality Modulator

Enables stage personality customization through PersonalityVectors.
Allows fine-tuning of inquiry stage outputs across dimensions like
tone, abstraction, verbosity, creativity, and technicality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("titan.workflows.personality")


@dataclass
class PersonalityVector:
    """
    A vector representing personality traits for prompt modulation.

    Each dimension ranges from -1.0 to +1.0:
    - tone: -1=formal, +1=casual
    - abstraction: -1=concrete, +1=abstract
    - verbosity: -1=terse, +1=comprehensive
    - creativity: -1=conventional, +1=experimental
    - technicality: -1=accessible, +1=expert

    These values modify how prompts are generated for inquiry stages,
    allowing for customized cognitive styles.
    """

    tone: float = 0.0
    abstraction: float = 0.0
    verbosity: float = 0.0
    creativity: float = 0.0
    technicality: float = 0.0

    def __post_init__(self) -> None:
        """Validate and clamp dimension values."""
        self.tone = max(-1.0, min(1.0, self.tone))
        self.abstraction = max(-1.0, min(1.0, self.abstraction))
        self.verbosity = max(-1.0, min(1.0, self.verbosity))
        self.creativity = max(-1.0, min(1.0, self.creativity))
        self.technicality = max(-1.0, min(1.0, self.technicality))

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "tone": self.tone,
            "abstraction": self.abstraction,
            "verbosity": self.verbosity,
            "creativity": self.creativity,
            "technicality": self.technicality,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonalityVector:
        """Create from dictionary."""
        return cls(
            tone=float(data.get("tone", 0.0)),
            abstraction=float(data.get("abstraction", 0.0)),
            verbosity=float(data.get("verbosity", 0.0)),
            creativity=float(data.get("creativity", 0.0)),
            technicality=float(data.get("technicality", 0.0)),
        )

    def blend(self, other: PersonalityVector, weight: float = 0.5) -> PersonalityVector:
        """
        Blend this vector with another.

        Args:
            other: Another personality vector
            weight: Blend weight (0=all self, 1=all other)

        Returns:
            New blended PersonalityVector
        """
        w = max(0.0, min(1.0, weight))
        return PersonalityVector(
            tone=self.tone * (1 - w) + other.tone * w,
            abstraction=self.abstraction * (1 - w) + other.abstraction * w,
            verbosity=self.verbosity * (1 - w) + other.verbosity * w,
            creativity=self.creativity * (1 - w) + other.creativity * w,
            technicality=self.technicality * (1 - w) + other.technicality * w,
        )

    def intensity(self) -> float:
        """Calculate the overall intensity (magnitude) of the vector."""
        import math

        return math.sqrt(
            self.tone**2
            + self.abstraction**2
            + self.verbosity**2
            + self.creativity**2
            + self.technicality**2
        ) / math.sqrt(5)  # Normalize to 0-1 range

    def dominant_trait(self) -> tuple[str, float]:
        """Get the most prominent trait and its value."""
        traits = self.to_dict()
        # Use absolute value for dominance
        dominant = max(traits.items(), key=lambda x: abs(x[1]))
        return dominant


# =============================================================================
# Preset Personalities
# =============================================================================

PRESET_PERSONALITIES: dict[str, PersonalityVector] = {
    # Academic style - formal, precise, technical
    "academic": PersonalityVector(
        tone=-0.5,
        abstraction=0.3,
        verbosity=0.4,
        creativity=-0.2,
        technicality=0.7,
    ),
    # Conversational style - casual, accessible, moderate detail
    "conversational": PersonalityVector(
        tone=0.6,
        abstraction=-0.3,
        verbosity=0.1,
        creativity=0.2,
        technicality=-0.5,
    ),
    # Creative style - experimental, abstract, expressive
    "creative": PersonalityVector(
        tone=0.3,
        abstraction=0.5,
        verbosity=0.3,
        creativity=0.8,
        technicality=-0.2,
    ),
    # Technical style - formal, concrete, expert-level
    "technical": PersonalityVector(
        tone=-0.4,
        abstraction=-0.3,
        verbosity=0.2,
        creativity=-0.3,
        technicality=0.9,
    ),
    # Executive style - concise, high-level, professional
    "executive": PersonalityVector(
        tone=-0.2,
        abstraction=0.4,
        verbosity=-0.6,
        creativity=-0.1,
        technicality=0.3,
    ),
    # Storyteller style - narrative, engaging, accessible
    "storyteller": PersonalityVector(
        tone=0.4,
        abstraction=0.2,
        verbosity=0.5,
        creativity=0.6,
        technicality=-0.4,
    ),
    # Analytical style - precise, detailed, systematic
    "analytical": PersonalityVector(
        tone=-0.3,
        abstraction=0.1,
        verbosity=0.6,
        creativity=-0.4,
        technicality=0.5,
    ),
    # Neutral/balanced style
    "neutral": PersonalityVector(
        tone=0.0,
        abstraction=0.0,
        verbosity=0.0,
        creativity=0.0,
        technicality=0.0,
    ),
}


def get_preset_personality(name: str) -> PersonalityVector | None:
    """Get a preset personality by name."""
    return PRESET_PERSONALITIES.get(name.lower())


def list_preset_personalities() -> list[str]:
    """List all available preset personality names."""
    return list(PRESET_PERSONALITIES.keys())


# =============================================================================
# Personality Modulator
# =============================================================================


class PersonalityModulator:
    """
    Modulates prompts based on personality vectors.

    Applies personality-based modifications to base prompts to adjust
    the tone, style, and approach of AI responses.
    """

    # Instruction mappings for each dimension
    _TONE_INSTRUCTIONS = {
        -1.0: "Use formal, professional language. Avoid colloquialisms.",
        -0.5: "Maintain a professional but approachable tone.",
        0.0: "",
        0.5: "Use a friendly, conversational tone.",
        1.0: "Be casual and personable. Use natural, everyday language.",
    }

    _ABSTRACTION_INSTRUCTIONS = {
        -1.0: "Focus on concrete, specific details and examples.",
        -0.5: "Emphasize practical applications with some conceptual framing.",
        0.0: "",
        0.5: "Balance concrete examples with broader concepts.",
        1.0: "Explore abstract principles and theoretical frameworks.",
    }

    _VERBOSITY_INSTRUCTIONS = {
        -1.0: "Be extremely concise. Use bullet points. Minimize explanation.",
        -0.5: "Keep responses brief and focused. Omit unnecessary detail.",
        0.0: "",
        0.5: "Provide thorough explanations with supporting details.",
        1.0: "Be comprehensive. Explore nuances and provide extensive detail.",
    }

    _CREATIVITY_INSTRUCTIONS = {
        -1.0: "Stick to conventional, well-established approaches.",
        -0.5: "Prefer proven methods with minor variations.",
        0.0: "",
        0.5: "Explore creative alternatives and novel perspectives.",
        1.0: "Be highly experimental. Challenge assumptions. Try unconventional approaches.",
    }

    _TECHNICALITY_INSTRUCTIONS = {
        -1.0: "Use simple language accessible to a general audience.",
        -0.5: "Minimize jargon. Explain technical concepts simply.",
        0.0: "",
        0.5: "Use appropriate technical terminology with brief explanations.",
        1.0: "Assume expert knowledge. Use precise technical language throughout.",
    }

    def __init__(self) -> None:
        """Initialize the personality modulator."""
        self._dimension_instructions = {
            "tone": self._TONE_INSTRUCTIONS,
            "abstraction": self._ABSTRACTION_INSTRUCTIONS,
            "verbosity": self._VERBOSITY_INSTRUCTIONS,
            "creativity": self._CREATIVITY_INSTRUCTIONS,
            "technicality": self._TECHNICALITY_INSTRUCTIONS,
        }

    def modulate_prompt(
        self,
        base_prompt: str,
        vector: PersonalityVector,
        include_prefix: bool = True,
    ) -> str:
        """
        Apply personality modulation to a base prompt.

        Args:
            base_prompt: The original prompt text
            vector: Personality vector to apply
            include_prefix: Whether to add personality prefix

        Returns:
            Modulated prompt string
        """
        # Skip if vector is essentially neutral
        if vector.intensity() < 0.1:
            return base_prompt

        # Build modulation instructions
        instructions = self._build_instructions(vector)

        if not instructions:
            return base_prompt

        if include_prefix:
            # Add as a prefix instruction block
            instruction_text = "\n".join(f"- {inst}" for inst in instructions)
            return f"""Style instructions:
{instruction_text}

---

{base_prompt}"""
        else:
            # Append to end
            instruction_text = " ".join(instructions)
            return f"{base_prompt}\n\nNote: {instruction_text}"

    def _build_instructions(
        self,
        vector: PersonalityVector,
    ) -> list[str]:
        """Build list of instruction strings from personality vector."""
        instructions = []
        vector_dict = vector.to_dict()

        for dimension, value in vector_dict.items():
            if abs(value) < 0.15:  # Skip near-neutral values
                continue

            instruction_map = self._dimension_instructions.get(dimension, {})
            instruction = self._interpolate_instruction(instruction_map, value)

            if instruction:
                instructions.append(instruction)

        return instructions

    def _interpolate_instruction(
        self,
        instruction_map: dict[float, str],
        value: float,
    ) -> str:
        """Get the appropriate instruction for a dimension value."""
        # Find nearest key
        keys = sorted(instruction_map.keys())

        if value <= keys[0]:
            return instruction_map[keys[0]]
        if value >= keys[-1]:
            return instruction_map[keys[-1]]

        # Find bracketing keys
        for i, k in enumerate(keys[:-1]):
            if keys[i] <= value <= keys[i + 1]:
                # Use the closer instruction
                if abs(value - keys[i]) < abs(value - keys[i + 1]):
                    return instruction_map[keys[i]]
                else:
                    return instruction_map[keys[i + 1]]

        return ""

    def get_style_description(self, vector: PersonalityVector) -> str:
        """
        Generate a human-readable description of the personality style.

        Args:
            vector: Personality vector to describe

        Returns:
            Description string
        """
        parts = []

        if vector.tone < -0.3:
            parts.append("formal")
        elif vector.tone > 0.3:
            parts.append("casual")

        if vector.abstraction < -0.3:
            parts.append("concrete")
        elif vector.abstraction > 0.3:
            parts.append("abstract")

        if vector.verbosity < -0.3:
            parts.append("concise")
        elif vector.verbosity > 0.3:
            parts.append("detailed")

        if vector.creativity < -0.3:
            parts.append("conventional")
        elif vector.creativity > 0.3:
            parts.append("creative")

        if vector.technicality < -0.3:
            parts.append("accessible")
        elif vector.technicality > 0.3:
            parts.append("technical")

        if not parts:
            return "balanced"

        return ", ".join(parts)

    def suggest_for_audience(self, audience: str) -> PersonalityVector:
        """
        Suggest a personality vector for a target audience.

        Args:
            audience: Audience description (e.g., "executives", "developers", "students")

        Returns:
            Suggested PersonalityVector
        """
        audience_lower = audience.lower()

        # Map common audiences to presets
        audience_presets = {
            "executive": "executive",
            "executives": "executive",
            "ceo": "executive",
            "leadership": "executive",
            "developer": "technical",
            "developers": "technical",
            "engineer": "technical",
            "engineers": "technical",
            "student": "conversational",
            "students": "conversational",
            "beginner": "conversational",
            "beginners": "conversational",
            "researcher": "academic",
            "researchers": "academic",
            "academic": "academic",
            "scientist": "academic",
            "scientists": "academic",
            "general": "conversational",
            "public": "conversational",
            "analyst": "analytical",
            "analysts": "analytical",
            "creative": "creative",
            "creatives": "creative",
            "designer": "creative",
            "designers": "creative",
        }

        preset_name = audience_presets.get(audience_lower, "neutral")
        return PRESET_PERSONALITIES.get(preset_name, PersonalityVector())

    def suggest_for_task(self, task_type: str) -> PersonalityVector:
        """
        Suggest a personality vector for a task type.

        Args:
            task_type: Type of task (e.g., "analysis", "brainstorm", "documentation")

        Returns:
            Suggested PersonalityVector
        """
        task_lower = task_type.lower()

        task_presets = {
            "analysis": "analytical",
            "analyze": "analytical",
            "brainstorm": "creative",
            "brainstorming": "creative",
            "ideation": "creative",
            "documentation": "technical",
            "document": "technical",
            "summary": "executive",
            "summarize": "executive",
            "explain": "conversational",
            "explanation": "conversational",
            "teach": "conversational",
            "teaching": "conversational",
            "research": "academic",
            "investigate": "academic",
            "code": "technical",
            "coding": "technical",
            "debug": "technical",
            "story": "storyteller",
            "narrative": "storyteller",
            "write": "storyteller",
            "writing": "storyteller",
            "review": "analytical",
            "critique": "analytical",
        }

        preset_name = task_presets.get(task_lower, "neutral")
        return PRESET_PERSONALITIES.get(preset_name, PersonalityVector())


# =============================================================================
# Singleton and Factory
# =============================================================================

_modulator: PersonalityModulator | None = None


def get_personality_modulator() -> PersonalityModulator:
    """Get the personality modulator singleton."""
    global _modulator
    if _modulator is None:
        _modulator = PersonalityModulator()
    return _modulator


def modulate_prompt(
    base_prompt: str,
    personality: PersonalityVector | str,
) -> str:
    """
    Convenience function to modulate a prompt.

    Args:
        base_prompt: Original prompt
        personality: PersonalityVector or preset name

    Returns:
        Modulated prompt
    """
    modulator = get_personality_modulator()

    if isinstance(personality, str):
        vector = PRESET_PERSONALITIES.get(personality.lower())
        if not vector:
            logger.warning(f"Unknown personality preset: {personality}")
            return base_prompt
    else:
        vector = personality

    return modulator.modulate_prompt(base_prompt, vector)
