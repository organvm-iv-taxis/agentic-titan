"""
Titan Workflows - Inquiry Configuration

Defines the configuration schema for multi-perspective inquiry workflows.
Each workflow consists of stages, where each stage has a specific cognitive
role and preferred model characteristics.

Based on the "Expansive Inquiry" framework from expand_AI_inquiry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CognitiveStyle(str, Enum):
    """Cognitive styles for inquiry stages."""

    STRUCTURED_REASONING = "structured_reasoning"
    CREATIVE_SYNTHESIS = "creative_synthesis"
    MATHEMATICAL_ANALYSIS = "mathematical_analysis"
    CROSS_DOMAIN = "cross_domain"
    META_ANALYSIS = "meta_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"


@dataclass
class InquiryStage:
    """
    Configuration for a single stage in an inquiry workflow.

    Each stage represents a different cognitive perspective on the topic,
    handled by a specific AI persona with optimal model characteristics.

    Attributes:
        name: Human-readable stage name (e.g., "Scope Clarification")
        role: AI role name (e.g., "Scope AI")
        description: Brief description of what this stage does
        prompt_template: Template key for the stage's prompt (in STAGE_PROMPTS)
        cognitive_style: Type of cognitive task for model routing
        preferred_model: Explicit model preference (overrides routing)
        model_traits: Traits to prioritize in model selection
        emoji: Visual identifier for the stage
        color: UI color for the stage
    """

    name: str
    role: str
    description: str
    prompt_template: str
    cognitive_style: CognitiveStyle = CognitiveStyle.STRUCTURED_REASONING
    preferred_model: str | None = None
    model_traits: list[str] = field(default_factory=list)
    emoji: str = ""
    color: str = "blue"
    metadata: dict[str, Any] = field(default_factory=dict)
    dependencies: list[int] | None = None  # Stage indices this depends on (for DAG execution)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "prompt_template": self.prompt_template,
            "cognitive_style": self.cognitive_style.value,
            "preferred_model": self.preferred_model,
            "model_traits": self.model_traits,
            "emoji": self.emoji,
            "color": self.color,
            "dependencies": self.dependencies,
        }


@dataclass
class InquiryWorkflow:
    """
    Configuration for a complete inquiry workflow.

    A workflow defines a sequence of stages that build upon each other
    to create a comprehensive exploration of a topic.

    Attributes:
        name: Workflow name (e.g., "Expansive Inquiry")
        description: What this workflow accomplishes
        stages: Ordered list of inquiry stages
        context_accumulation: Whether to pass previous stage results to next
        parallel_stages: Indices of stages that can run in parallel
        max_retries: Maximum retries per stage on failure
    """

    name: str
    description: str
    stages: list[InquiryStage]
    context_accumulation: bool = True
    parallel_stages: list[list[int]] | None = None
    max_retries: int = 2
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate workflow configuration."""
        if not self.stages:
            raise ValueError("Workflow must have at least one stage")

        # Validate parallel stage indices
        if self.parallel_stages:
            num_stages = len(self.stages)
            for group in self.parallel_stages:
                for idx in group:
                    if idx < 0 or idx >= num_stages:
                        raise ValueError(
                            f"Invalid parallel stage index {idx}, "
                            f"workflow has {num_stages} stages"
                        )

    def get_stage(self, index: int) -> InquiryStage | None:
        """Get stage by index."""
        if 0 <= index < len(self.stages):
            return self.stages[index]
        return None

    def get_stage_by_name(self, name: str) -> InquiryStage | None:
        """Get stage by name."""
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "stages": [s.to_dict() for s in self.stages],
            "context_accumulation": self.context_accumulation,
            "parallel_stages": self.parallel_stages,
            "max_retries": self.max_retries,
        }


# =============================================================================
# Default Workflows
# =============================================================================

# The 6 cognitive perspectives from expand_AI_inquiry
SCOPE_CLARIFICATION_STAGE = InquiryStage(
    name="Scope Clarification",
    role="Scope AI",
    description="Refines and distills complex topics into clear, focused questions",
    prompt_template="scope_clarification",
    cognitive_style=CognitiveStyle.STRUCTURED_REASONING,
    model_traits=["precise", "focused", "clarifying"],
    emoji="\U0001F3AF",  # Direct hit
    color="blue",
)

LOGICAL_BRANCHING_STAGE = InquiryStage(
    name="Logical Branching",
    role="Logic AI",
    description="Systematic rational exploration with layered questioning",
    prompt_template="logical_branching",
    cognitive_style=CognitiveStyle.STRUCTURED_REASONING,
    model_traits=["logical", "systematic", "rigorous"],
    emoji="\U0001F9E0",  # Brain
    color="green",
)

INTUITIVE_BRANCHING_STAGE = InquiryStage(
    name="Intuitive Branching",
    role="Mythos AI",
    description="Metaphorical and mythopoetic exploration through narrative and symbol",
    prompt_template="intuitive_branching",
    cognitive_style=CognitiveStyle.CREATIVE_SYNTHESIS,
    model_traits=["creative", "metaphorical", "narrative"],
    emoji="\U0001F4A1",  # Light bulb
    color="purple",
)

LATERAL_EXPLORATION_STAGE = InquiryStage(
    name="Lateral Exploration",
    role="Bridge AI",
    description="Cross-domain connections and hybrid thinking",
    prompt_template="lateral_exploration",
    cognitive_style=CognitiveStyle.CROSS_DOMAIN,
    model_traits=["creative", "pattern-matching", "broad-knowledge"],
    emoji="\U0001F310",  # Globe with meridians
    color="orange",
)

RECURSIVE_DESIGN_STAGE = InquiryStage(
    name="Recursive Design",
    role="Meta AI",
    description="Self-improving feedback loops and meta-analysis",
    prompt_template="recursive_design",
    cognitive_style=CognitiveStyle.META_ANALYSIS,
    model_traits=["consistent", "analytical", "self-referential"],
    emoji="\U0001F504",  # Counterclockwise arrows
    color="red",
)

PATTERN_RECOGNITION_STAGE = InquiryStage(
    name="Pattern Recognition",
    role="Pattern AI",
    description="Identifies emergent meta-patterns across all inquiry modes",
    prompt_template="pattern_recognition",
    cognitive_style=CognitiveStyle.PATTERN_RECOGNITION,
    model_traits=["analytical", "synthesizing", "emergent"],
    emoji="\U0001F332",  # Evergreen tree
    color="indigo",
)

# The default "Expansive Inquiry" workflow
EXPANSIVE_INQUIRY_WORKFLOW = InquiryWorkflow(
    name="Expansive Inquiry",
    description=(
        "A 6-stage collaborative inquiry system that explores topics from "
        "multiple cognitive perspectives. Each AI specialist contributes their "
        "unique approach, building upon previous insights to reveal dimensions "
        "that no single AI could discover alone."
    ),
    stages=[
        SCOPE_CLARIFICATION_STAGE,
        LOGICAL_BRANCHING_STAGE,
        INTUITIVE_BRANCHING_STAGE,
        LATERAL_EXPLORATION_STAGE,
        RECURSIVE_DESIGN_STAGE,
        PATTERN_RECOGNITION_STAGE,
    ],
    context_accumulation=True,
    metadata={
        "methodology": "Multi-AI Collaborative Inquiry",
        "complexity": "deep",
        "domain": "cross-disciplinary",
    },
)

# Quick inquiry workflow (3 stages for faster results)
QUICK_INQUIRY_WORKFLOW = InquiryWorkflow(
    name="Quick Inquiry",
    description=(
        "A streamlined 3-stage inquiry for faster exploration. "
        "Covers scope clarification, logical analysis, and pattern synthesis."
    ),
    stages=[
        SCOPE_CLARIFICATION_STAGE,
        LOGICAL_BRANCHING_STAGE,
        PATTERN_RECOGNITION_STAGE,
    ],
    context_accumulation=True,
    metadata={
        "methodology": "Streamlined Inquiry",
        "complexity": "moderate",
    },
)

# Creative exploration workflow (emphasizes intuitive and lateral thinking)
CREATIVE_INQUIRY_WORKFLOW = InquiryWorkflow(
    name="Creative Inquiry",
    description=(
        "A 4-stage workflow emphasizing creative and lateral exploration. "
        "Best for artistic, philosophical, or open-ended topics."
    ),
    stages=[
        SCOPE_CLARIFICATION_STAGE,
        INTUITIVE_BRANCHING_STAGE,
        LATERAL_EXPLORATION_STAGE,
        PATTERN_RECOGNITION_STAGE,
    ],
    context_accumulation=True,
    metadata={
        "methodology": "Creative Exploration",
        "complexity": "deep",
        "domain": "creative",
    },
)

# Registry of all default workflows
DEFAULT_WORKFLOWS: dict[str, InquiryWorkflow] = {
    "expansive": EXPANSIVE_INQUIRY_WORKFLOW,
    "quick": QUICK_INQUIRY_WORKFLOW,
    "creative": CREATIVE_INQUIRY_WORKFLOW,
}


def get_workflow(name: str) -> InquiryWorkflow | None:
    """Get a workflow by name."""
    return DEFAULT_WORKFLOWS.get(name.lower())


def list_workflows() -> list[str]:
    """List all available workflow names."""
    return list(DEFAULT_WORKFLOWS.keys())
