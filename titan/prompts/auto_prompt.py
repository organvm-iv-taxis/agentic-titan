"""
Titan Prompts - Auto Prompting System

Dynamically adapts prompts based on:
- Task complexity
- Budget constraints
- Model tier capabilities
- Few-shot example injection

Based on research:
- Anthropic: "Examples are pictures worth a thousand words"
- Model-tier adaptation: Cheaper models need more explicit instructions
- Complexity-aware verbosity: Simpler prompts for simpler tasks
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("titan.prompts.auto_prompt")


class TaskComplexity(str, Enum):
    """Task complexity levels for prompt adaptation."""

    TRIVIAL = "trivial"  # Simple lookups, basic questions
    LOW = "low"  # Single-step reasoning
    MEDIUM = "medium"  # Multi-step reasoning
    HIGH = "high"  # Complex analysis, synthesis
    EXPERT = "expert"  # Cutting-edge, research-level


class ModelTier(str, Enum):
    """Model capability tiers."""

    ECONOMY = "economy"  # GPT-4o-mini, Claude Haiku, Llama-8B
    STANDARD = "standard"  # GPT-4o, Claude Sonnet, Llama-70B
    PREMIUM = "premium"  # GPT-4-turbo, Claude Opus
    FRONTIER = "frontier"  # Latest flagship models


@dataclass
class PromptConfig:
    """Configuration for prompt adaptation."""

    # Verbosity settings by complexity
    include_examples: bool = True
    include_reasoning_hints: bool = True
    include_output_format: bool = True
    include_constraints: bool = True

    # Model tier adjustments
    explicit_for_economy: bool = True  # More explicit for cheaper models
    cot_for_complex: bool = True  # Chain-of-thought for complex tasks

    # Token limits
    max_examples_tokens: int = 500
    max_prompt_tokens: int = 4000


@dataclass
class AdaptedPrompt:
    """Result of prompt adaptation."""

    original_template: str
    adapted_prompt: str
    complexity: TaskComplexity
    model_tier: ModelTier
    adaptations_applied: list[str] = field(default_factory=list)
    estimated_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class AutoPrompter:
    """
    Dynamically adapts prompts for optimal performance and cost.

    Features:
    - Complexity-aware verbosity adjustment
    - Model-tier specific instructions
    - Few-shot example injection
    - Output constraint specification
    - Budget-aware prompt selection
    """

    # Model tier mappings - note: partial matching uses these as substrings
    MODEL_TIERS = {
        # Anthropic
        "claude-3-opus": ModelTier.PREMIUM,
        "claude-opus-4": ModelTier.FRONTIER,
        "claude-3-5-sonnet": ModelTier.STANDARD,
        "claude-3-sonnet": ModelTier.STANDARD,
        "claude-sonnet-4": ModelTier.STANDARD,
        "claude-3-haiku": ModelTier.ECONOMY,
        # OpenAI - check mini before gpt-4o for proper substring matching
        "gpt-4o-mini": ModelTier.ECONOMY,  # Must come before gpt-4o
        "gpt-4-turbo": ModelTier.PREMIUM,
        "gpt-4o": ModelTier.STANDARD,
        "gpt-3.5-turbo": ModelTier.ECONOMY,
        # Meta/Groq
        "llama-3-70b": ModelTier.STANDARD,
        "llama-3-8b": ModelTier.ECONOMY,
        "mixtral-8x7b": ModelTier.STANDARD,
    }

    # Complexity heuristics
    COMPLEXITY_KEYWORDS = {
        TaskComplexity.HIGH: [
            "analyze", "synthesize", "evaluate", "design", "architect",
            "compare and contrast", "meta-", "recursive", "emergent",
        ],
        TaskComplexity.MEDIUM: [
            "explain", "describe", "implement", "create", "develop",
            "identify", "classify", "organize",
        ],
        TaskComplexity.LOW: [
            "list", "define", "name", "state", "summarize",
            "what is", "who is", "when did",
        ],
    }

    def __init__(
        self,
        config: PromptConfig | None = None,
        example_bank: dict[str, list[dict[str, str]]] | None = None,
    ) -> None:
        """
        Initialize auto-prompter.

        Args:
            config: Prompt configuration
            example_bank: Dictionary of task_type -> list of examples
        """
        self.config = config or PromptConfig()
        self._example_bank = example_bank or {}

    def adapt_prompt(
        self,
        template: str,
        task_complexity: TaskComplexity | None = None,
        budget_remaining: float | None = None,
        budget_total: float | None = None,
        model: str = "claude-3-5-sonnet",
        task_type: str | None = None,
        max_output_tokens: int | None = None,
    ) -> AdaptedPrompt:
        """
        Adapt prompt based on context.

        Args:
            template: Base prompt template
            task_complexity: Override complexity (auto-detected if None)
            budget_remaining: Remaining budget in USD
            budget_total: Total budget in USD
            model: Target model name
            task_type: Task type for example selection
            max_output_tokens: Maximum output tokens

        Returns:
            AdaptedPrompt with optimized prompt
        """
        adaptations = []

        # Detect complexity if not provided
        if task_complexity is None:
            task_complexity = self._detect_complexity(template)
            adaptations.append(f"auto_detected_complexity:{task_complexity.value}")

        # Get model tier
        model_tier = self._get_model_tier(model)

        # Start with template
        adapted = template

        # Apply XML structure for clarity
        if not self._has_xml_structure(template):
            adapted = self._add_xml_structure(adapted, task_type)
            adaptations.append("added_xml_structure")

        # Add chain-of-thought for complex tasks
        if self.config.cot_for_complex and task_complexity in (
            TaskComplexity.HIGH,
            TaskComplexity.EXPERT,
        ):
            if "step-by-step" not in adapted.lower():
                adapted = self._add_chain_of_thought(adapted)
                adaptations.append("added_cot")

        # Add explicit instructions for economy models
        if self.config.explicit_for_economy and model_tier == ModelTier.ECONOMY:
            adapted = self._add_explicit_instructions(adapted)
            adaptations.append("added_explicit_instructions")

        # Inject few-shot examples
        if self.config.include_examples and task_type:
            examples_added = self.inject_few_shot(adapted, task_type)
            if examples_added != adapted:
                adapted = examples_added
                adaptations.append("injected_examples")

        # Add output constraints
        if self.config.include_constraints and max_output_tokens:
            adapted = self.add_output_constraints(adapted, max_output_tokens)
            adaptations.append("added_output_constraints")

        # Budget-aware simplification
        if budget_remaining is not None and budget_total is not None:
            utilization = 1 - (budget_remaining / budget_total)
            if utilization > 0.7:
                adapted = self._simplify_for_budget(adapted)
                adaptations.append("budget_simplified")

        # Estimate tokens
        estimated_tokens = int(len(adapted) * 0.25)  # Rough estimate

        return AdaptedPrompt(
            original_template=template,
            adapted_prompt=adapted,
            complexity=task_complexity,
            model_tier=model_tier,
            adaptations_applied=adaptations,
            estimated_tokens=estimated_tokens,
            metadata={
                "model": model,
                "task_type": task_type,
            },
        )

    def inject_few_shot(
        self,
        prompt: str,
        task_type: str,
        max_examples: int = 2,
    ) -> str:
        """
        Inject relevant few-shot examples into prompt.

        Args:
            prompt: Base prompt
            task_type: Task type for example selection
            max_examples: Maximum number of examples to inject

        Returns:
            Prompt with examples injected
        """
        examples = self._example_bank.get(task_type, [])

        if not examples:
            return prompt

        # Select diverse examples
        selected = examples[:max_examples]

        if not selected:
            return prompt

        # Format examples
        examples_text = "\n\n<examples>\n"
        for i, example in enumerate(selected, 1):
            examples_text += f"<example_{i}>\n"
            if "input" in example:
                examples_text += f"<input>{example['input']}</input>\n"
            if "output" in example:
                examples_text += f"<output>{example['output']}</output>\n"
            examples_text += f"</example_{i}>\n"
        examples_text += "</examples>\n"

        # Insert examples after system section or at beginning
        if "<system>" in prompt:
            # Insert after system section
            parts = prompt.split("</system>", 1)
            if len(parts) == 2:
                return parts[0] + "</system>\n" + examples_text + parts[1]

        # Insert at appropriate location
        if "<task>" in prompt:
            return prompt.replace("<task>", examples_text + "\n<task>")

        # Prepend to prompt
        return examples_text + "\n" + prompt

    def add_output_constraints(
        self,
        prompt: str,
        max_tokens: int,
        format_hint: str | None = None,
    ) -> str:
        """
        Add explicit output constraints to prompt.

        Args:
            prompt: Base prompt
            max_tokens: Maximum output tokens
            format_hint: Optional format specification

        Returns:
            Prompt with constraints
        """
        constraints = []

        # Token limit as word approximation
        max_words = int(max_tokens * 0.75)
        constraints.append(f"Maximum response length: ~{max_words} words")

        if format_hint:
            constraints.append(f"Output format: {format_hint}")

        constraints_text = "\n<constraints>\n" + "\n".join(f"- {c}" for c in constraints) + "\n</constraints>\n"

        # Add before closing or at end
        if "</output_format>" in prompt:
            return prompt.replace("</output_format>", constraints_text + "</output_format>")

        return prompt + "\n" + constraints_text

    def set_example_bank(self, example_bank: dict[str, list[dict[str, str]]]) -> None:
        """Set the example bank for few-shot injection."""
        self._example_bank = example_bank

    def add_examples(self, task_type: str, examples: list[dict[str, str]]) -> None:
        """Add examples for a task type."""
        if task_type not in self._example_bank:
            self._example_bank[task_type] = []
        self._example_bank[task_type].extend(examples)

    def _detect_complexity(self, prompt: str) -> TaskComplexity:
        """Detect task complexity from prompt content."""
        prompt_lower = prompt.lower()

        # Check for high complexity keywords
        for complexity, keywords in self.COMPLEXITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    return complexity

        # Default to medium
        return TaskComplexity.MEDIUM

    def _get_model_tier(self, model: str) -> ModelTier:
        """Get model tier from model name."""
        model_lower = model.lower()

        # Check for mini/economy indicators first (more specific)
        if "mini" in model_lower or "haiku" in model_lower:
            return ModelTier.ECONOMY

        # Direct match - sort by key length descending for proper substring matching
        # (longer, more specific keys should match first)
        sorted_tiers = sorted(
            self.MODEL_TIERS.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )
        for model_name, tier in sorted_tiers:
            if model_name in model_lower:
                return tier

        # Fuzzy matching for model families
        if "opus" in model_lower:
            return ModelTier.PREMIUM
        if "sonnet" in model_lower:
            return ModelTier.STANDARD

        return ModelTier.STANDARD

    def _has_xml_structure(self, prompt: str) -> bool:
        """Check if prompt already has XML structure."""
        xml_tags = ["<system>", "<task>", "<context>", "<output_format>"]
        return any(tag in prompt for tag in xml_tags)

    def _add_xml_structure(self, prompt: str, task_type: str | None = None) -> str:
        """Add XML structure to prompt for clarity."""
        # Parse existing structure
        parts = {
            "system": "",
            "task": "",
            "context": "",
            "output_format": "",
        }

        # Try to identify system/role instructions
        system_patterns = [
            r"(?:System:|You are)[^\n]+(?:\n[^\n]+)*",
            r"^[^\n]+AI[^\n]*\.[^\n]*$",
        ]

        remaining = prompt
        for pattern in system_patterns:
            match = re.search(pattern, remaining, re.MULTILINE | re.IGNORECASE)
            if match:
                parts["system"] = match.group(0).strip()
                remaining = remaining.replace(match.group(0), "").strip()
                break

        # Try to identify task
        task_patterns = [
            r"(?:Task:|Your task:)[^\n]+(?:\n[^\n]+)*",
            r"(?:Please|Now)[^\n]+(?:\n[^\n]+)*",
        ]

        for pattern in task_patterns:
            match = re.search(pattern, remaining, re.MULTILINE | re.IGNORECASE)
            if match:
                parts["task"] = match.group(0).strip()
                remaining = remaining.replace(match.group(0), "").strip()
                break

        # Check for context section
        if "Previous Context:" in remaining or "Context:" in remaining:
            context_match = re.search(
                r"(?:Previous )?Context:[\s\S]+?(?=(?:Task:|Format:|$))",
                remaining,
                re.IGNORECASE,
            )
            if context_match:
                parts["context"] = context_match.group(0).strip()
                remaining = remaining.replace(context_match.group(0), "").strip()

        # Check for format section
        format_match = re.search(
            r"Format[^\n]*:[\s\S]+",
            remaining,
            re.IGNORECASE,
        )
        if format_match:
            parts["output_format"] = format_match.group(0).strip()
            remaining = remaining.replace(format_match.group(0), "").strip()

        # Anything remaining goes to task
        if remaining and not parts["task"]:
            parts["task"] = remaining

        # Build structured prompt
        structured = ""

        if parts["system"]:
            structured += f"<system>\n{parts['system']}\n</system>\n\n"

        if parts["context"]:
            structured += f"<context>\n{parts['context']}\n</context>\n\n"

        if parts["task"]:
            structured += f"<task>\n{parts['task']}\n</task>\n\n"

        if parts["output_format"]:
            structured += f"<output_format>\n{parts['output_format']}\n</output_format>\n"

        return structured.strip() if structured.strip() else prompt

    def _add_chain_of_thought(self, prompt: str) -> str:
        """Add chain-of-thought instruction."""
        cot_instruction = """
<thinking_process>
Before providing your final response, think through this step-by-step:
1. Identify the key elements of the task
2. Consider multiple perspectives or approaches
3. Evaluate trade-offs between options
4. Synthesize your analysis into a coherent response
</thinking_process>
"""
        # Insert before task section
        if "<task>" in prompt:
            return prompt.replace("<task>", cot_instruction + "\n<task>")

        # Or at the beginning
        return cot_instruction + "\n" + prompt

    def _add_explicit_instructions(self, prompt: str) -> str:
        """Add more explicit instructions for economy models."""
        explicit_additions = """
<important_instructions>
- Follow the task requirements exactly as specified
- Provide concrete, specific responses (avoid vague generalizations)
- Structure your response clearly with headers or bullet points
- If uncertain about any aspect, state your uncertainty explicitly
</important_instructions>
"""
        if "</system>" in prompt:
            return prompt.replace("</system>", "</system>\n" + explicit_additions)

        return prompt + "\n" + explicit_additions

    def _simplify_for_budget(self, prompt: str) -> str:
        """Simplify prompt when budget is low."""
        # Remove examples if present (they use tokens)
        prompt = re.sub(r"<examples>[\s\S]*?</examples>", "", prompt)

        # Shorten thinking process instructions
        prompt = re.sub(
            r"<thinking_process>[\s\S]*?</thinking_process>",
            "<thinking_process>Think briefly before responding.</thinking_process>",
            prompt,
        )

        # Remove verbose constraints
        prompt = re.sub(r"<important_instructions>[\s\S]*?</important_instructions>", "", prompt)

        return prompt.strip()


# Default instance
_default_prompter: AutoPrompter | None = None


def get_auto_prompter() -> AutoPrompter:
    """Get the default auto-prompter instance."""
    global _default_prompter
    if _default_prompter is None:
        _default_prompter = AutoPrompter()
    return _default_prompter
