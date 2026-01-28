"""
Tests for titan.prompts.auto_prompt module.
"""

import pytest

from titan.prompts.auto_prompt import (
    AdaptedPrompt,
    AutoPrompter,
    ModelTier,
    PromptConfig,
    TaskComplexity,
    get_auto_prompter,
)


class TestTaskComplexity:
    """Tests for TaskComplexity enum."""

    def test_complexity_values(self):
        """Test complexity enum values."""
        assert TaskComplexity.TRIVIAL.value == "trivial"
        assert TaskComplexity.LOW.value == "low"
        assert TaskComplexity.MEDIUM.value == "medium"
        assert TaskComplexity.HIGH.value == "high"
        assert TaskComplexity.EXPERT.value == "expert"


class TestModelTier:
    """Tests for ModelTier enum."""

    def test_tier_values(self):
        """Test model tier enum values."""
        assert ModelTier.ECONOMY.value == "economy"
        assert ModelTier.STANDARD.value == "standard"
        assert ModelTier.PREMIUM.value == "premium"
        assert ModelTier.FRONTIER.value == "frontier"


class TestPromptConfig:
    """Tests for PromptConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PromptConfig()
        assert config.include_examples is True
        assert config.include_reasoning_hints is True
        assert config.max_examples_tokens == 500
        assert config.max_prompt_tokens == 4000


class TestAdaptedPrompt:
    """Tests for AdaptedPrompt dataclass."""

    def test_adapted_prompt_creation(self):
        """Test creating an adapted prompt."""
        adapted = AdaptedPrompt(
            original_template="Test template",
            adapted_prompt="Adapted version",
            complexity=TaskComplexity.MEDIUM,
            model_tier=ModelTier.STANDARD,
            adaptations_applied=["added_xml_structure"],
        )
        assert adapted.original_template == "Test template"
        assert "added_xml_structure" in adapted.adaptations_applied


class TestAutoPrompter:
    """Tests for AutoPrompter class."""

    @pytest.fixture
    def prompter(self):
        """Create an auto-prompter instance."""
        return AutoPrompter()

    @pytest.fixture
    def prompter_with_examples(self):
        """Create auto-prompter with example bank."""
        examples = {
            "analysis": [
                {"input": "Analyze X", "output": "Analysis of X..."},
                {"input": "Analyze Y", "output": "Analysis of Y..."},
            ]
        }
        return AutoPrompter(example_bank=examples)

    def test_detect_complexity_high(self, prompter):
        """Test complexity detection for high complexity."""
        prompt = "Analyze and synthesize the emergent patterns across domains"
        complexity = prompter._detect_complexity(prompt)
        assert complexity == TaskComplexity.HIGH

    def test_detect_complexity_low(self, prompter):
        """Test complexity detection for low complexity."""
        prompt = "List the main features of Python"
        complexity = prompter._detect_complexity(prompt)
        assert complexity == TaskComplexity.LOW

    def test_detect_complexity_medium(self, prompter):
        """Test complexity detection defaults to medium."""
        prompt = "A neutral prompt without specific keywords"
        complexity = prompter._detect_complexity(prompt)
        assert complexity == TaskComplexity.MEDIUM

    def test_get_model_tier_claude_opus(self, prompter):
        """Test model tier detection for Claude Opus."""
        tier = prompter._get_model_tier("claude-3-opus-20240229")
        assert tier == ModelTier.PREMIUM

    def test_get_model_tier_claude_sonnet(self, prompter):
        """Test model tier detection for Claude Sonnet."""
        tier = prompter._get_model_tier("claude-3-5-sonnet-20241022")
        assert tier == ModelTier.STANDARD

    def test_get_model_tier_claude_haiku(self, prompter):
        """Test model tier detection for Claude Haiku."""
        tier = prompter._get_model_tier("claude-3-haiku-20240307")
        assert tier == ModelTier.ECONOMY

    def test_get_model_tier_gpt4o_mini(self, prompter):
        """Test model tier detection for GPT-4o-mini."""
        tier = prompter._get_model_tier("gpt-4o-mini")
        assert tier == ModelTier.ECONOMY

    def test_adapt_prompt_adds_xml_structure(self, prompter):
        """Test that adaptation adds XML structure."""
        template = "You are an AI. Do this task."
        result = prompter.adapt_prompt(template)

        assert "added_xml_structure" in result.adaptations_applied
        assert "<system>" in result.adapted_prompt or "<task>" in result.adapted_prompt

    def test_adapt_prompt_detects_complexity(self, prompter):
        """Test that adaptation detects complexity."""
        template = "Analyze the recursive patterns in this system"
        result = prompter.adapt_prompt(template)

        assert result.complexity == TaskComplexity.HIGH

    def test_adapt_prompt_adds_cot_for_complex(self, prompter):
        """Test that chain-of-thought is added for complex tasks."""
        template = "Synthesize and evaluate the emergent structures"
        result = prompter.adapt_prompt(
            template,
            task_complexity=TaskComplexity.EXPERT,
        )

        assert "added_cot" in result.adaptations_applied
        assert "step-by-step" in result.adapted_prompt.lower() or "thinking" in result.adapted_prompt.lower()

    def test_adapt_prompt_explicit_for_economy(self, prompter):
        """Test explicit instructions for economy models."""
        template = "Do this task"
        result = prompter.adapt_prompt(
            template,
            model="gpt-4o-mini",
        )

        assert "added_explicit_instructions" in result.adaptations_applied
        assert result.model_tier == ModelTier.ECONOMY

    def test_adapt_prompt_preserves_existing_xml(self, prompter):
        """Test that existing XML structure is preserved."""
        template = "<system>You are an AI.</system>\n<task>Do this.</task>"
        result = prompter.adapt_prompt(template)

        # Should not add XML structure if already present
        assert "added_xml_structure" not in result.adaptations_applied

    def test_inject_few_shot_with_examples(self, prompter_with_examples):
        """Test few-shot injection when examples exist."""
        prompt = "Please analyze this data"
        result = prompter_with_examples.inject_few_shot(prompt, "analysis")

        assert "<examples>" in result
        assert "<example_1>" in result
        assert "Analyze X" in result

    def test_inject_few_shot_no_examples(self, prompter):
        """Test few-shot injection with no examples."""
        prompt = "Please analyze this data"
        result = prompter.inject_few_shot(prompt, "nonexistent_type")

        assert result == prompt  # Unchanged

    def test_add_output_constraints(self, prompter):
        """Test adding output constraints."""
        prompt = "Do this task"
        result = prompter.add_output_constraints(prompt, max_tokens=500)

        assert "<constraints>" in result
        assert "Maximum response length" in result

    def test_add_output_constraints_with_format(self, prompter):
        """Test adding constraints with format hint."""
        prompt = "Do this task"
        result = prompter.add_output_constraints(
            prompt,
            max_tokens=500,
            format_hint="JSON",
        )

        assert "JSON" in result

    def test_budget_simplification(self, prompter_with_examples):
        """Test prompt simplification when budget is low."""
        template = "Analyze this topic"
        result = prompter_with_examples.adapt_prompt(
            template,
            budget_remaining=0.1,
            budget_total=1.0,
            task_type="analysis",
        )

        assert "budget_simplified" in result.adaptations_applied

    def test_set_example_bank(self, prompter):
        """Test setting example bank."""
        examples = {"test": [{"input": "X", "output": "Y"}]}
        prompter.set_example_bank(examples)

        result = prompter.inject_few_shot("prompt", "test")
        assert "<examples>" in result

    def test_add_examples(self, prompter):
        """Test adding examples to bank."""
        prompter.add_examples("new_type", [{"input": "A", "output": "B"}])

        result = prompter.inject_few_shot("prompt", "new_type")
        assert "A" in result


class TestGetAutoPrompter:
    """Tests for get_auto_prompter singleton."""

    def test_returns_same_instance(self):
        """Test that get_auto_prompter returns singleton."""
        prompter1 = get_auto_prompter()
        prompter2 = get_auto_prompter()
        assert prompter1 is prompter2

    def test_is_auto_prompter(self):
        """Test that returned instance is AutoPrompter."""
        prompter = get_auto_prompter()
        assert isinstance(prompter, AutoPrompter)
