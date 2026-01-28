"""
Tests for titan.prompts.examples module.
"""

import pytest

from titan.prompts.examples import (
    Example,
    ExampleBank,
    STAGE_EXAMPLES,
    get_example_bank,
)


class TestExample:
    """Tests for Example dataclass."""

    def test_example_creation(self):
        """Test creating an example."""
        example = Example(
            input="What is AI?",
            output="AI is...",
            task_type="analysis",
            quality_score=0.9,
            tags=["technology"],
        )
        assert example.input == "What is AI?"
        assert example.task_type == "analysis"
        assert example.quality_score == 0.9


class TestStageExamples:
    """Tests for STAGE_EXAMPLES dictionary."""

    def test_all_stages_have_examples(self):
        """Test that all inquiry stages have examples."""
        expected_stages = [
            "scope_clarification",
            "logical_branching",
            "intuitive_branching",
            "lateral_exploration",
            "recursive_design",
            "pattern_recognition",
        ]

        for stage in expected_stages:
            assert stage in STAGE_EXAMPLES, f"Missing examples for {stage}"
            assert len(STAGE_EXAMPLES[stage]) > 0, f"No examples for {stage}"

    def test_examples_have_required_fields(self):
        """Test that all examples have input and output."""
        for stage, examples in STAGE_EXAMPLES.items():
            for i, example in enumerate(examples):
                assert "input" in example, f"Example {i} in {stage} missing input"
                assert "output" in example, f"Example {i} in {stage} missing output"

    def test_scope_clarification_examples_quality(self):
        """Test scope clarification examples have expected structure."""
        examples = STAGE_EXAMPLES["scope_clarification"]

        for example in examples:
            output = example["output"]
            # Should contain expected sections
            assert "Core Restatement" in output
            assert "Key Dimensions" in output

    def test_logical_branching_examples_quality(self):
        """Test logical branching examples have expected structure."""
        examples = STAGE_EXAMPLES["logical_branching"]

        for example in examples:
            output = example["output"]
            assert "Primary Inquiry Lines" in output or "Inquiry Lines" in output

    def test_pattern_recognition_examples_quality(self):
        """Test pattern recognition examples have expected structure."""
        examples = STAGE_EXAMPLES["pattern_recognition"]

        for example in examples:
            output = example["output"]
            assert "Meta-Pattern" in output or "Unified Theory" in output


class TestExampleBank:
    """Tests for ExampleBank class."""

    @pytest.fixture
    def bank(self):
        """Create an example bank instance."""
        return ExampleBank()

    def test_get_examples_existing(self, bank):
        """Test getting examples for existing task type."""
        examples = bank.get_examples("scope_clarification")
        assert len(examples) > 0

    def test_get_examples_nonexistent(self, bank):
        """Test getting examples for nonexistent task type."""
        examples = bank.get_examples("nonexistent_type")
        assert len(examples) == 0

    def test_get_examples_max_limit(self, bank):
        """Test max_examples parameter."""
        examples = bank.get_examples("scope_clarification", max_examples=1)
        assert len(examples) <= 1

    def test_add_example(self, bank):
        """Test adding a custom example."""
        bank.add_example(
            task_type="custom_type",
            input_text="Custom input",
            output_text="Custom output",
        )

        examples = bank.get_examples("custom_type")
        assert len(examples) == 1
        assert examples[0]["input"] == "Custom input"

    def test_add_example_to_existing(self, bank):
        """Test adding example to existing task type."""
        initial_count = len(bank.get_examples("scope_clarification", max_examples=100))

        bank.add_example(
            task_type="scope_clarification",
            input_text="New input",
            output_text="New output",
        )

        new_count = len(bank.get_examples("scope_clarification", max_examples=100))
        assert new_count == initial_count + 1

    def test_list_task_types(self, bank):
        """Test listing available task types."""
        types = bank.list_task_types()

        assert "scope_clarification" in types
        assert "logical_branching" in types
        assert "code_review" in types

    def test_get_example_count(self, bank):
        """Test getting example count."""
        count = bank.get_example_count("scope_clarification")
        assert count > 0

        count = bank.get_example_count("nonexistent")
        assert count == 0

    def test_format_examples_for_prompt(self, bank):
        """Test formatting examples as XML."""
        formatted = bank.format_examples_for_prompt("scope_clarification", max_examples=2)

        assert "<examples>" in formatted
        assert "</examples>" in formatted
        assert "<example_1>" in formatted
        assert "<input>" in formatted
        assert "<output>" in formatted

    def test_format_examples_empty(self, bank):
        """Test formatting with no examples."""
        formatted = bank.format_examples_for_prompt("nonexistent")
        assert formatted == ""

    def test_default_examples_loaded(self, bank):
        """Test that default examples are loaded on init."""
        # Stage examples
        assert bank.get_example_count("scope_clarification") > 0
        assert bank.get_example_count("logical_branching") > 0

        # Additional task types
        assert bank.get_example_count("code_review") > 0
        assert bank.get_example_count("analysis") > 0


class TestGetExampleBank:
    """Tests for get_example_bank singleton."""

    def test_returns_same_instance(self):
        """Test that get_example_bank returns singleton."""
        bank1 = get_example_bank()
        bank2 = get_example_bank()
        assert bank1 is bank2

    def test_is_example_bank(self):
        """Test that returned instance is ExampleBank."""
        bank = get_example_bank()
        assert isinstance(bank, ExampleBank)

    def test_modifications_persist(self):
        """Test that modifications to singleton persist."""
        bank = get_example_bank()
        original_count = bank.get_example_count("test_persistence")

        bank.add_example(
            task_type="test_persistence",
            input_text="Test",
            output_text="Test",
        )

        # Get again and verify
        bank2 = get_example_bank()
        new_count = bank2.get_example_count("test_persistence")
        assert new_count == original_count + 1
