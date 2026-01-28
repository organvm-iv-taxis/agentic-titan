"""
Tests for titan.prompts.token_optimizer module.
"""

import json
import pytest

from titan.prompts.token_optimizer import (
    CompressionResult,
    CompressionStrategy,
    TokenEstimate,
    TokenOptimizer,
    get_token_optimizer,
)


class TestTokenEstimate:
    """Tests for TokenEstimate dataclass."""

    def test_token_estimate_creation(self):
        """Test creating a token estimate."""
        estimate = TokenEstimate(
            text_length=100,
            estimated_tokens=25,
            model="claude",
            method="approximation",
            confidence=0.85,
        )
        assert estimate.text_length == 100
        assert estimate.estimated_tokens == 25
        assert estimate.model == "claude"
        assert estimate.confidence == 0.85


class TestCompressionResult:
    """Tests for CompressionResult dataclass."""

    def test_compression_ratio_calculation(self):
        """Test compression ratio is calculated correctly."""
        result = CompressionResult(
            original_text="This is a long text" * 100,
            compressed_text="Short summary",
            original_tokens=500,
            compressed_tokens=50,
            strategy_used=CompressionStrategy.EXTRACTIVE,
        )
        assert result.compression_ratio == pytest.approx(0.9, rel=0.01)

    def test_compression_ratio_zero_original(self):
        """Test compression ratio with zero original tokens."""
        result = CompressionResult(
            original_text="",
            compressed_text="",
            original_tokens=0,
            compressed_tokens=0,
            strategy_used=CompressionStrategy.EXTRACTIVE,
        )
        assert result.compression_ratio == 0.0


class TestTokenOptimizer:
    """Tests for TokenOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create a token optimizer instance."""
        return TokenOptimizer()

    def test_estimate_tokens_empty(self, optimizer):
        """Test token estimation for empty text."""
        estimate = optimizer.estimate_tokens("")
        assert estimate.estimated_tokens == 0
        assert estimate.confidence == 1.0

    def test_estimate_tokens_short_text(self, optimizer):
        """Test token estimation for short text."""
        text = "Hello, world!"
        estimate = optimizer.estimate_tokens(text, "claude")
        assert estimate.estimated_tokens > 0
        assert estimate.text_length == len(text)
        assert estimate.model == "claude"

    def test_estimate_tokens_with_code(self, optimizer):
        """Test token estimation adjusts for code blocks."""
        text = "Here is code:\n```python\ndef foo(): pass\n```"
        estimate = optimizer.estimate_tokens(text, "claude")
        # Code blocks should increase estimate slightly
        assert estimate.estimated_tokens > 0

    def test_estimate_tokens_different_models(self, optimizer):
        """Test estimation works for different model families."""
        text = "Test text for estimation"

        claude_estimate = optimizer.estimate_tokens(text, "claude-3-sonnet")
        gpt_estimate = optimizer.estimate_tokens(text, "gpt-4o")
        llama_estimate = optimizer.estimate_tokens(text, "llama-3-70b")

        # All should produce reasonable estimates
        assert claude_estimate.estimated_tokens > 0
        assert gpt_estimate.estimated_tokens > 0
        assert llama_estimate.estimated_tokens > 0

    def test_should_use_concise_first_stage(self, optimizer):
        """Test that first stage always uses full prompts."""
        result = optimizer.should_use_concise(
            budget_remaining=1.0,
            budget_total=10.0,
            stage=1,
            total_stages=6,
        )
        assert result is False

    def test_should_use_concise_budget_low(self, optimizer):
        """Test concise prompts when budget is low."""
        result = optimizer.should_use_concise(
            budget_remaining=0.2,
            budget_total=1.0,
            stage=3,
            total_stages=6,
        )
        assert result is True

    def test_should_use_concise_budget_healthy(self, optimizer):
        """Test full prompts when budget is healthy."""
        result = optimizer.should_use_concise(
            budget_remaining=8.0,
            budget_total=10.0,
            stage=2,
            total_stages=6,
        )
        assert result is False

    def test_compress_context_under_limit(self, optimizer):
        """Test compression returns original when under limit."""
        context = "Short context"
        result = optimizer.compress_context(context, max_tokens=1000)

        assert result.compressed_text == context
        assert result.compression_ratio == 0.0

    def test_compress_context_json(self, optimizer):
        """Test compression of JSON context."""
        context = json.dumps({
            "scope_clarification": {
                "role": "Scope AI",
                "content": "This is a long analysis " * 100,
                "stage_index": 0,
            },
            "logical_branching": {
                "role": "Logic AI",
                "content": "Another long analysis " * 100,
                "stage_index": 1,
            },
        })

        result = optimizer.compress_context(context, max_tokens=200)

        assert result.compressed_tokens <= 200 or result.compression_ratio > 0
        assert len(result.compressed_text) < len(context)

    def test_compress_stage_results(self, optimizer):
        """Test compressing stage results."""
        results = [
            {
                "stage_name": "scope_clarification",
                "content": "**Core Restatement**: This is about AI.\n**Key Dimensions**: 1. Technology 2. Ethics",
            },
            {
                "stage_name": "logical_branching",
                "content": "**Primary Inquiry Lines**: 1. How does AI work?\n- Why: Neural networks",
            },
        ]

        compressed = optimizer.compress_stage_results(results, max_tokens_per_stage=200)

        assert compressed  # Non-empty
        assert "scope_clarification" in compressed or "scope" in compressed.lower()

    def test_compress_text_context(self, optimizer):
        """Test compression of plain text context."""
        # Create text with important sentences
        sentences = [
            "This is an important finding about the topic.",
            "Another key insight emerges from the analysis.",
            "The conclusion is significant for understanding.",
            "This filler sentence provides background.",
            "Additional context helps frame the discussion.",
        ] * 20

        context = " ".join(sentences)
        result = optimizer.compress_context(context, max_tokens=100)

        assert result.compression_ratio > 0
        assert result.compressed_tokens <= result.original_tokens

    def test_hierarchical_compression(self, optimizer):
        """Test hierarchical compression strategy."""
        context = json.dumps({
            f"stage_{i}": {"content": "Analysis " * 200, "stage_index": i}
            for i in range(5)
        })

        result = optimizer.compress_context(
            context,
            max_tokens=300,
            strategy=CompressionStrategy.HIERARCHICAL,
        )

        assert result.strategy_used == CompressionStrategy.HIERARCHICAL
        assert result.compressed_tokens <= result.original_tokens

    def test_cache_functionality(self, optimizer):
        """Test that compression results are cached."""
        context = "Test context for caching " * 50

        # First call
        result1 = optimizer.compress_context(context, max_tokens=100)

        # Second call should hit cache
        result2 = optimizer.compress_context(context, max_tokens=100)

        assert result1.compressed_text == result2.compressed_text

        # Clear cache
        optimizer.clear_cache()

    def test_extract_key_findings(self, optimizer):
        """Test key findings extraction."""
        content = """
        **Core Finding**: AI is transformative.

        - Point 1: Technology advances rapidly
        - Point 2: Ethics matter

        ### Important Section

        This is crucial information about the topic.
        """

        findings = optimizer._extract_key_findings(content, max_tokens=200)

        assert "findings" in findings
        assert "summary" in findings
        assert len(findings["findings"]) > 0


class TestGetTokenOptimizer:
    """Tests for get_token_optimizer singleton."""

    def test_returns_same_instance(self):
        """Test that get_token_optimizer returns singleton."""
        optimizer1 = get_token_optimizer()
        optimizer2 = get_token_optimizer()
        assert optimizer1 is optimizer2

    def test_is_token_optimizer(self):
        """Test that returned instance is TokenOptimizer."""
        optimizer = get_token_optimizer()
        assert isinstance(optimizer, TokenOptimizer)
