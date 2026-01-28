"""Tests for quality evaluation metrics."""

import pytest

from .quality_metrics import (
    QualityEvaluator,
    QualityScore,
    evaluate_response_quality,
)


class TestQualityScore:
    """Tests for QualityScore dataclass."""

    def test_quality_score_creation(self):
        """Test creating a QualityScore."""
        score = QualityScore(overall=0.8, relevance=0.9, coherence=0.7)
        assert score.overall == 0.8
        assert score.relevance == 0.9
        assert score.coherence == 0.7

    def test_quality_score_defaults(self):
        """Test default values."""
        score = QualityScore(overall=0.5)
        assert score.relevance == 0.0
        assert score.coherence == 0.0
        assert score.completeness == 0.0
        assert score.safety == 0.0
        assert score.details == {}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = QualityScore(
            overall=0.8,
            relevance=0.9,
            coherence=0.7,
            completeness=0.6,
            safety=1.0,
        )
        d = score.to_dict()
        assert d["overall"] == 0.8
        assert d["relevance"] == 0.9
        assert d["coherence"] == 0.7
        assert d["completeness"] == 0.6
        assert d["safety"] == 1.0


class TestQualityEvaluator:
    """Tests for QualityEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return QualityEvaluator()

    def test_evaluate_coding_response(self, evaluator):
        """Test evaluation of coding response."""
        prompt = "Write a Python function to check if a number is prime"
        response = '''def is_prime(n):
            """Check if n is prime."""
            if n <= 1:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True'''

        score = evaluator.evaluate_response(response, prompt)
        assert score.overall > 0.5
        assert score.completeness > 0.5  # Has function def

    def test_evaluate_safety_with_forbidden_patterns(self, evaluator):
        """Test safety scoring with forbidden patterns."""
        response = "Here's the API key: sk-1234567890"
        score = evaluator.evaluate_response(
            response,
            "Help me configure the API",
            forbidden_patterns=[r"sk-\d+"],
        )
        assert score.safety < 1.0  # Should be penalized

    def test_evaluate_safety_clean_response(self, evaluator):
        """Test safety scoring with clean response."""
        response = "To configure the API, set the environment variable."
        score = evaluator.evaluate_response(
            response,
            "Help me configure the API",
        )
        assert score.safety == 1.0

    def test_code_quality_evaluation(self, evaluator):
        """Test code quality metrics."""
        code = '''
def calculate(x: int) -> int:
    """Calculate the result."""
    return x * 2
'''
        metrics = evaluator.evaluate_code_quality(code)
        assert metrics["has_function_def"]
        assert metrics["has_type_hints"]
        assert metrics["has_docstring"]
        assert metrics["quality_score"] > 0.5

    def test_code_quality_minimal(self, evaluator):
        """Test code quality with minimal code."""
        code = "x = 1"
        metrics = evaluator.evaluate_code_quality(code)
        assert not metrics["has_function_def"]
        assert not metrics["has_type_hints"]
        assert not metrics["has_docstring"]
        assert metrics["quality_score"] == 0.5

    def test_coherence_with_transitions(self, evaluator):
        """Test coherence scoring with transition words."""
        response = """
        First, we need to consider the problem.
        However, there are some challenges.
        Therefore, we must be careful.
        Finally, we can conclude.
        """
        score = evaluator.evaluate_response(response, "Explain the approach")
        assert score.coherence > 0.5

    def test_relevance_calculation(self, evaluator):
        """Test relevance based on keyword overlap."""
        prompt = "Explain how Python handles memory management"
        good_response = "Python uses automatic memory management with garbage collection."
        bad_response = "The weather today is sunny and warm."

        good_score = evaluator.evaluate_response(good_response, prompt)
        bad_score = evaluator.evaluate_response(bad_response, prompt)

        assert good_score.relevance > bad_score.relevance


class TestConvenienceFunction:
    """Tests for evaluate_response_quality function."""

    def test_basic_usage(self):
        """Test evaluate_response_quality convenience function."""
        score = evaluate_response_quality(
            "The answer is 42",
            "What is the meaning of life?",
        )
        assert isinstance(score, QualityScore)
        assert 0 <= score.overall <= 1

    def test_with_patterns(self):
        """Test with expected and forbidden patterns."""
        score = evaluate_response_quality(
            "def hello(): return 'world'",
            "Write a function",
            expected_patterns=[r"def\s+\w+"],
            forbidden_patterns=[r"password"],
        )
        assert score.overall > 0
        assert score.safety == 1.0
