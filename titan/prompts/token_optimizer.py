"""
Titan Prompts - Token Optimizer

Provides token conservation through context compression, accurate token estimation,
and budget-aware prompt variant selection.

Based on research:
- Anthropic's context engineering principles (treat context as precious, finite resource)
- 30-40% token waste identified in long sessions from context accumulation
- Hierarchical summarization for sub-agent context management
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Any, Callable

logger = logging.getLogger("titan.prompts.token_optimizer")


class CompressionStrategy(str, Enum):
    """Strategies for context compression."""

    EXTRACTIVE = "extractive"  # Extract key sentences
    ABSTRACTIVE = "abstractive"  # Semantic summarization
    HIERARCHICAL = "hierarchical"  # Multi-level summarization
    SELECTIVE = "selective"  # Keep only recent + important


@dataclass
class CompressionResult:
    """Result of context compression."""

    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    strategy_used: CompressionStrategy
    compression_ratio: float = 0.0
    key_findings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.original_tokens > 0:
            self.compression_ratio = 1 - (self.compressed_tokens / self.original_tokens)


@dataclass
class TokenEstimate:
    """Token count estimate for text."""

    text_length: int
    estimated_tokens: int
    model: str
    method: str  # "tiktoken", "approximation", "model_specific"
    confidence: float = 0.9  # Higher for exact methods


class TokenOptimizer:
    """
    Optimizes token usage through context compression and estimation.

    Features:
    - Context compaction via summarization
    - Accurate per-model token estimation
    - Budget-aware prompt variant selection
    - Hierarchical summarization for sub-agent results
    """

    # Approximate tokens per character for different models
    TOKENS_PER_CHAR = {
        "claude": 0.25,  # ~4 chars per token
        "gpt": 0.25,
        "llama": 0.28,
        "mistral": 0.27,
        "default": 0.25,
    }

    # Model context windows (for reference)
    MODEL_CONTEXT_WINDOWS = {
        "claude-3-opus": 200_000,
        "claude-3-sonnet": 200_000,
        "claude-3-haiku": 200_000,
        "claude-3-5-sonnet": 200_000,
        "gpt-4-turbo": 128_000,
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
        "llama-3-70b": 8_192,
        "mixtral-8x7b": 32_768,
    }

    def __init__(
        self,
        summarizer: Callable[[str, int], str] | None = None,
        max_context_tokens: int = 4000,
        compression_threshold: float = 0.7,  # Compress when >70% of budget used
    ) -> None:
        """
        Initialize token optimizer.

        Args:
            summarizer: Async function to summarize text (text, max_tokens) -> summary
            max_context_tokens: Default max tokens for context
            compression_threshold: Budget threshold to trigger compression
        """
        self._summarizer = summarizer
        self.max_context_tokens = max_context_tokens
        self.compression_threshold = compression_threshold
        self._cache: dict[str, CompressionResult] = {}

    def estimate_tokens(self, text: str, model: str = "claude") -> TokenEstimate:
        """
        Estimate token count for text.

        Uses model-specific approximations. For more accurate counts,
        use tiktoken when available.

        Args:
            text: Text to estimate
            model: Model name or family

        Returns:
            TokenEstimate with count and confidence
        """
        if not text:
            return TokenEstimate(
                text_length=0,
                estimated_tokens=0,
                model=model,
                method="empty",
                confidence=1.0,
            )

        # Determine model family
        model_family = "default"
        model_lower = model.lower()
        for family in ["claude", "gpt", "llama", "mistral"]:
            if family in model_lower:
                model_family = family
                break

        # Get tokens per char ratio
        ratio = self.TOKENS_PER_CHAR.get(model_family, self.TOKENS_PER_CHAR["default"])

        # Base estimate
        char_count = len(text)
        base_estimate = int(char_count * ratio)

        # Adjust for special characters and formatting
        # Code blocks tend to have more tokens
        code_blocks = len(re.findall(r"```[\s\S]*?```", text))
        json_blocks = text.count("{") + text.count("[")

        adjustment = 1.0
        if code_blocks > 0:
            adjustment += 0.1 * code_blocks
        if json_blocks > 10:
            adjustment += 0.05

        estimated = int(base_estimate * adjustment)

        return TokenEstimate(
            text_length=char_count,
            estimated_tokens=estimated,
            model=model,
            method="approximation",
            confidence=0.85,
        )

    def should_use_concise(
        self,
        budget_remaining: float,
        budget_total: float,
        stage: int = 1,
        total_stages: int = 6,
    ) -> bool:
        """
        Determine if concise prompt variants should be used.

        Uses budget-aware heuristics to decide when to switch
        from full prompts to concise variants.

        Args:
            budget_remaining: Remaining budget in USD
            budget_total: Total allocated budget
            stage: Current stage number (1-indexed)
            total_stages: Total stages in workflow

        Returns:
            True if concise prompts should be used
        """
        if budget_total <= 0:
            return False

        utilization = 1 - (budget_remaining / budget_total)
        stages_remaining = total_stages - stage

        # Always use full prompts for first stage (most important)
        if stage == 1:
            return False

        # Use concise if budget >70% used and stages remain
        if utilization > self.compression_threshold and stages_remaining > 0:
            return True

        # Use concise if budget won't cover full prompts for remaining stages
        # Assume ~$0.01 per full stage, ~$0.004 per concise stage
        full_cost = stages_remaining * 0.01
        concise_cost = stages_remaining * 0.004

        if budget_remaining < full_cost and budget_remaining >= concise_cost:
            return True

        return False

    def compress_context(
        self,
        context: str,
        max_tokens: int | None = None,
        strategy: CompressionStrategy = CompressionStrategy.EXTRACTIVE,
        preserve_recent: int = 2,
    ) -> CompressionResult:
        """
        Compress accumulated context to fit within token limits.

        Args:
            context: Context string (often JSON from previous stages)
            max_tokens: Maximum tokens for compressed result
            strategy: Compression strategy to use
            preserve_recent: Number of recent entries to preserve fully

        Returns:
            CompressionResult with compressed context
        """
        max_tokens = max_tokens or self.max_context_tokens

        # Check cache
        cache_key = self._cache_key(context, max_tokens, strategy)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Estimate current tokens
        estimate = self.estimate_tokens(context)

        # If already under limit, return as-is
        if estimate.estimated_tokens <= max_tokens:
            result = CompressionResult(
                original_text=context,
                compressed_text=context,
                original_tokens=estimate.estimated_tokens,
                compressed_tokens=estimate.estimated_tokens,
                strategy_used=strategy,
                key_findings=[],
            )
            self._cache[cache_key] = result
            return result

        # Apply compression strategy
        if strategy == CompressionStrategy.EXTRACTIVE:
            compressed = self._extractive_compress(context, max_tokens, preserve_recent)
        elif strategy == CompressionStrategy.HIERARCHICAL:
            compressed = self._hierarchical_compress(context, max_tokens, preserve_recent)
        elif strategy == CompressionStrategy.SELECTIVE:
            compressed = self._selective_compress(context, max_tokens, preserve_recent)
        else:
            # Default to extractive
            compressed = self._extractive_compress(context, max_tokens, preserve_recent)

        compressed_estimate = self.estimate_tokens(compressed["text"])

        result = CompressionResult(
            original_text=context,
            compressed_text=compressed["text"],
            original_tokens=estimate.estimated_tokens,
            compressed_tokens=compressed_estimate.estimated_tokens,
            strategy_used=strategy,
            key_findings=compressed.get("key_findings", []),
            metadata=compressed.get("metadata", {}),
        )

        self._cache[cache_key] = result
        return result

    def compress_stage_results(
        self,
        results: list[dict[str, Any]],
        max_tokens_per_stage: int = 500,
    ) -> str:
        """
        Compress stage results for context accumulation.

        Extracts key findings from each stage rather than
        full responses.

        Args:
            results: List of stage result dictionaries
            max_tokens_per_stage: Max tokens per stage summary

        Returns:
            Compressed context string
        """
        if not results:
            return ""

        compressed_stages = []

        for result in results:
            stage_name = result.get("stage_name", "unknown")
            content = result.get("content", "")

            # Extract key findings using extractive summarization
            summary = self._extract_key_findings(content, max_tokens_per_stage)

            compressed_stages.append({
                "stage": stage_name,
                "key_insights": summary["findings"],
                "summary": summary["summary"],
            })

        return json.dumps(compressed_stages, indent=1)

    def _extractive_compress(
        self,
        context: str,
        max_tokens: int,
        preserve_recent: int,
    ) -> dict[str, Any]:
        """Extract key sentences based on importance signals."""
        try:
            # Try to parse as JSON (common for stage results)
            data = json.loads(context)
            return self._compress_json_context(data, max_tokens, preserve_recent)
        except json.JSONDecodeError:
            # Plain text compression
            return self._compress_text_context(context, max_tokens)

    def _compress_json_context(
        self,
        data: dict[str, Any],
        max_tokens: int,
        preserve_recent: int,
    ) -> dict[str, Any]:
        """Compress JSON-structured context (stage results)."""
        if not isinstance(data, dict):
            return {
                "text": json.dumps(data)[:max_tokens * 4],
                "key_findings": [],
            }

        # Identify stage entries (they have content, role, stage_index)
        stages = []
        for key, value in data.items():
            if isinstance(value, dict) and "content" in value:
                stages.append((key, value))

        if not stages:
            # Not stage data, just truncate
            return {
                "text": json.dumps(data)[:max_tokens * 4],
                "key_findings": [],
            }

        # Sort by stage index if available
        stages.sort(key=lambda x: x[1].get("stage_index", 0))

        # Preserve recent stages fully
        preserved = stages[-preserve_recent:] if preserve_recent > 0 else []
        to_compress = stages[:-preserve_recent] if preserve_recent > 0 else stages

        compressed_data = {}
        key_findings = []

        # Compress older stages
        for key, value in to_compress:
            content = value.get("content", "")
            findings = self._extract_key_findings(content, 200)

            compressed_data[key] = {
                "role": value.get("role", ""),
                "summary": findings["summary"],
                "key_points": findings["findings"][:3],
            }
            key_findings.extend(findings["findings"][:2])

        # Keep recent stages more complete
        for key, value in preserved:
            content = value.get("content", "")
            # Keep more of recent content
            truncated = content[:2000] if len(content) > 2000 else content
            compressed_data[key] = {
                "role": value.get("role", ""),
                "content": truncated,
                "stage_index": value.get("stage_index", 0),
            }

        return {
            "text": json.dumps(compressed_data, indent=1),
            "key_findings": key_findings,
            "metadata": {"stages_compressed": len(to_compress)},
        }

    def _compress_text_context(
        self,
        text: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Compress plain text context."""
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)

        if not sentences:
            return {"text": "", "key_findings": []}

        # Score sentences by importance signals
        scored = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, i, len(sentences))
            scored.append((score, sentence))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top sentences until token limit
        selected = []
        current_tokens = 0
        key_findings = []

        for score, sentence in scored:
            tokens = self.estimate_tokens(sentence).estimated_tokens
            if current_tokens + tokens <= max_tokens:
                selected.append(sentence)
                current_tokens += tokens
                if score > 0.7 and len(key_findings) < 5:
                    key_findings.append(sentence[:100])

        # Re-order by original position for coherence
        original_order = {s: i for i, s in enumerate(sentences)}
        selected.sort(key=lambda s: original_order.get(s, 0))

        return {
            "text": " ".join(selected),
            "key_findings": key_findings,
        }

    def _hierarchical_compress(
        self,
        context: str,
        max_tokens: int,
        preserve_recent: int,
    ) -> dict[str, Any]:
        """Multi-level hierarchical summarization."""
        # First level: extractive compression
        extracted = self._extractive_compress(context, max_tokens * 2, preserve_recent)

        # If still too large, apply second level
        estimate = self.estimate_tokens(extracted["text"])
        if estimate.estimated_tokens > max_tokens:
            # Further compress the extracted text
            second_pass = self._compress_text_context(extracted["text"], max_tokens)
            return {
                "text": second_pass["text"],
                "key_findings": extracted["key_findings"] + second_pass["key_findings"],
                "metadata": {"compression_levels": 2},
            }

        return extracted

    def _selective_compress(
        self,
        context: str,
        max_tokens: int,
        preserve_recent: int,
    ) -> dict[str, Any]:
        """Keep only recent + high-importance content."""
        try:
            data = json.loads(context)
            if not isinstance(data, dict):
                return self._compress_text_context(context, max_tokens)

            # Keep only the most recent entries
            stages = [(k, v) for k, v in data.items() if isinstance(v, dict)]
            stages.sort(key=lambda x: x[1].get("stage_index", 0), reverse=True)

            selected = {}
            key_findings = []
            tokens_used = 0

            for key, value in stages[:preserve_recent + 1]:
                content = value.get("content", "")
                tokens = self.estimate_tokens(json.dumps({key: value})).estimated_tokens

                if tokens_used + tokens <= max_tokens:
                    selected[key] = value
                    tokens_used += tokens

                    # Extract key finding
                    findings = self._extract_key_findings(content, 100)
                    if findings["findings"]:
                        key_findings.append(findings["findings"][0])

            return {
                "text": json.dumps(selected, indent=1),
                "key_findings": key_findings,
            }

        except json.JSONDecodeError:
            return self._compress_text_context(context, max_tokens)

    def _extract_key_findings(
        self,
        content: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Extract key findings from content."""
        findings = []

        # Look for explicit key sections
        key_patterns = [
            r"\*\*([^*]+)\*\*:\s*([^\n]+)",  # **Key**: value
            r"^[-*]\s+(.+)$",  # Bullet points
            r"^\d+\.\s+(.+)$",  # Numbered lists
            r"^###?\s+(.+)$",  # Headers
        ]

        for pattern in key_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches[:5]:
                if isinstance(match, tuple):
                    finding = f"{match[0]}: {match[1]}" if len(match) > 1 else match[0]
                else:
                    finding = match
                if len(finding) > 10:
                    findings.append(finding[:200])

        # Create brief summary
        sentences = re.split(r"(?<=[.!?])\s+", content)
        summary_sentences = [s for s in sentences[:3] if len(s) > 20]
        summary = " ".join(summary_sentences)[:max_tokens * 4]

        return {
            "findings": findings[:5],
            "summary": summary,
        }

    def _score_sentence(
        self,
        sentence: str,
        position: int,
        total: int,
    ) -> float:
        """Score sentence importance (0-1)."""
        score = 0.5

        # Position bias (first and last sentences often important)
        if position < 3:
            score += 0.2
        if position >= total - 2:
            score += 0.1

        # Key phrase indicators
        key_phrases = [
            "key", "important", "critical", "essential", "main",
            "primary", "core", "fundamental", "insight", "finding",
            "conclusion", "result", "therefore", "thus", "in summary",
        ]
        for phrase in key_phrases:
            if phrase in sentence.lower():
                score += 0.1
                break

        # Formatting indicators (markdown emphasis)
        if "**" in sentence or "##" in sentence:
            score += 0.1

        # Penalize very short or very long sentences
        word_count = len(sentence.split())
        if word_count < 5:
            score -= 0.2
        elif word_count > 50:
            score -= 0.1

        return min(1.0, max(0.0, score))

    def _cache_key(
        self,
        context: str,
        max_tokens: int,
        strategy: CompressionStrategy,
    ) -> str:
        """Generate cache key for compression result."""
        content_hash = hashlib.md5(context.encode()).hexdigest()[:16]
        return f"{content_hash}_{max_tokens}_{strategy.value}"

    def clear_cache(self) -> None:
        """Clear compression cache."""
        self._cache.clear()


# Default instance
_default_optimizer: TokenOptimizer | None = None


def get_token_optimizer() -> TokenOptimizer:
    """Get the default token optimizer instance."""
    global _default_optimizer
    if _default_optimizer is None:
        _default_optimizer = TokenOptimizer()
    return _default_optimizer
