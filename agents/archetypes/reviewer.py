"""
Reviewer Agent - Reviews work for quality.

Capabilities:
- Code review
- Document review
- Quality assessment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agents.framework.base_agent import BaseAgent
from agents.personas import REVIEWER, say, think
from adapters.base import LLMMessage
from adapters.router import get_router

logger = logging.getLogger("titan.agents.reviewer")


class ReviewSeverity(str, Enum):
    """Review comment severity levels."""

    CRITICAL = "critical"    # Must fix
    SUGGESTION = "suggestion"  # Should consider
    NITPICK = "nitpick"       # Minor preference
    PRAISE = "praise"         # Good pattern to highlight


@dataclass
class ReviewComment:
    """A review comment."""

    severity: ReviewSeverity
    message: str
    line: int | None = None
    file: str | None = None


@dataclass
class ReviewResult:
    """Result of a review."""

    target: str
    review_type: str
    comments: list[ReviewComment] = field(default_factory=list)
    summary: str = ""
    approved: bool = False
    confidence: float = 0.0

    @property
    def critical_count(self) -> int:
        return sum(1 for c in self.comments if c.severity == ReviewSeverity.CRITICAL)

    @property
    def suggestion_count(self) -> int:
        return sum(1 for c in self.comments if c.severity == ReviewSeverity.SUGGESTION)


class ReviewerAgent(BaseAgent):
    """
    Agent specialized in reviewing work for quality.

    Reviews:
    - Code for bugs, security, style
    - Documents for accuracy, clarity
    - Agent outputs for correctness
    """

    def __init__(
        self,
        content: str | None = None,
        review_type: str = "code",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="reviewer",
            capabilities=["code_review", "document_analysis", "summarization"],
            **kwargs,
        )
        self.content = content
        self.review_type = review_type
        self.result: ReviewResult | None = None
        self._router = get_router()

    async def initialize(self) -> None:
        """Initialize the reviewer agent."""
        say(REVIEWER, f"Initializing reviewer agent (type: {self.review_type})")
        await self._router.initialize()

    async def work(self) -> ReviewResult:
        """
        Execute review workflow.

        Returns:
            ReviewResult with comments and verdict
        """
        if not self.content:
            say(REVIEWER, "No content to review")
            return ReviewResult(target="", review_type=self.review_type)

        say(REVIEWER, f"Beginning {self.review_type} review...")
        self.result = ReviewResult(
            target=self.content[:50] + "...",
            review_type=self.review_type,
        )

        # Step 1: Initial scan
        think(REVIEWER, "Performing initial scan...")
        self.increment_turn()

        # Step 2: Detailed review based on type
        if self.review_type == "code":
            comments = await self._review_code()
        else:
            comments = await self._review_document()

        self.result.comments = comments

        # Step 3: Generate summary
        think(REVIEWER, "Generating review summary...")
        self.increment_turn()
        summary = await self._generate_summary()
        self.result.summary = summary

        # Step 4: Make approval decision
        self.result.approved = self.result.critical_count == 0
        self.result.confidence = self._calculate_confidence()

        # Log decision
        await self.log_decision(
            decision=f"Review {'approved' if self.result.approved else 'rejected'}",
            category="review",
            rationale=f"Found {self.result.critical_count} critical issues",
            tags=["review", self.review_type],
        )

        status = "[green]APPROVED[/green]" if self.result.approved else "[red]CHANGES REQUESTED[/red]"
        say(REVIEWER, f"Review complete: {status}")
        say(REVIEWER, f"Critical: {self.result.critical_count}, Suggestions: {self.result.suggestion_count}")

        return self.result

    async def shutdown(self) -> None:
        """Cleanup reviewer agent."""
        say(REVIEWER, "Reviewer agent shutting down")

        # Store review patterns
        if self._hive_mind and self.result:
            await self.remember(
                content=f"Review of {self.review_type}: {self.result.summary}",
                importance=0.5,
                tags=["review", self.review_type],
            )

    async def _review_code(self) -> list[ReviewComment]:
        """Perform code review."""
        messages = [
            LLMMessage(
                role="user",
                content=f"""Review this code for issues:

```
{self.content}
```

Check for:
1. Bugs and logic errors (CRITICAL)
2. Security vulnerabilities (CRITICAL)
3. Performance issues (SUGGESTION)
4. Code style and readability (SUGGESTION/NITPICK)
5. Good patterns to highlight (PRAISE)

Format each comment as:
SEVERITY: message

Where SEVERITY is one of: CRITICAL, SUGGESTION, NITPICK, PRAISE""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a senior code reviewer. Be thorough but constructive.",
            max_tokens=1000,
        )

        return self._parse_comments(response.content)

    async def _review_document(self) -> list[ReviewComment]:
        """Perform document review."""
        messages = [
            LLMMessage(
                role="user",
                content=f"""Review this document for quality:

{self.content}

Check for:
1. Factual accuracy (CRITICAL)
2. Clarity and organization (SUGGESTION)
3. Grammar and style (NITPICK)
4. Strong sections (PRAISE)

Format each comment as:
SEVERITY: message""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are an editor. Review for clarity and accuracy.",
            max_tokens=1000,
        )

        return self._parse_comments(response.content)

    def _parse_comments(self, content: str) -> list[ReviewComment]:
        """Parse review comments from LLM response."""
        comments = []

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            for severity in ReviewSeverity:
                prefix = f"{severity.value.upper()}:"
                if line.upper().startswith(prefix):
                    message = line[len(prefix):].strip()
                    comments.append(ReviewComment(severity=severity, message=message))
                    break

        return comments

    async def _generate_summary(self) -> str:
        """Generate review summary."""
        if not self.result:
            return ""

        comments_text = "\n".join(
            f"- [{c.severity.value}] {c.message}" for c in self.result.comments
        )

        messages = [
            LLMMessage(
                role="user",
                content=f"""Summarize this {self.review_type} review:

Comments:
{comments_text}

Provide:
1. Overall assessment (1-2 sentences)
2. Most important points
3. Recommendation

Keep it under 100 words.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a review summarizer. Be concise and actionable.",
            max_tokens=200,
        )

        return response.content

    def _calculate_confidence(self) -> float:
        """Calculate confidence in review."""
        if not self.result:
            return 0.0

        # More comments = more thorough = higher confidence
        comment_score = min(1.0, len(self.result.comments) / 5)

        # Balance of comment types suggests thorough review
        has_critical = any(c.severity == ReviewSeverity.CRITICAL for c in self.result.comments)
        has_praise = any(c.severity == ReviewSeverity.PRAISE for c in self.result.comments)
        balance_score = 0.5 + (0.25 if has_critical else 0) + (0.25 if has_praise else 0)

        return (comment_score + balance_score) / 2
