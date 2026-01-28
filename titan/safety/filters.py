"""
Titan Safety - Content Filters

Implements content filtering for LLM outputs.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from titan.safety.patterns import (
    DangerousPattern,
    PatternCategory,
    PatternSeverity,
    get_all_patterns,
    get_patterns_by_category,
)

if TYPE_CHECKING:
    from titan.persistence.audit import AuditLogger

logger = logging.getLogger("titan.safety.filters")


@dataclass
class FilterMatch:
    """Result of a pattern match during filtering."""

    pattern_name: str
    category: PatternCategory
    severity: PatternSeverity
    matched_text: str
    position: tuple[int, int]  # start, end
    action: str  # block, sanitize, warn
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_name": self.pattern_name,
            "category": self.category.value,
            "severity": self.severity.value,
            "matched_text": self.matched_text[:50] + "..." if len(self.matched_text) > 50 else self.matched_text,
            "position": self.position,
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FilterResult:
    """Result of content filtering."""

    id: UUID = field(default_factory=uuid4)
    original_content: str = ""
    filtered_content: str | None = None
    blocked: bool = False
    sanitized: bool = False
    warnings: list[str] = field(default_factory=list)
    matches: list[FilterMatch] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_issues(self) -> bool:
        """Check if any issues were found."""
        return bool(self.matches)

    @property
    def critical_matches(self) -> list[FilterMatch]:
        """Get critical severity matches."""
        return [m for m in self.matches if m.severity == PatternSeverity.CRITICAL]

    @property
    def high_matches(self) -> list[FilterMatch]:
        """Get high severity matches."""
        return [m for m in self.matches if m.severity == PatternSeverity.HIGH]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "blocked": self.blocked,
            "sanitized": self.sanitized,
            "warnings": self.warnings,
            "match_count": len(self.matches),
            "critical_count": len(self.critical_matches),
            "high_count": len(self.high_matches),
            "matches": [m.to_dict() for m in self.matches[:10]],  # Limit for brevity
            "metadata": self.metadata,
        }


class ContentFilter(ABC):
    """Abstract base class for content filters."""

    name: str = "base_filter"
    description: str = "Base content filter"

    @abstractmethod
    def filter(self, content: str) -> FilterResult:
        """
        Filter content and return result.

        Args:
            content: Content to filter

        Returns:
            FilterResult with any matches and actions taken
        """
        pass

    @abstractmethod
    def sanitize(self, content: str, matches: list[FilterMatch]) -> str:
        """
        Sanitize content by removing or replacing matched patterns.

        Args:
            content: Original content
            matches: Pattern matches to sanitize

        Returns:
            Sanitized content
        """
        pass


class PatternBasedFilter(ContentFilter):
    """Filter that uses pattern matching."""

    def __init__(
        self,
        name: str,
        patterns: list[DangerousPattern],
        description: str = "",
    ) -> None:
        self.name = name
        self.description = description or f"Pattern-based filter: {name}"
        self._patterns = patterns

    def filter(self, content: str) -> FilterResult:
        """Filter content against configured patterns."""
        result = FilterResult(original_content=content)
        all_matches: list[FilterMatch] = []

        for pattern in self._patterns:
            matches = pattern.matches(content)
            for match in matches:
                filter_match = FilterMatch(
                    pattern_name=pattern.name,
                    category=pattern.category,
                    severity=pattern.severity,
                    matched_text=match.group(),
                    position=(match.start(), match.end()),
                    action=pattern.action,
                )
                all_matches.append(filter_match)

                if pattern.action == "block":
                    result.blocked = True
                elif pattern.action == "warn":
                    result.warnings.append(
                        f"{pattern.name}: {pattern.description}"
                    )

        result.matches = all_matches

        # Sanitize if needed (but not if blocked)
        if not result.blocked:
            sanitize_matches = [m for m in all_matches if m.action == "sanitize"]
            if sanitize_matches:
                result.filtered_content = self.sanitize(content, sanitize_matches)
                result.sanitized = True
            else:
                result.filtered_content = content
        else:
            result.filtered_content = None

        return result

    def sanitize(self, content: str, matches: list[FilterMatch]) -> str:
        """Replace matched patterns with redacted text."""
        # Sort matches by position (reverse order to preserve positions)
        sorted_matches = sorted(matches, key=lambda m: m.position[0], reverse=True)

        sanitized = content
        for match in sorted_matches:
            start, end = match.position
            replacement = f"[REDACTED:{match.pattern_name}]"
            sanitized = sanitized[:start] + replacement + sanitized[end:]

        return sanitized


class PromptInjectionFilter(PatternBasedFilter):
    """Filter for detecting prompt injection attempts."""

    def __init__(self) -> None:
        super().__init__(
            name="prompt_injection",
            patterns=get_patterns_by_category(PatternCategory.PROMPT_INJECTION),
            description="Detects prompt injection and jailbreak attempts",
        )


class CredentialLeakFilter(PatternBasedFilter):
    """Filter for detecting credential exposure."""

    def __init__(self) -> None:
        super().__init__(
            name="credential_leak",
            patterns=get_patterns_by_category(PatternCategory.CREDENTIAL_LEAK),
            description="Detects API keys, passwords, and other credentials",
        )


class CommandInjectionFilter(PatternBasedFilter):
    """Filter for detecting shell command injection."""

    def __init__(self) -> None:
        super().__init__(
            name="command_injection",
            patterns=get_patterns_by_category(PatternCategory.COMMAND_INJECTION),
            description="Detects dangerous shell commands and injection attempts",
        )


class PathTraversalFilter(PatternBasedFilter):
    """Filter for detecting path traversal attacks."""

    def __init__(self) -> None:
        super().__init__(
            name="path_traversal",
            patterns=get_patterns_by_category(PatternCategory.PATH_TRAVERSAL),
            description="Detects path traversal and sensitive file access",
        )


class SQLInjectionFilter(PatternBasedFilter):
    """Filter for detecting SQL injection."""

    def __init__(self) -> None:
        super().__init__(
            name="sql_injection",
            patterns=get_patterns_by_category(PatternCategory.SQL_INJECTION),
            description="Detects SQL injection attempts",
        )


class XSSFilter(PatternBasedFilter):
    """Filter for detecting cross-site scripting."""

    def __init__(self) -> None:
        super().__init__(
            name="xss",
            patterns=get_patterns_by_category(PatternCategory.XSS),
            description="Detects cross-site scripting (XSS) attempts",
        )


class PIIFilter(PatternBasedFilter):
    """Filter for detecting personally identifiable information."""

    def __init__(self) -> None:
        super().__init__(
            name="pii",
            patterns=get_patterns_by_category(PatternCategory.PII_EXPOSURE),
            description="Detects PII like SSN, credit cards, phone numbers",
        )


class CompositeFilter(ContentFilter):
    """
    Composite filter that combines multiple filters.

    Runs all child filters and aggregates results.
    """

    def __init__(
        self,
        name: str = "composite",
        filters: list[ContentFilter] | None = None,
        description: str = "Composite content filter",
    ) -> None:
        self.name = name
        self.description = description
        self._filters = filters or []

    def add_filter(self, filter_: ContentFilter) -> None:
        """Add a filter to the composite."""
        self._filters.append(filter_)

    def filter(self, content: str) -> FilterResult:
        """Run all filters and aggregate results."""
        result = FilterResult(original_content=content)
        all_matches: list[FilterMatch] = []

        filtered_content = content

        for child_filter in self._filters:
            child_result = child_filter.filter(filtered_content)

            all_matches.extend(child_result.matches)

            if child_result.blocked:
                result.blocked = True
                break  # Stop processing if blocked

            if child_result.sanitized and child_result.filtered_content:
                filtered_content = child_result.filtered_content
                result.sanitized = True

            result.warnings.extend(child_result.warnings)

        result.matches = all_matches
        result.filtered_content = filtered_content if not result.blocked else None

        return result

    def sanitize(self, content: str, matches: list[FilterMatch]) -> str:
        """Sanitize using all child filters."""
        sanitized = content
        for child_filter in self._filters:
            child_matches = [m for m in matches if m.pattern_name in
                           [p.name for p in getattr(child_filter, '_patterns', [])]]
            if child_matches:
                sanitized = child_filter.sanitize(sanitized, child_matches)
        return sanitized


class FilterPipeline:
    """
    Pipeline for running content through multiple filter stages.

    Supports async audit logging and configurable filter chains.
    """

    def __init__(
        self,
        filters: list[ContentFilter] | None = None,
        audit_logger: AuditLogger | None = None,
        block_on_critical: bool = True,
        block_on_high: bool = False,
    ) -> None:
        self._filters = filters or self._default_filters()
        self._audit_logger = audit_logger
        self._block_on_critical = block_on_critical
        self._block_on_high = block_on_high

    def _default_filters(self) -> list[ContentFilter]:
        """Create default filter set."""
        return [
            PromptInjectionFilter(),
            CredentialLeakFilter(),
            CommandInjectionFilter(),
            PathTraversalFilter(),
        ]

    def add_filter(self, filter_: ContentFilter) -> None:
        """Add a filter to the pipeline."""
        self._filters.append(filter_)

    async def filter(
        self,
        content: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> FilterResult:
        """
        Run content through all filters.

        Args:
            content: Content to filter
            agent_id: Agent ID for logging
            session_id: Session ID for logging

        Returns:
            Combined FilterResult
        """
        result = FilterResult(original_content=content)
        all_matches: list[FilterMatch] = []
        filtered_content = content
        is_blocked = False

        # Run all filters to accumulate all matches
        for filter_ in self._filters:
            filter_result = filter_.filter(filtered_content)
            all_matches.extend(filter_result.matches)

            # Track if any filter blocks
            if filter_result.blocked:
                is_blocked = True

            # Only sanitize if not blocked
            if not is_blocked and filter_result.sanitized and filter_result.filtered_content:
                filtered_content = filter_result.filtered_content
                result.sanitized = True

            result.warnings.extend(filter_result.warnings)

        result.matches = all_matches

        # Apply blocking
        if is_blocked:
            result.blocked = True
            result.filtered_content = None
        else:
            result.filtered_content = filtered_content

            # Check severity-based blocking
            if self._block_on_critical and result.critical_matches:
                result.blocked = True
                result.filtered_content = None
            elif self._block_on_high and result.high_matches:
                result.blocked = True
                result.filtered_content = None

        # Audit log the result
        await self._audit_filter_result(result, agent_id, session_id)

        return result

    async def _audit_filter_result(
        self,
        result: FilterResult,
        agent_id: str | None,
        session_id: str | None,
    ) -> None:
        """Log filter result to audit."""
        if not self._audit_logger or not result.has_issues:
            return

        try:
            for match in result.matches[:5]:  # Limit audit entries
                await self._audit_logger.log_content_filtered(
                    agent_id=agent_id or "unknown",
                    session_id=session_id or "unknown",
                    filter_type=match.pattern_name,
                    original_content=result.original_content[:200],
                    filtered_content=result.filtered_content[:200] if result.filtered_content else None,
                    reason=f"{match.category.value}: {match.pattern_name}",
                )
        except Exception as e:
            logger.warning(f"Failed to audit filter result: {e}")


def create_default_pipeline(audit_logger: AuditLogger | None = None) -> FilterPipeline:
    """Create a default filter pipeline with all filters."""
    return FilterPipeline(
        filters=[
            PromptInjectionFilter(),
            CredentialLeakFilter(),
            CommandInjectionFilter(),
            PathTraversalFilter(),
            SQLInjectionFilter(),
            XSSFilter(),
            PIIFilter(),
        ],
        audit_logger=audit_logger,
        block_on_critical=True,
        block_on_high=False,
    )
