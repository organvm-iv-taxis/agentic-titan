"""
Titan Safety - Output Sanitizer

Provides output sanitization utilities for cleaning LLM responses.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("titan.safety.sanitizer")


@dataclass
class SanitizationConfig:
    """Configuration for output sanitization."""

    # HTML/XSS protection
    escape_html: bool = True
    strip_scripts: bool = True
    strip_event_handlers: bool = True

    # Credential protection
    redact_api_keys: bool = True
    redact_passwords: bool = True
    redact_tokens: bool = True

    # PII protection
    redact_ssn: bool = True
    redact_credit_cards: bool = True
    partial_redact_emails: bool = True
    partial_redact_phones: bool = False

    # Code safety
    sanitize_shell_commands: bool = True
    strip_dangerous_imports: bool = True

    # Limits
    max_length: int | None = None  # None = no limit
    truncate_marker: str = "... [truncated]"


@dataclass
class SanitizationResult:
    """Result of sanitization operation."""

    original: str
    sanitized: str
    changes_made: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def was_modified(self) -> bool:
        """Check if content was modified."""
        return self.original != self.sanitized

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "was_modified": self.was_modified,
            "changes_made": self.changes_made,
            "warnings": self.warnings,
            "truncated": self.truncated,
            "original_length": len(self.original),
            "sanitized_length": len(self.sanitized),
        }


class OutputSanitizer:
    """
    Sanitizes LLM outputs to remove dangerous content.

    Applies configurable sanitization rules to clean output
    before presenting to users or using in downstream operations.
    """

    def __init__(self, config: SanitizationConfig | None = None) -> None:
        self.config = config or SanitizationConfig()

    def sanitize(self, content: str) -> SanitizationResult:
        """
        Sanitize content according to configuration.

        Args:
            content: Content to sanitize

        Returns:
            SanitizationResult with sanitized content and metadata
        """
        result = SanitizationResult(original=content, sanitized=content)

        # Apply each sanitization rule
        if self.config.escape_html:
            result = self._escape_html(result)

        if self.config.strip_scripts:
            result = self._strip_scripts(result)

        if self.config.strip_event_handlers:
            result = self._strip_event_handlers(result)

        if self.config.redact_api_keys:
            result = self._redact_api_keys(result)

        if self.config.redact_passwords:
            result = self._redact_passwords(result)

        if self.config.redact_tokens:
            result = self._redact_tokens(result)

        if self.config.redact_ssn:
            result = self._redact_ssn(result)

        if self.config.redact_credit_cards:
            result = self._redact_credit_cards(result)

        if self.config.partial_redact_emails:
            result = self._partial_redact_emails(result)

        if self.config.partial_redact_phones:
            result = self._partial_redact_phones(result)

        if self.config.sanitize_shell_commands:
            result = self._sanitize_shell_commands(result)

        if self.config.strip_dangerous_imports:
            result = self._strip_dangerous_imports(result)

        # Apply length limit last
        if self.config.max_length and len(result.sanitized) > self.config.max_length:
            result.sanitized = result.sanitized[: self.config.max_length - len(self.config.truncate_marker)]
            result.sanitized += self.config.truncate_marker
            result.truncated = True
            result.changes_made.append(f"truncated to {self.config.max_length} chars")

        return result

    def _escape_html(self, result: SanitizationResult) -> SanitizationResult:
        """Escape HTML entities."""
        # Don't escape if already escaped or in code blocks
        content = result.sanitized

        # Preserve code blocks
        code_blocks: list[tuple[str, str]] = []
        code_pattern = r"(```[\s\S]*?```|`[^`]+`)"
        for i, match in enumerate(re.finditer(code_pattern, content)):
            placeholder = f"__CODE_BLOCK_{i}__"
            code_blocks.append((placeholder, match.group()))

        # Replace code blocks with placeholders
        for placeholder, block in code_blocks:
            content = content.replace(block, placeholder, 1)

        # Escape remaining HTML
        if re.search(r"<[^>]+>", content):
            content = html.escape(content)
            result.changes_made.append("escaped HTML entities")

        # Restore code blocks
        for placeholder, block in code_blocks:
            content = content.replace(placeholder, block, 1)

        result.sanitized = content
        return result

    def _strip_scripts(self, result: SanitizationResult) -> SanitizationResult:
        """Remove script tags."""
        pattern = r"<script[^>]*>[\s\S]*?</script>|<script[^>]*/>"
        if re.search(pattern, result.sanitized, re.IGNORECASE):
            result.sanitized = re.sub(pattern, "[script removed]", result.sanitized, flags=re.IGNORECASE)
            result.changes_made.append("removed script tags")
        return result

    def _strip_event_handlers(self, result: SanitizationResult) -> SanitizationResult:
        """Remove JavaScript event handlers."""
        pattern = r'\s*on\w+\s*=\s*["\'][^"\']*["\']'
        if re.search(pattern, result.sanitized, re.IGNORECASE):
            result.sanitized = re.sub(pattern, "", result.sanitized, flags=re.IGNORECASE)
            result.changes_made.append("removed event handlers")
        return result

    def _redact_api_keys(self, result: SanitizationResult) -> SanitizationResult:
        """Redact API keys."""
        patterns = [
            (r"(?i)(api[_-]?key|apikey|api_secret)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{8,})['\"]?", r"\1=[REDACTED]"),
            (r"(?i)(AKIA[0-9A-Z]{16})", "[AWS_KEY_REDACTED]"),
            (r"sk-[a-zA-Z0-9]{20,}", "[OPENAI_KEY_REDACTED]"),
            (r"ghp_[a-zA-Z0-9]{36}", "[GITHUB_TOKEN_REDACTED]"),
        ]

        redacted = False
        for pattern, replacement in patterns:
            if re.search(pattern, result.sanitized):
                result.sanitized = re.sub(pattern, replacement, result.sanitized)
                redacted = True

        if redacted:
            result.changes_made.append("redacted API keys")

        return result

    def _redact_passwords(self, result: SanitizationResult) -> SanitizationResult:
        """Redact passwords, preserving code blocks."""
        content = result.sanitized

        # Preserve code blocks
        code_blocks: list[tuple[str, str]] = []
        code_pattern = r"(```[\s\S]*?```|`[^`]+`)"
        for i, match in enumerate(re.finditer(code_pattern, content)):
            placeholder = f"__PWD_CODE_BLOCK_{i}__"
            code_blocks.append((placeholder, match.group()))

        # Replace code blocks with placeholders
        for placeholder, block in code_blocks:
            content = content.replace(block, placeholder, 1)

        # Redact passwords outside code blocks
        pattern = r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"]{8,})['\"]?"
        if re.search(pattern, content):
            content = re.sub(pattern, r"\1=[REDACTED]", content)
            result.changes_made.append("redacted passwords")

        # Restore code blocks
        for placeholder, block in code_blocks:
            content = content.replace(placeholder, block, 1)

        result.sanitized = content
        return result

    def _redact_tokens(self, result: SanitizationResult) -> SanitizationResult:
        """Redact bearer tokens and JWTs."""
        patterns = [
            (r"(?i)(bearer\s+)[a-zA-Z0-9_\-\.]+", r"\1[REDACTED]"),
            (r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*", "[JWT_REDACTED]"),
        ]

        for pattern, replacement in patterns:
            if re.search(pattern, result.sanitized):
                result.sanitized = re.sub(pattern, replacement, result.sanitized)
                result.changes_made.append("redacted tokens")
                break

        return result

    def _redact_ssn(self, result: SanitizationResult) -> SanitizationResult:
        """Redact Social Security Numbers."""
        pattern = r"\b(\d{3})[-\s]?(\d{2})[-\s]?(\d{4})\b"
        if re.search(pattern, result.sanitized):
            result.sanitized = re.sub(pattern, r"XXX-XX-\3", result.sanitized)
            result.changes_made.append("redacted SSN")
        return result

    def _redact_credit_cards(self, result: SanitizationResult) -> SanitizationResult:
        """Redact credit card numbers."""
        # Visa, MasterCard, Amex, Discover
        pattern = r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"
        if re.search(pattern, result.sanitized):
            result.sanitized = re.sub(
                pattern,
                lambda m: "**** **** **** " + m.group()[-4:],
                result.sanitized
            )
            result.changes_made.append("redacted credit cards")
        return result

    def _partial_redact_emails(self, result: SanitizationResult) -> SanitizationResult:
        """Partially redact email addresses."""
        pattern = r"\b([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b"

        def redact_email(match: re.Match[str]) -> str:
            local = match.group(1)
            domain = match.group(2)
            if len(local) > 2:
                local = local[0] + "***" + local[-1]
            return f"{local}@{domain}"

        if re.search(pattern, result.sanitized):
            result.sanitized = re.sub(pattern, redact_email, result.sanitized)
            result.changes_made.append("partially redacted emails")

        return result

    def _partial_redact_phones(self, result: SanitizationResult) -> SanitizationResult:
        """Partially redact phone numbers."""
        pattern = r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b"

        def redact_phone(match: re.Match[str]) -> str:
            return f"(***) ***-{match.group(3)}"

        if re.search(pattern, result.sanitized):
            result.sanitized = re.sub(pattern, redact_phone, result.sanitized)
            result.changes_made.append("partially redacted phone numbers")

        return result

    def _sanitize_shell_commands(self, result: SanitizationResult) -> SanitizationResult:
        """Warn about dangerous shell commands (don't modify code blocks)."""
        dangerous_patterns = [
            r"rm\s+-rf\s+/",
            r">\s*/dev/sd[a-z]",
            r"dd\s+if=.+of=/dev/",
            r"chmod\s+777\s+/",
            r"curl\s+.+\|\s*(sh|bash)",
            r"wget\s+.+\|\s*(sh|bash)",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, result.sanitized, re.IGNORECASE):
                result.warnings.append(f"Dangerous shell command detected: {pattern}")

        return result

    def _strip_dangerous_imports(self, result: SanitizationResult) -> SanitizationResult:
        """Warn about dangerous Python imports."""
        dangerous_imports = [
            r"import\s+subprocess",
            r"from\s+subprocess\s+import",
            r"import\s+os\s*;.*os\.system",
            r"__import__\s*\(",
            r"exec\s*\(",
            r"eval\s*\(",
        ]

        for pattern in dangerous_imports:
            if re.search(pattern, result.sanitized):
                result.warnings.append(f"Potentially dangerous code pattern: {pattern}")

        return result


# Singleton instance with default config
_default_sanitizer: OutputSanitizer | None = None


def get_sanitizer(config: SanitizationConfig | None = None) -> OutputSanitizer:
    """Get the default sanitizer or create one with custom config."""
    global _default_sanitizer
    if config:
        return OutputSanitizer(config)
    if _default_sanitizer is None:
        _default_sanitizer = OutputSanitizer()
    return _default_sanitizer


def sanitize_output(content: str, config: SanitizationConfig | None = None) -> str:
    """Convenience function to sanitize content."""
    sanitizer = get_sanitizer(config)
    result = sanitizer.sanitize(content)
    return result.sanitized
