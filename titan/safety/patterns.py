"""
Titan Safety - Dangerous Pattern Definitions

Defines patterns for detecting dangerous content in LLM outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PatternCategory(str, Enum):
    """Categories of dangerous patterns."""

    PROMPT_INJECTION = "prompt_injection"
    CREDENTIAL_LEAK = "credential_leak"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    PII_EXPOSURE = "pii_exposure"
    HARMFUL_CONTENT = "harmful_content"


class PatternSeverity(str, Enum):
    """Severity levels for pattern matches."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DangerousPattern:
    """
    Definition of a dangerous pattern to detect.

    Attributes:
        name: Human-readable name
        category: Pattern category
        severity: How dangerous a match is
        pattern: Regex pattern to match
        description: Explanation of why this is dangerous
        action: What to do on match (block, sanitize, warn)
    """

    name: str
    category: PatternCategory
    severity: PatternSeverity
    pattern: str
    description: str
    action: str = "block"  # block, sanitize, warn
    compiled: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.compiled = re.compile(self.pattern, re.IGNORECASE | re.MULTILINE)

    def matches(self, text: str) -> list[re.Match[str]]:
        """Find all matches in the text."""
        return list(self.compiled.finditer(text))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category.value,
            "severity": self.severity.value,
            "pattern": self.pattern,
            "description": self.description,
            "action": self.action,
        }


# Prompt Injection Patterns
PROMPT_INJECTION_PATTERNS = [
    DangerousPattern(
        name="ignore_instructions",
        category=PatternCategory.PROMPT_INJECTION,
        severity=PatternSeverity.CRITICAL,
        pattern=r"(ignore|disregard|forget)\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|guidelines?)",
        description="Attempt to override system instructions",
        action="block",
    ),
    DangerousPattern(
        name="ignore_your_instructions",
        category=PatternCategory.PROMPT_INJECTION,
        severity=PatternSeverity.CRITICAL,
        pattern=r"(ignore|disregard|forget)\s+your\s+(previous\s+|prior\s+|earlier\s+)?(instructions?|prompts?|rules?|guidelines?)",
        description="Attempt to override AI instructions",
        action="block",
    ),
    DangerousPattern(
        name="ignore_and_do",
        category=PatternCategory.PROMPT_INJECTION,
        severity=PatternSeverity.CRITICAL,
        pattern=r"ignore\s+the\s+(above|previous)\s+and\s+(instead\s+)?(do|say|tell|output)",
        description="Attempt to redirect AI behavior",
        action="block",
    ),
    DangerousPattern(
        name="new_persona",
        category=PatternCategory.PROMPT_INJECTION,
        severity=PatternSeverity.CRITICAL,
        pattern=r"(you\s+are\s+now|act\s+as\s+if|pretend\s+(to\s+be|you\s+are)|from\s+now\s+on\s+you)",
        description="Attempt to change AI persona",
        action="block",
    ),
    DangerousPattern(
        name="system_prompt_override",
        category=PatternCategory.PROMPT_INJECTION,
        severity=PatternSeverity.CRITICAL,
        pattern=r"(new\s+)?system\s+prompt|override\s+(system|safety)|bypass\s+(all\s+)?(restrictions?|filters?|safety)",
        description="Attempt to override system prompt",
        action="block",
    ),
    DangerousPattern(
        name="jailbreak_keywords",
        category=PatternCategory.PROMPT_INJECTION,
        severity=PatternSeverity.HIGH,
        pattern=r"(DAN|do\s+anything\s+now|jailbreak|unrestricted\s+mode|developer\s+mode|god\s+mode)",
        description="Known jailbreak attempt keywords",
        action="block",
    ),
    DangerousPattern(
        name="base64_injection",
        category=PatternCategory.PROMPT_INJECTION,
        severity=PatternSeverity.HIGH,
        pattern=r"(execute|run|decode|eval)\s+(the\s+)?base64",
        description="Attempt to execute base64-encoded content",
        action="block",
    ),
]

# Credential Leak Patterns
CREDENTIAL_LEAK_PATTERNS = [
    DangerousPattern(
        name="api_key_exposure",
        category=PatternCategory.CREDENTIAL_LEAK,
        severity=PatternSeverity.CRITICAL,
        pattern=r"(?i)(api[_-]?key|apikey|api_secret)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{8,})['\"]?",
        description="API key in plain text",
        action="sanitize",
    ),
    DangerousPattern(
        name="password_exposure",
        category=PatternCategory.CREDENTIAL_LEAK,
        severity=PatternSeverity.CRITICAL,
        pattern=r"(?i)(password|passwd|pwd)\s*(\s+is)?[:=]\s*['\"]?([^\s'\"]{8,})['\"]?",
        description="Password in plain text",
        action="sanitize",
    ),
    DangerousPattern(
        name="aws_credentials",
        category=PatternCategory.CREDENTIAL_LEAK,
        severity=PatternSeverity.CRITICAL,
        pattern=r"(AKIA[0-9A-Z]{16}|aws[_-]?(access[_-]?key|secret)[_-]?id?)",
        description="AWS credentials detected",
        action="sanitize",
    ),
    DangerousPattern(
        name="private_key",
        category=PatternCategory.CREDENTIAL_LEAK,
        severity=PatternSeverity.CRITICAL,
        pattern=r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
        description="Private key detected",
        action="block",
    ),
    DangerousPattern(
        name="jwt_token",
        category=PatternCategory.CREDENTIAL_LEAK,
        severity=PatternSeverity.HIGH,
        pattern=r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*",
        description="JWT token detected",
        action="sanitize",
    ),
    DangerousPattern(
        name="bearer_token",
        category=PatternCategory.CREDENTIAL_LEAK,
        severity=PatternSeverity.HIGH,
        pattern=r"(?i)bearer\s+[a-zA-Z0-9_\-\.]+",
        description="Bearer token detected",
        action="sanitize",
    ),
]

# Command Injection Patterns
COMMAND_INJECTION_PATTERNS = [
    DangerousPattern(
        name="shell_command_chaining",
        category=PatternCategory.COMMAND_INJECTION,
        severity=PatternSeverity.CRITICAL,
        pattern=r"[;&|`]\s*(rm|mv|dd|chmod|chown|kill|shutdown|reboot|mkfs|format)",
        description="Dangerous shell command chaining",
        action="block",
    ),
    DangerousPattern(
        name="command_substitution",
        category=PatternCategory.COMMAND_INJECTION,
        severity=PatternSeverity.HIGH,
        pattern=r"\$\([^)]+\)|\`[^`]+\`",
        description="Command substitution detected",
        action="warn",
    ),
    DangerousPattern(
        name="dangerous_commands",
        category=PatternCategory.COMMAND_INJECTION,
        severity=PatternSeverity.CRITICAL,
        pattern=r"(?i)(rm\s+-rf|dd\s+if=|>\s*/dev/|chmod\s+777|curl\s+.*\|\s*(sh|bash)|wget\s+.*\|\s*(sh|bash))",
        description="Dangerous command detected",
        action="block",
    ),
    DangerousPattern(
        name="reverse_shell",
        category=PatternCategory.COMMAND_INJECTION,
        severity=PatternSeverity.CRITICAL,
        pattern=r"(?i)(nc\s+-[elp]|/bin/(ba)?sh\s+-i|python\s+-c.*socket|bash\s+-i\s+>&\s+/dev/tcp)",
        description="Potential reverse shell",
        action="block",
    ),
]

# Path Traversal Patterns
PATH_TRAVERSAL_PATTERNS = [
    DangerousPattern(
        name="path_traversal",
        category=PatternCategory.PATH_TRAVERSAL,
        severity=PatternSeverity.HIGH,
        pattern=r"\.\.[\\/]|\.\.%2[fF]|%2[eE]%2[eE][\\/]",
        description="Path traversal attempt",
        action="block",
    ),
    DangerousPattern(
        name="sensitive_files",
        category=PatternCategory.PATH_TRAVERSAL,
        severity=PatternSeverity.HIGH,
        pattern=r"(?i)(/etc/passwd|/etc/shadow|\.ssh/|\.aws/credentials|\.env)",
        description="Access to sensitive files",
        action="block",
    ),
]

# SQL Injection Patterns
SQL_INJECTION_PATTERNS = [
    DangerousPattern(
        name="sql_union",
        category=PatternCategory.SQL_INJECTION,
        severity=PatternSeverity.HIGH,
        pattern=r"(?i)\bunion\s+(all\s+)?select\b",
        description="SQL UNION injection",
        action="block",
    ),
    DangerousPattern(
        name="sql_comment",
        category=PatternCategory.SQL_INJECTION,
        severity=PatternSeverity.MEDIUM,
        pattern=r"(--)|(#)|(/\*)",
        description="SQL comment injection",
        action="warn",
    ),
    DangerousPattern(
        name="sql_or_bypass",
        category=PatternCategory.SQL_INJECTION,
        severity=PatternSeverity.HIGH,
        pattern=r"(?i)'\s*or\s+'?1'?\s*=\s*'?1'?|'\s*or\s+'?'?\s*=\s*'?'?",
        description="SQL OR bypass attempt",
        action="block",
    ),
    DangerousPattern(
        name="sql_drop",
        category=PatternCategory.SQL_INJECTION,
        severity=PatternSeverity.CRITICAL,
        pattern=r"(?i)\b(drop|truncate|delete\s+from)\s+(table|database)\b",
        description="SQL destructive operation",
        action="block",
    ),
]

# XSS Patterns
XSS_PATTERNS = [
    DangerousPattern(
        name="script_tag",
        category=PatternCategory.XSS,
        severity=PatternSeverity.HIGH,
        pattern=r"<script[^>]*>.*?</script>|<script[^>]*/>",
        description="Script tag detected",
        action="sanitize",
    ),
    DangerousPattern(
        name="event_handler",
        category=PatternCategory.XSS,
        severity=PatternSeverity.HIGH,
        pattern=r"(?i)\bon(click|load|error|mouseover|focus|blur|change|submit)\s*=",
        description="JavaScript event handler",
        action="sanitize",
    ),
    DangerousPattern(
        name="javascript_url",
        category=PatternCategory.XSS,
        severity=PatternSeverity.HIGH,
        pattern=r"(?i)javascript\s*:",
        description="JavaScript URL scheme",
        action="sanitize",
    ),
]

# PII Exposure Patterns
PII_PATTERNS = [
    DangerousPattern(
        name="ssn",
        category=PatternCategory.PII_EXPOSURE,
        severity=PatternSeverity.CRITICAL,
        pattern=r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        description="Social Security Number format",
        action="sanitize",
    ),
    DangerousPattern(
        name="credit_card",
        category=PatternCategory.PII_EXPOSURE,
        severity=PatternSeverity.CRITICAL,
        pattern=r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
        description="Credit card number format",
        action="sanitize",
    ),
    DangerousPattern(
        name="phone_number",
        category=PatternCategory.PII_EXPOSURE,
        severity=PatternSeverity.MEDIUM,
        pattern=r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
        description="Phone number format",
        action="warn",
    ),
]

# All patterns grouped by category
ALL_PATTERNS: dict[PatternCategory, list[DangerousPattern]] = {
    PatternCategory.PROMPT_INJECTION: PROMPT_INJECTION_PATTERNS,
    PatternCategory.CREDENTIAL_LEAK: CREDENTIAL_LEAK_PATTERNS,
    PatternCategory.COMMAND_INJECTION: COMMAND_INJECTION_PATTERNS,
    PatternCategory.PATH_TRAVERSAL: PATH_TRAVERSAL_PATTERNS,
    PatternCategory.SQL_INJECTION: SQL_INJECTION_PATTERNS,
    PatternCategory.XSS: XSS_PATTERNS,
    PatternCategory.PII_EXPOSURE: PII_PATTERNS,
}


def get_all_patterns() -> list[DangerousPattern]:
    """Get all defined patterns."""
    patterns = []
    for category_patterns in ALL_PATTERNS.values():
        patterns.extend(category_patterns)
    return patterns


def get_patterns_by_category(category: PatternCategory) -> list[DangerousPattern]:
    """Get patterns for a specific category."""
    return ALL_PATTERNS.get(category, [])


def get_patterns_by_severity(severity: PatternSeverity) -> list[DangerousPattern]:
    """Get all patterns of a specific severity."""
    return [p for p in get_all_patterns() if p.severity == severity]
