"""
Titan Safety - Action Risk Classification

Classifies actions by risk level and defines policies for handling them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("titan.safety.policies")


class RiskLevel(str, Enum):
    """Risk levels for actions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionCategory(str, Enum):
    """Categories of actions."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    NETWORK = "network"
    AUTH = "auth"
    SYSTEM = "system"


@dataclass
class ActionPolicy:
    """
    Policy for handling an action.

    Defines whether an action requires approval and how to handle it.
    """

    risk_level: RiskLevel
    requires_approval: bool = False
    timeout_seconds: int = 300
    fallback_action: str = "deny"  # "deny" | "allow" | "escalate"
    max_retries: int = 1
    description: str = ""
    categories: list[ActionCategory] = field(default_factory=list)

    def should_approve_automatically(self) -> bool:
        """Check if action should be auto-approved based on risk level."""
        return self.risk_level == RiskLevel.LOW and not self.requires_approval


@dataclass
class ActionPattern:
    """Pattern for matching and classifying actions."""

    pattern: str
    policy: ActionPolicy
    compiled: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.compiled = re.compile(self.pattern, re.IGNORECASE)

    def matches(self, action: str) -> bool:
        """Check if action matches this pattern."""
        return bool(self.compiled.search(action))


# Default action patterns with risk classifications
DEFAULT_ACTION_PATTERNS = [
    # CRITICAL - Always require approval
    ActionPattern(
        pattern=r"(delete|remove|destroy)\s+(all|entire|database|system)",
        policy=ActionPolicy(
            risk_level=RiskLevel.CRITICAL,
            requires_approval=True,
            timeout_seconds=600,
            fallback_action="deny",
            description="Destructive operation affecting critical resources",
            categories=[ActionCategory.DELETE, ActionCategory.SYSTEM],
        ),
    ),
    ActionPattern(
        pattern=r"(credential|password|secret|api[_-]?key|token)\s*(leak|expose|send|share)",
        policy=ActionPolicy(
            risk_level=RiskLevel.CRITICAL,
            requires_approval=True,
            timeout_seconds=600,
            fallback_action="deny",
            description="Potential credential exposure",
            categories=[ActionCategory.AUTH],
        ),
    ),
    ActionPattern(
        pattern=r"(execute|run|eval)\s+(arbitrary|untrusted|user[_-]?input)",
        policy=ActionPolicy(
            risk_level=RiskLevel.CRITICAL,
            requires_approval=True,
            timeout_seconds=600,
            fallback_action="deny",
            description="Arbitrary code execution",
            categories=[ActionCategory.EXECUTE],
        ),
    ),
    # HIGH - Require approval
    ActionPattern(
        pattern=r"(delete|remove)\s+(file|directory|folder|message|email)",
        policy=ActionPolicy(
            risk_level=RiskLevel.HIGH,
            requires_approval=True,
            timeout_seconds=300,
            fallback_action="deny",
            description="File or data deletion",
            categories=[ActionCategory.DELETE],
        ),
    ),
    ActionPattern(
        pattern=r"(send|post|submit)\s+(email|message|request|data)",
        policy=ActionPolicy(
            risk_level=RiskLevel.HIGH,
            requires_approval=True,
            timeout_seconds=300,
            fallback_action="deny",
            description="External communication",
            categories=[ActionCategory.NETWORK],
        ),
    ),
    ActionPattern(
        pattern=r"(modify|change|update)\s+(permission|access|role|config)",
        policy=ActionPolicy(
            risk_level=RiskLevel.HIGH,
            requires_approval=True,
            timeout_seconds=300,
            fallback_action="deny",
            description="Permission or configuration change",
            categories=[ActionCategory.AUTH, ActionCategory.SYSTEM],
        ),
    ),
    ActionPattern(
        pattern=r"(connect|request)\s+(external|api|internet|network)",
        policy=ActionPolicy(
            risk_level=RiskLevel.HIGH,
            requires_approval=True,
            timeout_seconds=300,
            fallback_action="deny",
            description="External network connection",
            categories=[ActionCategory.NETWORK],
        ),
    ),
    ActionPattern(
        pattern=r"(execute|run)\s+(command|script|code)",
        policy=ActionPolicy(
            risk_level=RiskLevel.HIGH,
            requires_approval=True,
            timeout_seconds=300,
            fallback_action="deny",
            description="Code execution",
            categories=[ActionCategory.EXECUTE],
        ),
    ),
    # MEDIUM - May require approval based on context
    ActionPattern(
        pattern=r"(write|create|save)\s+(file|document|data)",
        policy=ActionPolicy(
            risk_level=RiskLevel.MEDIUM,
            requires_approval=False,
            timeout_seconds=120,
            fallback_action="deny",
            description="File write operation",
            categories=[ActionCategory.WRITE],
        ),
    ),
    ActionPattern(
        pattern=r"(install|download)\s+(package|dependency|module)",
        policy=ActionPolicy(
            risk_level=RiskLevel.MEDIUM,
            requires_approval=True,
            timeout_seconds=180,
            fallback_action="deny",
            description="Package installation",
            categories=[ActionCategory.SYSTEM, ActionCategory.NETWORK],
        ),
    ),
    ActionPattern(
        pattern=r"(git|database)\s+(commit|push|migrate)",
        policy=ActionPolicy(
            risk_level=RiskLevel.MEDIUM,
            requires_approval=True,
            timeout_seconds=120,
            fallback_action="deny",
            description="Version control or database operation",
            categories=[ActionCategory.WRITE, ActionCategory.SYSTEM],
        ),
    ),
    # LOW - Auto-approve
    ActionPattern(
        pattern=r"(read|get|fetch|list|search)[\s_\-]",
        policy=ActionPolicy(
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            timeout_seconds=60,
            fallback_action="allow",
            description="Read operation",
            categories=[ActionCategory.READ],
        ),
    ),
    ActionPattern(
        pattern=r"(analyze|check|validate|verify)",
        policy=ActionPolicy(
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            timeout_seconds=60,
            fallback_action="allow",
            description="Analysis operation",
            categories=[ActionCategory.READ],
        ),
    ),
]


class ActionClassifier:
    """
    Classifies actions by risk level.

    Uses pattern matching and configurable rules to determine
    the appropriate policy for each action.
    """

    def __init__(
        self,
        patterns: list[ActionPattern] | None = None,
        default_policy: ActionPolicy | None = None,
    ) -> None:
        self._patterns = patterns or DEFAULT_ACTION_PATTERNS
        self._default_policy = default_policy or ActionPolicy(
            risk_level=RiskLevel.MEDIUM,
            requires_approval=True,
            timeout_seconds=120,
            fallback_action="deny",
            description="Unknown action type",
        )

    def classify(self, action: str, context: dict[str, Any] | None = None) -> ActionPolicy:
        """
        Classify an action and return the appropriate policy.

        Args:
            action: Description of the action to classify
            context: Optional context for more accurate classification

        Returns:
            ActionPolicy for the action
        """
        # Check patterns in order (more specific first)
        for pattern in self._patterns:
            if pattern.matches(action):
                logger.debug(
                    f"Action '{action[:50]}...' matched pattern '{pattern.pattern}' "
                    f"-> {pattern.policy.risk_level.value}"
                )
                return pattern.policy

        # Apply context-based adjustments if available
        if context:
            adjusted = self._adjust_for_context(self._default_policy, context)
            return adjusted

        return self._default_policy

    def _adjust_for_context(
        self,
        policy: ActionPolicy,
        context: dict[str, Any],
    ) -> ActionPolicy:
        """Adjust policy based on context."""
        # Escalate risk if action involves external resources
        if context.get("involves_external", False):
            if policy.risk_level == RiskLevel.LOW:
                return ActionPolicy(
                    risk_level=RiskLevel.MEDIUM,
                    requires_approval=True,
                    timeout_seconds=policy.timeout_seconds,
                    fallback_action=policy.fallback_action,
                    description=f"{policy.description} (escalated: external)",
                    categories=policy.categories,
                )

        # Escalate if operating outside workspace
        if context.get("outside_workspace", False):
            if policy.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM):
                return ActionPolicy(
                    risk_level=RiskLevel.HIGH,
                    requires_approval=True,
                    timeout_seconds=policy.timeout_seconds,
                    fallback_action="deny",
                    description=f"{policy.description} (escalated: outside workspace)",
                    categories=policy.categories,
                )

        return policy

    def add_pattern(self, pattern: ActionPattern) -> None:
        """Add a custom pattern to the classifier."""
        self._patterns.insert(0, pattern)  # Custom patterns take priority

    def classify_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ActionPolicy:
        """
        Classify a tool call.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            ActionPolicy for the tool call
        """
        # Build action description from tool call
        action = f"{tool_name} {' '.join(str(v) for v in arguments.values())}"
        context = {
            "tool_name": tool_name,
            "arguments": arguments,
        }

        # Special handling for known dangerous tools
        if tool_name in ("shell", "bash", "exec", "execute_code"):
            return ActionPolicy(
                risk_level=RiskLevel.HIGH,
                requires_approval=True,
                timeout_seconds=300,
                fallback_action="deny",
                description=f"Shell execution: {tool_name}",
                categories=[ActionCategory.EXECUTE],
            )

        if tool_name in ("delete_file", "remove", "rm"):
            return ActionPolicy(
                risk_level=RiskLevel.HIGH,
                requires_approval=True,
                timeout_seconds=180,
                fallback_action="deny",
                description=f"File deletion: {tool_name}",
                categories=[ActionCategory.DELETE],
            )

        if tool_name in ("send_email", "send_message", "post"):
            return ActionPolicy(
                risk_level=RiskLevel.HIGH,
                requires_approval=True,
                timeout_seconds=180,
                fallback_action="deny",
                description=f"External communication: {tool_name}",
                categories=[ActionCategory.NETWORK],
            )

        return self.classify(action, context)


# Singleton instance
_default_classifier: ActionClassifier | None = None


def get_action_classifier() -> ActionClassifier:
    """Get the default action classifier."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = ActionClassifier()
    return _default_classifier
