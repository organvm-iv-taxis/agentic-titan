"""
Agentic Titan - Persona System

Provides persona-based logging and communication for agents.
Each persona has a distinct identity and communication style.

Ported from: my--father-mother (dual-persona model)
Extended with: Multiple agent personas, color coding, rich output
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, TextIO

try:
    from rich.console import Console
    from rich.theme import Theme

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


logger = logging.getLogger("titan.personas")


class PersonaRole(Enum):
    """Agent roles in the swarm."""

    ORCHESTRATOR = "orchestrator"  # Coordinates other agents
    RESEARCHER = "researcher"      # Gathers information
    CODER = "coder"               # Writes code
    REVIEWER = "reviewer"         # Reviews work
    PLANNER = "planner"           # Plans tasks
    EXECUTOR = "executor"         # Executes tasks
    LEARNER = "learner"           # Learns from interactions
    CUSTOM = "custom"             # Custom role


@dataclass
class Persona:
    """
    Agent persona with identity and styling.

    Personas give agents distinct identities for logging and communication.
    Inspired by the Mother/Father metaphor from my--father-mother.
    """

    name: str
    role: PersonaRole
    emoji: str = ""
    color: str = "white"
    traits: list[str] | None = None
    communication_style: str = "neutral"

    def __post_init__(self) -> None:
        self.traits = self.traits or []

    def format_message(self, message: str) -> str:
        """Format a message with persona styling."""
        prefix = f"{self.emoji} " if self.emoji else ""
        return f"[{prefix}{self.name}] {message}"

    def __str__(self) -> str:
        return f"{self.emoji} {self.name}" if self.emoji else self.name


# ============================================================================
# Pre-defined Personas
# ============================================================================

ORCHESTRATOR = Persona(
    name="Orchestrator",
    role=PersonaRole.ORCHESTRATOR,
    emoji="\U0001F3AF",  # Direct hit target
    color="cyan",
    traits=["commanding", "organized", "decisive"],
    communication_style="directive",
)

RESEARCHER = Persona(
    name="Researcher",
    role=PersonaRole.RESEARCHER,
    emoji="\U0001F50D",  # Magnifying glass
    color="blue",
    traits=["thorough", "curious", "skeptical"],
    communication_style="academic",
)

CODER = Persona(
    name="Coder",
    role=PersonaRole.CODER,
    emoji="\U0001F4BB",  # Laptop
    color="green",
    traits=["precise", "efficient", "pragmatic"],
    communication_style="technical",
)

REVIEWER = Persona(
    name="Reviewer",
    role=PersonaRole.REVIEWER,
    emoji="\U0001F50E",  # Right-pointing magnifying glass
    color="yellow",
    traits=["critical", "thorough", "constructive"],
    communication_style="evaluative",
)

PLANNER = Persona(
    name="Planner",
    role=PersonaRole.PLANNER,
    emoji="\U0001F4CB",  # Clipboard
    color="magenta",
    traits=["strategic", "systematic", "forward-thinking"],
    communication_style="structured",
)

EXECUTOR = Persona(
    name="Executor",
    role=PersonaRole.EXECUTOR,
    emoji="\u26A1",  # High voltage
    color="red",
    traits=["focused", "reliable", "efficient"],
    communication_style="action-oriented",
)

LEARNER = Persona(
    name="Learner",
    role=PersonaRole.LEARNER,
    emoji="\U0001F4D6",  # Open book
    color="white",
    traits=["adaptive", "reflective", "growth-oriented"],
    communication_style="inquisitive",
)


# ============================================================================
# Persona Registry
# ============================================================================

_PERSONA_REGISTRY: dict[str, Persona] = {
    "orchestrator": ORCHESTRATOR,
    "researcher": RESEARCHER,
    "coder": CODER,
    "reviewer": REVIEWER,
    "planner": PLANNER,
    "executor": EXECUTOR,
    "learner": LEARNER,
}


def register_persona(persona: Persona) -> None:
    """Register a custom persona."""
    _PERSONA_REGISTRY[persona.name.lower()] = persona


def get_persona(name: str) -> Persona | None:
    """Get a persona by name."""
    return _PERSONA_REGISTRY.get(name.lower())


def list_personas() -> list[Persona]:
    """List all registered personas."""
    return list(_PERSONA_REGISTRY.values())


# ============================================================================
# Output Functions
# ============================================================================

# Rich console for colored output (if available)
_console: Console | None = None

if RICH_AVAILABLE:
    _theme = Theme(
        {
            "orchestrator": "bold cyan",
            "researcher": "bold blue",
            "coder": "bold green",
            "reviewer": "bold yellow",
            "planner": "bold magenta",
            "executor": "bold red",
            "learner": "white",
            "timestamp": "dim",
            "error": "bold red",
            "warning": "bold yellow",
            "success": "bold green",
        }
    )
    _console = Console(theme=_theme)


def say(
    persona: Persona | str,
    message: str,
    *,
    level: str = "info",
    file: TextIO | None = None,
    timestamp: bool = True,
) -> None:
    """
    Output a message with persona styling.

    This is the primary way for agents to communicate.
    Inspired by my--father-mother's say() function.

    Args:
        persona: Persona or persona name
        message: Message to output
        level: Log level (info, warning, error, debug)
        file: Output file (default: stdout)
        timestamp: Include timestamp
    """
    # Resolve persona
    if isinstance(persona, str):
        persona = get_persona(persona) or Persona(
            name=persona,
            role=PersonaRole.CUSTOM,
        )

    # Build timestamp
    ts = ""
    if timestamp:
        ts = datetime.now().strftime("%H:%M:%S") + " "

    # Format message
    formatted = persona.format_message(message)

    # Output
    if RICH_AVAILABLE and _console is not None:
        style = persona.role.value if persona.role != PersonaRole.CUSTOM else "white"
        _console.print(f"[dim]{ts}[/dim][{style}]{formatted}[/{style}]")
    else:
        # Fallback to plain print
        output = file or sys.stdout
        print(f"{ts}{formatted}", file=output)

    # Also log
    log_func = getattr(logger, level, logger.info)
    log_func(f"[{persona.name}] {message}")


def announce(
    persona: Persona | str,
    title: str,
    details: dict[str, Any] | None = None,
) -> None:
    """
    Make an announcement with optional details.

    Args:
        persona: Persona making the announcement
        title: Announcement title
        details: Optional details to display
    """
    say(persona, f"=== {title} ===")
    if details:
        for key, value in details.items():
            say(persona, f"  {key}: {value}")


def think(
    persona: Persona | str,
    thought: str,
) -> None:
    """
    Express a thought (internal monologue style).

    Args:
        persona: Persona thinking
        thought: The thought
    """
    say(persona, f"[thinking] {thought}", level="debug")


def report_error(
    persona: Persona | str,
    error: str | Exception,
    context: str = "",
) -> None:
    """
    Report an error with persona context.

    Args:
        persona: Persona reporting
        error: Error or error message
        context: Additional context
    """
    error_msg = str(error)
    if context:
        error_msg = f"{context}: {error_msg}"
    say(persona, f"ERROR: {error_msg}", level="error")


def report_success(
    persona: Persona | str,
    message: str,
) -> None:
    """
    Report success with persona context.

    Args:
        persona: Persona reporting
        message: Success message
    """
    if RICH_AVAILABLE and _console is not None:
        _console.print(f"[success]{message}[/success]")
    say(persona, f"SUCCESS: {message}")
