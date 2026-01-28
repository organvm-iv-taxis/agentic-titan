"""
Project Context Loader - Load TITAN.md project configuration.

Implements the AGENTS.md pattern from codex-ai:
- Searches for TITAN.md in project root and parent directories
- Parses structured configuration sections
- Injects context into agent system prompts
- Supports hierarchical configuration (global + project-specific)

TITAN.md Format:
```markdown
# Project Name

## Overview
Project description and purpose.

## Architecture
Key architectural decisions and patterns.

## Conventions
Coding conventions, naming patterns, etc.

## Dependencies
Key dependencies and their purposes.

## Agent Instructions
Specific instructions for AI agents working on this project.
```

Reference: vendor/cli/codex-ai AGENTS.md pattern
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("titan.core.context")


# ============================================================================
# Configuration
# ============================================================================

# Files to search for (in order of priority)
CONTEXT_FILES = [
    "TITAN.md",
    "AGENTS.md",  # Compatibility with codex-ai
    ".titan/config.md",
    ".titan/context.md",
]

# Global context file location
GLOBAL_CONTEXT_PATH = Path.home() / ".titan" / "TITAN.md"

# Maximum file size to load (prevent memory issues)
MAX_CONTEXT_FILE_SIZE = 1_000_000  # 1MB


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class ProjectSection:
    """A section from the TITAN.md file."""

    name: str
    level: int  # Heading level (1-6)
    content: str
    subsections: list[ProjectSection] = field(default_factory=list)

    def to_markdown(self, include_heading: bool = True) -> str:
        """Convert section to markdown string."""
        parts = []
        if include_heading:
            parts.append(f"{'#' * self.level} {self.name}")
        if self.content.strip():
            parts.append(self.content.strip())
        for subsection in self.subsections:
            parts.append(subsection.to_markdown())
        return "\n\n".join(parts)


@dataclass
class ProjectContext:
    """
    Project context loaded from TITAN.md.

    Contains structured information about the project that can be
    injected into agent system prompts.
    """

    # Source information
    source_file: Path | None = None
    global_context_file: Path | None = None

    # Raw content
    raw_content: str = ""
    global_content: str = ""

    # Parsed sections
    sections: dict[str, ProjectSection] = field(default_factory=dict)

    # Extracted metadata
    project_name: str = ""
    description: str = ""

    # Special sections (commonly used)
    overview: str = ""
    architecture: str = ""
    conventions: str = ""
    dependencies: str = ""
    agent_instructions: str = ""

    # Custom sections
    custom_sections: dict[str, str] = field(default_factory=dict)

    @property
    def has_context(self) -> bool:
        """Check if any context was loaded."""
        return bool(self.raw_content or self.global_content)

    def to_system_prompt(self) -> str:
        """
        Generate a system prompt injection from the project context.

        Returns a formatted string suitable for prepending to agent system prompts.
        """
        if not self.has_context:
            return ""

        parts = ["# Project Context"]

        # Global context first (if exists)
        if self.global_content:
            parts.append("## Global Configuration")
            parts.append(self.global_content.strip())

        # Project-specific context
        if self.raw_content:
            parts.append("## Project Configuration")

            # Project name and description
            if self.project_name:
                parts.append(f"**Project:** {self.project_name}")
            if self.description:
                parts.append(f"\n{self.description}")

            # Key sections
            if self.overview:
                parts.append("\n### Overview")
                parts.append(self.overview)

            if self.architecture:
                parts.append("\n### Architecture")
                parts.append(self.architecture)

            if self.conventions:
                parts.append("\n### Conventions")
                parts.append(self.conventions)

            if self.agent_instructions:
                parts.append("\n### Agent Instructions")
                parts.append(self.agent_instructions)

        # Add separator
        parts.append("\n---\n")

        return "\n".join(parts)

    def get_section(self, name: str) -> str | None:
        """Get content of a specific section by name (case-insensitive)."""
        name_lower = name.lower().strip()

        # Check direct match
        section = self.sections.get(name_lower)
        if section:
            return section.content

        # Check special sections
        special = {
            "overview": self.overview,
            "architecture": self.architecture,
            "conventions": self.conventions,
            "dependencies": self.dependencies,
            "agent instructions": self.agent_instructions,
            "agent_instructions": self.agent_instructions,
        }
        return special.get(name_lower) or self.custom_sections.get(name_lower)


# ============================================================================
# Parsing Functions
# ============================================================================


def parse_markdown_sections(content: str) -> list[ProjectSection]:
    """
    Parse markdown content into hierarchical sections.

    Args:
        content: Markdown content to parse

    Returns:
        List of top-level sections with nested subsections
    """
    # Split by headings
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    sections: list[ProjectSection] = []
    current_content_start = 0
    matches = list(heading_pattern.finditer(content))

    for i, match in enumerate(matches):
        # Get content before this heading (for root content)
        if i == 0 and match.start() > 0:
            root_content = content[:match.start()].strip()
            if root_content:
                sections.append(
                    ProjectSection(name="_root", level=0, content=root_content)
                )

        # Determine content end (start of next heading or end of content)
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

        level = len(match.group(1))
        name = match.group(2).strip()
        section_content = content[match.end():content_end].strip()

        section = ProjectSection(name=name, level=level, content=section_content)
        sections.append(section)

    return sections


def extract_project_context(content: str) -> ProjectContext:
    """
    Extract structured project context from TITAN.md content.

    Args:
        content: Raw TITAN.md content

    Returns:
        Populated ProjectContext
    """
    context = ProjectContext(raw_content=content)
    sections = parse_markdown_sections(content)

    # Map sections by normalized name
    for section in sections:
        name_lower = section.name.lower().strip()
        context.sections[name_lower] = section

        # Extract special sections
        if name_lower == "_root":
            # First line is often project name
            lines = section.content.split("\n")
            if lines:
                context.description = section.content

        elif name_lower in ("overview", "description", "about"):
            context.overview = section.content

        elif name_lower in ("architecture", "design", "structure"):
            context.architecture = section.content

        elif name_lower in ("conventions", "style", "coding conventions", "guidelines"):
            context.conventions = section.content

        elif name_lower in ("dependencies", "requirements", "stack"):
            context.dependencies = section.content

        elif name_lower in (
            "agent instructions",
            "agent_instructions",
            "ai instructions",
            "llm instructions",
            "titan instructions",
        ):
            context.agent_instructions = section.content

        else:
            context.custom_sections[name_lower] = section.content

    # Extract project name from first H1
    for section in sections:
        if section.level == 1:
            context.project_name = section.name
            break

    return context


# ============================================================================
# File Discovery Functions
# ============================================================================


def find_titan_md(
    start_path: str | Path | None = None,
    search_parents: bool = True,
    max_depth: int = 10,
) -> Path | None:
    """
    Find TITAN.md file starting from the given path.

    Args:
        start_path: Starting directory (defaults to cwd)
        search_parents: Whether to search parent directories
        max_depth: Maximum parent directories to search

    Returns:
        Path to TITAN.md if found, None otherwise
    """
    start = Path(start_path) if start_path else Path.cwd()

    if start.is_file():
        start = start.parent

    current = start.resolve()
    depth = 0

    while depth < max_depth:
        # Check for context files in priority order
        for filename in CONTEXT_FILES:
            context_file = current / filename
            if context_file.is_file():
                logger.debug(f"Found context file: {context_file}")
                return context_file

        # Check for .titan directory with config
        titan_dir = current / ".titan"
        if titan_dir.is_dir():
            for filename in ["config.md", "context.md", "TITAN.md"]:
                context_file = titan_dir / filename
                if context_file.is_file():
                    logger.debug(f"Found context file: {context_file}")
                    return context_file

        # Move to parent
        if not search_parents:
            break

        parent = current.parent
        if parent == current:
            # Reached root
            break
        current = parent
        depth += 1

    return None


def load_file_content(path: Path) -> str | None:
    """
    Load content from a file with size checks.

    Args:
        path: Path to file

    Returns:
        File content or None if too large/unreadable
    """
    try:
        size = path.stat().st_size
        if size > MAX_CONTEXT_FILE_SIZE:
            logger.warning(
                f"Context file too large ({size} bytes): {path}"
            )
            return None

        return path.read_text(encoding="utf-8")

    except Exception as e:
        logger.warning(f"Failed to read context file {path}: {e}")
        return None


# ============================================================================
# Main API
# ============================================================================


def load_project_context(
    project_path: str | Path | None = None,
    include_global: bool = True,
) -> ProjectContext:
    """
    Load project context from TITAN.md and optional global config.

    Args:
        project_path: Path to project directory (defaults to cwd)
        include_global: Whether to include global ~/.titan/TITAN.md

    Returns:
        ProjectContext with loaded configuration
    """
    context = ProjectContext()

    # Load global context first
    if include_global and GLOBAL_CONTEXT_PATH.is_file():
        global_content = load_file_content(GLOBAL_CONTEXT_PATH)
        if global_content:
            context.global_context_file = GLOBAL_CONTEXT_PATH
            context.global_content = global_content
            logger.info(f"Loaded global context from {GLOBAL_CONTEXT_PATH}")

    # Find and load project context
    project_file = find_titan_md(project_path)
    if project_file:
        project_content = load_file_content(project_file)
        if project_content:
            # Parse and populate context
            parsed = extract_project_context(project_content)

            context.source_file = project_file
            context.raw_content = project_content
            context.sections = parsed.sections
            context.project_name = parsed.project_name
            context.description = parsed.description
            context.overview = parsed.overview
            context.architecture = parsed.architecture
            context.conventions = parsed.conventions
            context.dependencies = parsed.dependencies
            context.agent_instructions = parsed.agent_instructions
            context.custom_sections = parsed.custom_sections

            logger.info(f"Loaded project context from {project_file}")

    if not context.has_context:
        logger.debug("No project context found")

    return context


def get_context_for_agent(
    project_path: str | Path | None = None,
    agent_type: str | None = None,
) -> str:
    """
    Get formatted context string for agent system prompt injection.

    Args:
        project_path: Path to project directory
        agent_type: Optional agent type for filtering instructions

    Returns:
        Formatted context string for system prompt
    """
    context = load_project_context(project_path)

    if not context.has_context:
        return ""

    prompt_parts = [context.to_system_prompt()]

    # Add agent-specific instructions if available
    if agent_type:
        agent_section = context.get_section(f"{agent_type} instructions")
        if agent_section:
            prompt_parts.append(f"\n### {agent_type.title()} Agent Instructions")
            prompt_parts.append(agent_section)

    return "\n".join(prompt_parts)


# ============================================================================
# CLI Helper (for testing)
# ============================================================================


def print_context_summary(context: ProjectContext) -> None:
    """Print a summary of loaded context (for debugging)."""
    print("=" * 60)
    print("Project Context Summary")
    print("=" * 60)

    if context.global_context_file:
        print(f"Global config: {context.global_context_file}")
    if context.source_file:
        print(f"Project config: {context.source_file}")

    if context.project_name:
        print(f"\nProject: {context.project_name}")

    print(f"\nSections found: {len(context.sections)}")
    for name in context.sections:
        section = context.sections[name]
        content_preview = section.content[:50].replace("\n", " ")
        print(f"  - {name}: {content_preview}...")

    print("\nSpecial sections:")
    print(f"  Overview: {'Yes' if context.overview else 'No'}")
    print(f"  Architecture: {'Yes' if context.architecture else 'No'}")
    print(f"  Conventions: {'Yes' if context.conventions else 'No'}")
    print(f"  Agent Instructions: {'Yes' if context.agent_instructions else 'No'}")

    print("=" * 60)


if __name__ == "__main__":
    # Quick test
    context = load_project_context()
    print_context_summary(context)
    print("\nSystem prompt injection:")
    print(context.to_system_prompt())
