"""
Titan Core - Core functionality for agent execution.

This module provides:
- Project context loading (TITAN.md)
- Configuration management
- Core utilities
"""

from titan.core.project_context import (
    ProjectContext,
    find_titan_md,
    load_project_context,
)

__all__ = [
    "ProjectContext",
    "load_project_context",
    "find_titan_md",
]
