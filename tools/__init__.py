"""
Titan Tools - Executable tools for agents.

Provides:
- Tool protocol and registry
- Built-in tools (file, web, shell)
- MCP bridge for external tools
"""

from tools.base import Tool, ToolResult, ToolRegistry, get_registry
from tools.executor import ToolExecutor, get_executor

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "get_registry",
    "ToolExecutor",
    "get_executor",
]
