"""
Built-in Tools - Core tools available to all agents.

Provides:
- File operations (read, write, list)
- Web search (simulated)
- Shell commands (sandboxed)
- Math/calculation
"""

from __future__ import annotations

import asyncio
import logging
import os
import json
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolResult, ToolParameter, register_tool

logger = logging.getLogger("titan.tools.builtin")


# ============================================================================
# File Tools
# ============================================================================

class ReadFileTool(Tool):
    """Read contents of a file."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file from the filesystem."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file to read",
                required=True,
            ),
            ToolParameter(
                name="max_lines",
                type="integer",
                description="Maximum number of lines to read (default: 500)",
                required=False,
                default=500,
            ),
        ]

    async def execute(self, path: str, max_lines: int = 500) -> ToolResult:
        try:
            file_path = Path(path).expanduser().resolve()

            if not file_path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"File not found: {path}",
                )

            if not file_path.is_file():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Not a file: {path}",
                )

            content = file_path.read_text()
            lines = content.split("\n")

            if len(lines) > max_lines:
                content = "\n".join(lines[:max_lines])
                content += f"\n\n... (truncated, {len(lines) - max_lines} more lines)"

            return ToolResult(
                success=True,
                output=content,
                metadata={"path": str(file_path), "lines": len(lines)},
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class WriteFileTool(Tool):
    """Write content to a file."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file. Creates parent directories if needed."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file to write",
                required=True,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="Content to write to the file",
                required=True,
            ),
            ToolParameter(
                name="append",
                type="boolean",
                description="Append to file instead of overwriting",
                required=False,
                default=False,
            ),
        ]

    async def execute(
        self,
        path: str,
        content: str,
        append: bool = False,
    ) -> ToolResult:
        try:
            file_path = Path(path).expanduser().resolve()

            # Create parent directories
            file_path.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            with open(file_path, mode) as f:
                f.write(content)

            return ToolResult(
                success=True,
                output=f"Wrote {len(content)} bytes to {path}",
                metadata={"path": str(file_path), "bytes": len(content)},
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class ListDirectoryTool(Tool):
    """List contents of a directory."""

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return "List files and directories in a given path."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to the directory to list",
                required=True,
            ),
            ToolParameter(
                name="recursive",
                type="boolean",
                description="Whether to list recursively",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="max_depth",
                type="integer",
                description="Maximum depth for recursive listing",
                required=False,
                default=3,
            ),
        ]

    async def execute(
        self,
        path: str,
        recursive: bool = False,
        max_depth: int = 3,
    ) -> ToolResult:
        try:
            dir_path = Path(path).expanduser().resolve()

            if not dir_path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Directory not found: {path}",
                )

            if not dir_path.is_dir():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Not a directory: {path}",
                )

            entries = []

            if recursive:
                for item in dir_path.rglob("*"):
                    rel_path = item.relative_to(dir_path)
                    if len(rel_path.parts) <= max_depth:
                        entry_type = "dir" if item.is_dir() else "file"
                        entries.append({"path": str(rel_path), "type": entry_type})
            else:
                for item in dir_path.iterdir():
                    entry_type = "dir" if item.is_dir() else "file"
                    entries.append({"path": item.name, "type": entry_type})

            # Sort: directories first, then files
            entries.sort(key=lambda x: (x["type"] == "file", x["path"]))

            return ToolResult(
                success=True,
                output=entries,
                metadata={"path": str(dir_path), "count": len(entries)},
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


# ============================================================================
# Web Tools
# ============================================================================

class WebSearchTool(Tool):
    """Search the web for information."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for information on a topic. Returns relevant results."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="Search query",
                required=True,
            ),
            ToolParameter(
                name="num_results",
                type="integer",
                description="Number of results to return (default: 5)",
                required=False,
                default=5,
            ),
        ]

    async def execute(self, query: str, num_results: int = 5) -> ToolResult:
        # TODO: Integrate with actual search API (Serper, Brave, etc.)
        # For now, return a simulated response
        logger.warning("WebSearchTool: Using simulated response")

        simulated_results = [
            {
                "title": f"Result {i+1} for: {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a simulated search result for '{query}'. "
                          f"In production, this would be real web content.",
            }
            for i in range(min(num_results, 5))
        ]

        return ToolResult(
            success=True,
            output=simulated_results,
            metadata={
                "query": query,
                "simulated": True,
                "note": "Integrate with Serper/Brave API for real results",
            },
        )


class WebFetchTool(Tool):
    """Fetch content from a URL."""

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch the content of a web page by URL."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                type="string",
                description="URL to fetch",
                required=True,
            ),
            ToolParameter(
                name="max_length",
                type="integer",
                description="Maximum content length to return",
                required=False,
                default=10000,
            ),
        ]

    async def execute(self, url: str, max_length: int = 10000) -> ToolResult:
        try:
            import httpx

            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()

            content = response.text
            if len(content) > max_length:
                content = content[:max_length] + "\n\n... (truncated)"

            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "url": url,
                    "status_code": response.status_code,
                    "content_type": response.headers.get("content-type", "unknown"),
                },
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


# ============================================================================
# Shell Tools
# ============================================================================

class ShellCommandTool(Tool):
    """Execute a shell command."""

    # Commands that are allowed by default
    ALLOWED_COMMANDS = {
        "ls", "cat", "head", "tail", "wc", "grep", "find", "echo",
        "pwd", "date", "whoami", "uname", "env", "which",
        "python", "pip", "npm", "node", "git", "curl",
    }

    # Commands that are never allowed
    BLOCKED_COMMANDS = {
        "rm", "rmdir", "mv", "cp", "chmod", "chown", "sudo", "su",
        "kill", "pkill", "shutdown", "reboot", "dd", "mkfs",
    }

    @property
    def name(self) -> str:
        return "shell_command"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command. Some commands are blocked for safety. "
            f"Allowed: {', '.join(sorted(self.ALLOWED_COMMANDS)[:10])}..."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description="Shell command to execute",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Command timeout in seconds (default: 30)",
                required=False,
                default=30,
            ),
        ]

    async def execute(self, command: str, timeout: int = 30) -> ToolResult:
        # Extract the base command
        base_command = command.split()[0] if command.split() else ""

        # Check if command is blocked
        if base_command in self.BLOCKED_COMMANDS:
            return ToolResult(
                success=False,
                output=None,
                error=f"Command '{base_command}' is blocked for safety",
            )

        # Check if command is allowed
        if base_command not in self.ALLOWED_COMMANDS:
            return ToolResult(
                success=False,
                output=None,
                error=f"Command '{base_command}' is not in allowed list. "
                      f"Allowed: {', '.join(sorted(self.ALLOWED_COMMANDS)[:10])}...",
            )

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            output = stdout.decode() if stdout else ""
            error_output = stderr.decode() if stderr else ""

            if process.returncode == 0:
                return ToolResult(
                    success=True,
                    output=output or "(no output)",
                    metadata={
                        "command": command,
                        "return_code": process.returncode,
                        "stderr": error_output if error_output else None,
                    },
                )
            else:
                return ToolResult(
                    success=False,
                    output=output,
                    error=error_output or f"Command failed with code {process.returncode}",
                )

        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Command timed out after {timeout} seconds",
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


# ============================================================================
# Utility Tools
# ============================================================================

class CalculatorTool(Tool):
    """Perform mathematical calculations using ast.literal_eval for safety."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Perform mathematical calculations. Supports basic arithmetic."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')",
                required=True,
            ),
        ]

    async def execute(self, expression: str) -> ToolResult:
        import ast
        import operator
        import math

        # Supported operators
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        # Supported functions
        funcs = {
            "abs": abs,
            "round": round,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
        }

        # Supported constants
        consts = {
            "pi": math.pi,
            "e": math.e,
        }

        def safe_eval(node: ast.AST) -> float:
            """Safely evaluate an AST node."""
            if isinstance(node, ast.Constant):  # Numbers
                return node.value
            elif isinstance(node, ast.Name):  # Named constants
                if node.id in consts:
                    return consts[node.id]
                raise ValueError(f"Unknown constant: {node.id}")
            elif isinstance(node, ast.BinOp):  # Binary operations
                op = ops.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
                return op(safe_eval(node.left), safe_eval(node.right))
            elif isinstance(node, ast.UnaryOp):  # Unary operations
                op = ops.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
                return op(safe_eval(node.operand))
            elif isinstance(node, ast.Call):  # Function calls
                if isinstance(node.func, ast.Name) and node.func.id in funcs:
                    args = [safe_eval(arg) for arg in node.args]
                    return funcs[node.func.id](*args)
                raise ValueError(f"Unknown function: {getattr(node.func, 'id', 'unknown')}")
            else:
                raise ValueError(f"Unsupported expression type: {type(node).__name__}")

        try:
            tree = ast.parse(expression, mode="eval")
            result = safe_eval(tree.body)
            return ToolResult(
                success=True,
                output=result,
                metadata={"expression": expression},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Calculation error: {e}",
            )


class JsonTool(Tool):
    """Parse or format JSON data."""

    @property
    def name(self) -> str:
        return "json_tool"

    @property
    def description(self) -> str:
        return "Parse JSON string to object, or format object as pretty JSON."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="Action: 'parse' or 'format'",
                required=True,
                enum=["parse", "format"],
            ),
            ToolParameter(
                name="data",
                type="string",
                description="JSON string to parse, or object to format",
                required=True,
            ),
        ]

    async def execute(self, action: str, data: str) -> ToolResult:
        try:
            if action == "parse":
                result = json.loads(data)
                return ToolResult(success=True, output=result)
            elif action == "format":
                obj = json.loads(data) if isinstance(data, str) else data
                formatted = json.dumps(obj, indent=2, default=str)
                return ToolResult(success=True, output=formatted)
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown action: {action}",
                )
        except json.JSONDecodeError as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"JSON error: {e}",
            )


# ============================================================================
# Registration
# ============================================================================

def register_builtin_tools() -> None:
    """Register all built-in tools."""
    tools = [
        ReadFileTool(),
        WriteFileTool(),
        ListDirectoryTool(),
        WebSearchTool(),
        WebFetchTool(),
        ShellCommandTool(),
        CalculatorTool(),
        JsonTool(),
    ]

    for tool in tools:
        register_tool(tool)

    logger.info(f"Registered {len(tools)} built-in tools")


# Auto-register on import
register_builtin_tools()
