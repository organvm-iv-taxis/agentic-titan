"""
Tool Executor - Executes tools based on LLM tool calls.

Handles:
- Parsing tool calls from LLM responses
- Executing tools with proper arguments
- Returning results for conversation continuation
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from tools.base import Tool, ToolResult, ToolRegistry, get_registry

logger = logging.getLogger("titan.tools.executor")


@dataclass
class ToolCall:
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolExecution:
    """Result of a tool execution."""

    call: ToolCall
    result: ToolResult
    execution_time_ms: int = 0


class ToolExecutor:
    """
    Executes tools based on LLM tool calls.

    Features:
    - Parse tool calls from various formats
    - Execute tools with argument validation
    - Handle parallel tool execution
    - Timeout and error handling
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        max_concurrent: int = 5,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.registry = registry or get_registry()
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def parse_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[ToolCall]:
        """
        Parse tool calls from LLM response.

        Handles both Anthropic and OpenAI formats.
        """
        parsed = []

        for tc in tool_calls:
            # Anthropic format: {id, name, arguments}
            # OpenAI format: {id, function: {name, arguments}}
            if "function" in tc:
                # OpenAI format
                name = tc["function"]["name"]
                args_raw = tc["function"]["arguments"]
                if isinstance(args_raw, str):
                    args = json.loads(args_raw)
                else:
                    args = args_raw
            else:
                # Anthropic format
                name = tc["name"]
                args = tc.get("arguments") or tc.get("input", {})

            parsed.append(ToolCall(
                id=tc.get("id", f"call_{len(parsed)}"),
                name=name,
                arguments=args if isinstance(args, dict) else {},
            ))

        return parsed

    async def execute_one(self, call: ToolCall) -> ToolExecution:
        """Execute a single tool call."""
        import time
        start = time.perf_counter()

        tool = self.registry.get(call.name)
        if not tool:
            result = ToolResult(
                success=False,
                output=None,
                error=f"Tool not found: {call.name}",
            )
        else:
            async with self._semaphore:
                try:
                    result = await asyncio.wait_for(
                        tool.execute(**call.arguments),
                        timeout=self.timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    result = ToolResult(
                        success=False,
                        output=None,
                        error=f"Tool timed out after {self.timeout_seconds}s",
                    )
                except Exception as e:
                    logger.error(f"Tool {call.name} failed: {e}", exc_info=True)
                    result = ToolResult(
                        success=False,
                        output=None,
                        error=str(e),
                    )

        execution_time_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"Tool {call.name} executed in {execution_time_ms}ms "
            f"(success={result.success})"
        )

        return ToolExecution(
            call=call,
            result=result,
            execution_time_ms=execution_time_ms,
        )

    async def execute_all(
        self,
        tool_calls: list[dict[str, Any]],
        parallel: bool = True,
    ) -> list[ToolExecution]:
        """
        Execute all tool calls.

        Args:
            tool_calls: List of tool calls from LLM
            parallel: Whether to execute in parallel

        Returns:
            List of execution results
        """
        calls = self.parse_tool_calls(tool_calls)

        if not calls:
            return []

        if parallel and len(calls) > 1:
            # Execute in parallel
            tasks = [self.execute_one(call) for call in calls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            executions = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    executions.append(ToolExecution(
                        call=calls[i],
                        result=ToolResult(success=False, output=None, error=str(result)),
                    ))
                else:
                    executions.append(result)
            return executions
        else:
            # Execute sequentially
            return [await self.execute_one(call) for call in calls]

    def format_results_for_llm(
        self,
        executions: list[ToolExecution],
        format: str = "anthropic",
    ) -> list[dict[str, Any]]:
        """
        Format execution results for LLM continuation.

        Args:
            executions: List of tool executions
            format: "anthropic" or "openai"

        Returns:
            List of tool result messages
        """
        if format == "openai":
            return [
                {
                    "role": "tool",
                    "tool_call_id": ex.call.id,
                    "content": ex.result.to_message(),
                }
                for ex in executions
            ]
        else:
            # Anthropic format
            return [
                {
                    "type": "tool_result",
                    "tool_use_id": ex.call.id,
                    "content": ex.result.to_message(),
                }
                for ex in executions
            ]


# Singleton executor
_default_executor: ToolExecutor | None = None


def get_executor() -> ToolExecutor:
    """Get the default tool executor."""
    global _default_executor
    if _default_executor is None:
        _default_executor = ToolExecutor()
    return _default_executor


async def execute_tools(tool_calls: list[dict[str, Any]]) -> list[ToolExecution]:
    """Execute tool calls using the default executor."""
    return await get_executor().execute_all(tool_calls)
