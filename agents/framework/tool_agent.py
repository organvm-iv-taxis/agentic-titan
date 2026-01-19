"""
Tool-Using Agent - Base class for agents that can use tools.

Provides:
- Tool execution loop (ReAct pattern)
- Tool call parsing and execution
- Conversation management with tool results
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, TYPE_CHECKING

from agents.framework.base_agent import BaseAgent
from adapters.base import LLMMessage, Tool as LLMTool
from adapters.router import get_router, LLMRouter
from tools.base import get_registry, ToolRegistry
from tools.executor import get_executor, ToolExecutor

if TYPE_CHECKING:
    pass

logger = logging.getLogger("titan.agents.tool_agent")


class ToolUsingAgent(BaseAgent):
    """
    Base class for agents that can use tools.

    Implements the ReAct pattern:
    1. Reason about what to do
    2. Act by calling tools
    3. Observe tool results
    4. Repeat until task is complete

    Subclasses must implement:
    - get_system_prompt() - Define agent's personality and instructions
    - is_task_complete() - Determine when to stop

    Optional overrides:
    - get_available_tools() - Filter which tools agent can use
    - pre_tool_call() - Hook before tool execution
    - post_tool_call() - Hook after tool execution
    """

    def __init__(
        self,
        task: str | None = None,
        *,
        max_tool_calls: int = 10,
        router: LLMRouter | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_executor: ToolExecutor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.task = task
        self.max_tool_calls = max_tool_calls

        self._router = router or get_router()
        self._tool_registry = tool_registry or get_registry()
        self._tool_executor = tool_executor or get_executor()

        self._messages: list[LLMMessage] = []
        self._tool_calls_made: int = 0

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.

        Should include:
        - Agent's role and personality
        - Instructions for how to complete tasks
        - Guidelines for tool usage
        """
        pass

    def is_task_complete(self, response: str) -> bool:
        """
        Determine if the task is complete.

        Override this for custom completion detection.
        Default: Task is complete when response contains "TASK_COMPLETE"
        """
        return "TASK_COMPLETE" in response.upper()

    def get_available_tools(self) -> list[str]:
        """
        Get list of tool names this agent can use.

        Override to restrict tool access.
        Default: All registered tools
        """
        return self._tool_registry.list_names()

    async def pre_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """
        Hook called before executing a tool.

        Returns True to proceed, False to skip the tool call.
        Override for custom validation or logging.
        """
        logger.debug(f"Pre-tool-call: {tool_name}")
        return True

    async def post_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> None:
        """
        Hook called after executing a tool.

        Override for custom handling of tool results.
        """
        logger.debug(f"Post-tool-call: {tool_name} -> {str(result)[:100]}")

    async def initialize(self) -> None:
        """Initialize the agent."""
        await self._router.initialize()

        # Import builtin tools if not already registered
        if not self._tool_registry.list():
            from tools.builtin import register_builtin_tools
            register_builtin_tools()

        logger.info(
            f"Agent '{self.name}' initialized with "
            f"{len(self.get_available_tools())} tools"
        )

    async def work(self) -> dict[str, Any]:
        """
        Execute the agent's task using tools.

        Returns:
            Dict with conversation history and final result
        """
        if not self.task:
            return {"error": "No task specified", "messages": []}

        # Start conversation
        self._messages = [
            LLMMessage(role="user", content=self.task)
        ]

        # Get available tools
        available_tool_names = self.get_available_tools()
        tools = [
            self._tool_registry.get(name)
            for name in available_tool_names
            if self._tool_registry.get(name)
        ]
        llm_tools = [
            LLMTool(
                name=t.name,
                description=t.description,
                parameters=t.to_schema()["input_schema"],
            )
            for t in tools
        ]

        logger.info(f"Starting task with {len(llm_tools)} tools available")

        final_response = ""
        while self._tool_calls_made < self.max_tool_calls:
            self.increment_turn()

            # Get LLM response
            response = await self._router.complete(
                self._messages,
                system=self.get_system_prompt(),
                tools=llm_tools if llm_tools else None,
            )

            # Check for tool calls
            if response.tool_calls:
                # Process tool calls
                await self._process_tool_calls(response)
            else:
                # No tool calls - check if done
                final_response = response.content
                self._messages.append(LLMMessage(
                    role="assistant",
                    content=final_response,
                ))

                if self.is_task_complete(final_response):
                    logger.info("Task marked as complete")
                    break

                # If not complete and no tools, we're stuck
                if not response.tool_calls:
                    logger.warning("No tool calls and task not complete")
                    break

        return {
            "final_response": final_response,
            "tool_calls_made": self._tool_calls_made,
            "turns": self._context.turn_number if self._context else 0,
            "messages": [
                {"role": m.role, "content": m.content[:500]}
                for m in self._messages
            ],
        }

    async def _process_tool_calls(self, response: Any) -> None:
        """Process tool calls from LLM response."""
        # Add assistant message with tool calls
        self._messages.append(LLMMessage(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls,
        ))

        # Execute each tool
        for tc in response.tool_calls:
            tool_name = tc["name"]
            arguments = tc.get("arguments", {})
            tool_id = tc.get("id", f"call_{self._tool_calls_made}")

            # Pre-hook
            if not await self.pre_tool_call(tool_name, arguments):
                # Tool call skipped
                self._messages.append(LLMMessage(
                    role="user",
                    content=f"Tool call to {tool_name} was skipped.",
                    tool_call_id=tool_id,
                ))
                continue

            # Execute tool
            executions = await self._tool_executor.execute_all([tc])
            self._tool_calls_made += 1

            if executions:
                result = executions[0].result
                result_text = result.to_message()

                # Post-hook
                await self.post_tool_call(tool_name, arguments, result)

                # Add tool result to conversation
                self._messages.append(LLMMessage(
                    role="user",  # Tool results are "user" role in many APIs
                    content=f"[Tool Result for {tool_name}]\n{result_text}",
                    tool_call_id=tool_id,
                ))

                logger.info(
                    f"Tool {tool_name} executed "
                    f"(success={result.success}, calls={self._tool_calls_made})"
                )

    async def shutdown(self) -> None:
        """Cleanup the agent."""
        logger.info(
            f"Agent '{self.name}' shutdown. "
            f"Made {self._tool_calls_made} tool calls"
        )


class SimpleToolAgent(ToolUsingAgent):
    """
    Simple tool-using agent with configurable system prompt.

    Use for quick tasks that need tool access.
    """

    def __init__(
        self,
        task: str,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("name", "simple-tool-agent")
        super().__init__(task=task, **kwargs)
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._allowed_tools = tools

    def _default_system_prompt(self) -> str:
        return """You are a helpful assistant with access to tools.

When given a task:
1. Think about what tools you need
2. Use the tools to complete the task
3. Provide a clear final answer

When you have completed the task, include "TASK_COMPLETE" in your response.

Be efficient - only use tools when necessary."""

    def get_system_prompt(self) -> str:
        return self._system_prompt

    def get_available_tools(self) -> list[str]:
        if self._allowed_tools:
            return self._allowed_tools
        return super().get_available_tools()
