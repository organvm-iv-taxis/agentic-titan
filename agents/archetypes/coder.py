"""
Coder Agent - Writes and tests code.

Capabilities:
- Code generation
- Code review
- Test writing
- Refactoring
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agents.framework.base_agent import BaseAgent
from agents.personas import CODER, say, think
from adapters.base import LLMMessage
from adapters.router import get_router

logger = logging.getLogger("titan.agents.coder")


@dataclass
class CodeTask:
    """A coding task."""

    description: str
    language: str
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    tests_written: int = 0
    code_output: str = ""


class CoderAgent(BaseAgent):
    """
    Agent specialized in software development.

    Follows a systematic approach:
    1. Understand requirements
    2. Plan implementation
    3. Write code
    4. Write tests
    5. Review own work
    """

    def __init__(
        self,
        task_description: str | None = None,
        language: str = "python",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="coder",
            capabilities=["code_generation", "code_review", "execution"],
            **kwargs,
        )
        self.task_description = task_description
        self.language = language
        self.task: CodeTask | None = None
        self._router = get_router()

    async def initialize(self) -> None:
        """Initialize the coder agent."""
        say(CODER, f"Initializing coder agent (language: {self.language})")

        await self._router.initialize()

        # Check for relevant patterns in Hive Mind
        if self._hive_mind and self.task_description:
            patterns = await self.recall(
                f"code patterns for {self.task_description}",
                k=3,
            )
            if patterns:
                say(CODER, f"Found {len(patterns)} relevant code patterns")

    async def work(self) -> CodeTask:
        """
        Execute coding workflow.

        Returns:
            CodeTask with results
        """
        if not self.task_description:
            say(CODER, "No task specified, waiting for input")
            return CodeTask(description="", language=self.language)

        say(CODER, f"Starting task: {self.task_description[:50]}...")
        self.task = CodeTask(
            description=self.task_description,
            language=self.language,
        )

        # Step 1: Plan implementation
        think(CODER, "Planning implementation approach...")
        self.increment_turn()
        plan = await self._plan_implementation()
        say(CODER, "Implementation plan ready")

        # Step 2: Generate code
        think(CODER, "Writing code...")
        self.increment_turn()
        code = await self._generate_code(plan)
        self.task.code_output = code
        say(CODER, f"Generated {len(code.split(chr(10)))} lines of code")

        # Step 3: Generate tests
        think(CODER, "Writing tests...")
        self.increment_turn()
        tests = await self._generate_tests(code)
        self.task.tests_written = len(tests.split("def test_")) - 1
        say(CODER, f"Wrote {self.task.tests_written} tests")

        # Step 4: Self-review
        think(CODER, "Reviewing my own code...")
        self.increment_turn()
        issues = await self._self_review(code)

        if issues:
            say(CODER, f"Found {len(issues)} issues, refactoring...")
            code = await self._refactor(code, issues)
            self.task.code_output = code

        # Log decision
        await self.log_decision(
            decision=f"Implemented: {self.task_description[:50]}",
            category="implementation",
            rationale=plan[:200] if plan else "Direct implementation",
            tags=["code", self.language],
        )

        say(CODER, "Coding task complete")
        return self.task

    async def shutdown(self) -> None:
        """Cleanup coder agent."""
        say(CODER, "Coder agent shutting down")

        # Store code patterns in Hive Mind
        if self._hive_mind and self.task and self.task.code_output:
            await self.remember(
                content=f"Code for {self.task.description}:\n\n```{self.language}\n{self.task.code_output[:500]}...\n```",
                importance=0.6,
                tags=["code", self.language, "pattern"],
                metadata={
                    "language": self.language,
                    "lines": len(self.task.code_output.split("\n")),
                },
            )

    async def _plan_implementation(self) -> str:
        """Plan the implementation approach."""
        messages = [
            LLMMessage(
                role="user",
                content=f"""Plan the implementation for this task:

Task: {self.task_description}
Language: {self.language}

Provide:
1. High-level approach
2. Key components/functions needed
3. Data structures to use
4. Edge cases to handle

Keep it concise (under 200 words).""",
            )
        ]

        response = await self._router.complete(
            messages,
            system=f"You are an expert {self.language} developer. Plan implementations thoroughly.",
            max_tokens=400,
        )

        return response.content

    async def _generate_code(self, plan: str) -> str:
        """Generate code based on plan."""
        messages = [
            LLMMessage(
                role="user",
                content=f"""Write {self.language} code for this task:

Task: {self.task_description}

Plan:
{plan}

Requirements:
- Clean, readable code
- Proper error handling
- Type hints (if applicable)
- Docstrings for functions

Return only the code, no explanations.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system=f"You are an expert {self.language} developer. Write production-quality code.",
            max_tokens=2000,
        )

        # Extract code from markdown if present
        content = response.content
        if "```" in content:
            code_blocks = content.split("```")
            for i, block in enumerate(code_blocks):
                if i % 2 == 1:  # Odd indices are code blocks
                    # Remove language identifier
                    lines = block.split("\n")
                    if lines[0].strip() in ["python", "javascript", "typescript", "go", "rust"]:
                        return "\n".join(lines[1:])
                    return block

        return content

    async def _generate_tests(self, code: str) -> str:
        """Generate tests for the code."""
        messages = [
            LLMMessage(
                role="user",
                content=f"""Write tests for this {self.language} code:

```{self.language}
{code}
```

Write:
1. Unit tests for each function
2. Edge case tests
3. Integration tests if applicable

Use pytest (Python) or appropriate test framework.
Return only the test code.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system=f"You are an expert in testing {self.language} code. Write thorough tests.",
            max_tokens=1000,
        )

        return response.content

    async def _self_review(self, code: str) -> list[str]:
        """Review own code for issues."""
        messages = [
            LLMMessage(
                role="user",
                content=f"""Review this {self.language} code for issues:

```{self.language}
{code}
```

Look for:
1. Bugs or logic errors
2. Security issues
3. Performance problems
4. Code style issues

List only actual issues, one per line. If no issues, say "No issues found".""",
            )
        ]

        response = await self._router.complete(
            messages,
            system=f"You are a code reviewer. Be thorough but fair.",
            max_tokens=500,
        )

        if "no issues" in response.content.lower():
            return []

        return [line.strip() for line in response.content.split("\n") if line.strip()]

    async def _refactor(self, code: str, issues: list[str]) -> str:
        """Refactor code to fix issues."""
        messages = [
            LLMMessage(
                role="user",
                content=f"""Fix these issues in the {self.language} code:

Issues:
{chr(10).join(f'- {issue}' for issue in issues)}

Code:
```{self.language}
{code}
```

Return only the fixed code.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system=f"You are an expert {self.language} developer. Fix code issues.",
            max_tokens=2000,
        )

        return response.content
