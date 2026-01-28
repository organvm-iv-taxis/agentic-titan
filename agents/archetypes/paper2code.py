"""
Paper2Code Agent - Converts research papers to code implementations.

Capabilities:
- Paper analysis and understanding
- Algorithm extraction
- Code implementation from paper descriptions
- Experiment reproduction

Reference: vendor/agents/deepcode/ Paper2Code patterns
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from agents.framework.base_agent import BaseAgent
from agents.personas import think
from adapters.base import LLMMessage
from adapters.router import get_router

logger = logging.getLogger("titan.agents.paper2code")


# ============================================================================
# Paper2Code Data Structures
# ============================================================================


@dataclass
class PaperSection:
    """A section from a research paper."""

    title: str
    content: str
    section_type: str = "text"  # text, algorithm, equation, figure, table


@dataclass
class Algorithm:
    """An algorithm extracted from a paper."""

    name: str
    description: str
    pseudocode: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    complexity: str = ""


@dataclass
class Paper2CodeResult:
    """Result of paper-to-code conversion."""

    paper_title: str
    algorithms_found: int
    code_files: dict[str, str] = field(default_factory=dict)
    test_files: dict[str, str] = field(default_factory=dict)
    readme: str = ""
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_title": self.paper_title,
            "algorithms_found": self.algorithms_found,
            "code_files": list(self.code_files.keys()),
            "test_files": list(self.test_files.keys()),
            "dependencies": self.dependencies,
        }


# ============================================================================
# Paper2Code Agent
# ============================================================================


class Paper2CodeAgent(BaseAgent):
    """
    Agent specialized in converting research papers to code implementations.

    Workflow:
    1. Analyze paper structure and content
    2. Extract algorithms and key concepts
    3. Identify implementation requirements
    4. Generate code implementations
    5. Create tests and documentation
    """

    def __init__(
        self,
        paper_content: str | None = None,
        paper_url: str | None = None,
        target_language: str = "python",
        include_tests: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("name", "paper2code")
        kwargs.setdefault(
            "capabilities",
            ["paper_analysis", "algorithm_extraction", "code_generation"],
        )
        super().__init__(**kwargs)

        self.paper_content = paper_content
        self.paper_url = paper_url
        self.target_language = target_language
        self.include_tests = include_tests

        self.result: Paper2CodeResult | None = None
        self._router = get_router()

        # Extracted data
        self._paper_title: str = ""
        self._sections: list[PaperSection] = []
        self._algorithms: list[Algorithm] = []

    async def initialize(self) -> None:
        """Initialize the Paper2Code agent."""
        logger.info("Initializing Paper2Code agent")

        await self._router.initialize()

        # Check for similar implementations in Hive Mind
        if self._hive_mind:
            similar = await self.recall(
                f"paper implementation {self.target_language}",
                k=3,
            )
            if similar:
                logger.info(f"Found {len(similar)} similar paper implementations")

    async def work(self) -> Paper2CodeResult:
        """
        Execute paper-to-code conversion workflow.

        Returns:
            Paper2CodeResult with generated code
        """
        if not self.paper_content:
            logger.warning("No paper content provided")
            return Paper2CodeResult(paper_title="Unknown", algorithms_found=0)

        logger.info("Starting paper-to-code conversion")
        self.result = Paper2CodeResult(paper_title="", algorithms_found=0)

        # Step 1: Analyze paper structure
        think("paper2code", "Analyzing paper structure...")
        self.increment_turn()
        await self._analyze_paper_structure()
        logger.info(f"Paper: {self._paper_title}")
        self.result.paper_title = self._paper_title

        # Step 2: Extract algorithms
        think("paper2code", "Extracting algorithms...")
        self.increment_turn()
        await self._extract_algorithms()
        self.result.algorithms_found = len(self._algorithms)
        logger.info(f"Found {len(self._algorithms)} algorithms")

        # Step 3: Identify dependencies
        think("paper2code", "Identifying dependencies...")
        self.increment_turn()
        await self._identify_dependencies()
        logger.info(f"Dependencies: {self.result.dependencies}")

        # Step 4: Generate implementations
        think("paper2code", "Generating code implementations...")
        for i, algo in enumerate(self._algorithms):
            self.increment_turn()
            code = await self._implement_algorithm(algo)
            filename = self._generate_filename(algo.name)
            self.result.code_files[filename] = code
            logger.info(f"Generated: {filename}")

        # Step 5: Generate tests
        if self.include_tests and self._algorithms:
            think("paper2code", "Generating tests...")
            self.increment_turn()
            tests = await self._generate_tests()
            self.result.test_files["test_implementation.py"] = tests

        # Step 6: Generate README
        think("paper2code", "Generating documentation...")
        self.increment_turn()
        self.result.readme = await self._generate_readme()

        # Log decision
        await self.log_decision(
            decision=f"Implemented paper: {self._paper_title}",
            category="implementation",
            rationale=f"Converted {len(self._algorithms)} algorithms to {self.target_language}",
            tags=["paper2code", self.target_language],
        )

        logger.info("Paper-to-code conversion complete")
        return self.result

    async def shutdown(self) -> None:
        """Cleanup Paper2Code agent."""
        logger.info("Paper2Code agent shutting down")

        # Store implementation patterns in Hive Mind
        if self._hive_mind and self.result:
            for algo in self._algorithms:
                await self.remember(
                    content=f"Algorithm: {algo.name}\n{algo.description}\nFrom paper: {self._paper_title}",
                    importance=0.7,
                    tags=["algorithm", "paper2code", self.target_language],
                    metadata={
                        "paper": self._paper_title,
                        "algorithm": algo.name,
                        "complexity": algo.complexity,
                    },
                )

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _analyze_paper_structure(self) -> None:
        """Analyze the paper structure and extract sections."""
        messages = [
            LLMMessage(
                role="user",
                content=f"""Analyze this research paper and extract its structure:

{self.paper_content[:8000]}

Provide:
1. Paper title
2. Main sections and their types (abstract, introduction, method, algorithm, experiments, etc.)
3. Key algorithms or methods described

Format as:
TITLE: [paper title]
SECTIONS:
- [section name]: [type]
ALGORITHMS:
- [algorithm name]: [brief description]""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are an expert at analyzing research papers. Extract structure accurately.",
            max_tokens=1000,
        )

        # Parse response
        content = response.content

        # Extract title
        title_match = re.search(r"TITLE:\s*(.+?)(?:\n|$)", content)
        if title_match:
            self._paper_title = title_match.group(1).strip()

        # Extract sections
        sections_match = re.search(r"SECTIONS:\s*\n((?:[-*]\s*.+\n?)+)", content)
        if sections_match:
            for line in sections_match.group(1).split("\n"):
                if line.strip().startswith(("-", "*")):
                    parts = line.strip("- *").split(":", 1)
                    if len(parts) == 2:
                        self._sections.append(
                            PaperSection(
                                title=parts[0].strip(),
                                content="",
                                section_type=parts[1].strip().lower(),
                            )
                        )

    async def _extract_algorithms(self) -> None:
        """Extract algorithms from the paper."""
        messages = [
            LLMMessage(
                role="user",
                content=f"""Extract all algorithms from this research paper:

{self.paper_content[:10000]}

For each algorithm, provide:
1. Name
2. Description (what it does)
3. Pseudocode (if available, or your best reconstruction)
4. Inputs and outputs
5. Time/space complexity (if mentioned)

Format as:
ALGORITHM: [name]
DESCRIPTION: [what it does]
PSEUDOCODE:
[pseudocode here]
INPUTS: [list of inputs]
OUTPUTS: [list of outputs]
COMPLEXITY: [complexity if known]
---""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are an expert at extracting algorithms from research papers. Be thorough and accurate.",
            max_tokens=2000,
        )

        # Parse algorithms
        algo_blocks = response.content.split("---")
        for block in algo_blocks:
            if "ALGORITHM:" not in block:
                continue

            algo = Algorithm(name="", description="", pseudocode="")

            name_match = re.search(r"ALGORITHM:\s*(.+?)(?:\n|$)", block)
            if name_match:
                algo.name = name_match.group(1).strip()

            desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?:PSEUDOCODE|INPUTS|$)", block, re.DOTALL)
            if desc_match:
                algo.description = desc_match.group(1).strip()

            pseudo_match = re.search(r"PSEUDOCODE:\s*(.+?)(?:INPUTS|OUTPUTS|COMPLEXITY|$)", block, re.DOTALL)
            if pseudo_match:
                algo.pseudocode = pseudo_match.group(1).strip()

            inputs_match = re.search(r"INPUTS:\s*(.+?)(?:OUTPUTS|COMPLEXITY|$)", block, re.DOTALL)
            if inputs_match:
                algo.inputs = [i.strip() for i in inputs_match.group(1).split(",")]

            outputs_match = re.search(r"OUTPUTS:\s*(.+?)(?:COMPLEXITY|$)", block, re.DOTALL)
            if outputs_match:
                algo.outputs = [o.strip() for o in outputs_match.group(1).split(",")]

            complexity_match = re.search(r"COMPLEXITY:\s*(.+?)(?:\n|$)", block)
            if complexity_match:
                algo.complexity = complexity_match.group(1).strip()

            if algo.name:
                self._algorithms.append(algo)

    async def _identify_dependencies(self) -> None:
        """Identify required dependencies for implementation."""
        messages = [
            LLMMessage(
                role="user",
                content=f"""Based on these algorithms, list the {self.target_language} dependencies needed:

Algorithms:
{chr(10).join(f'- {a.name}: {a.description}' for a in self._algorithms)}

List only package names, one per line.
Common packages to consider: numpy, scipy, torch, tensorflow, sklearn, networkx, etc.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system=f"You are an expert {self.target_language} developer.",
            max_tokens=200,
        )

        # Parse dependencies
        deps = []
        for line in response.content.split("\n"):
            line = line.strip("- *").strip()
            if line and not line.startswith("#"):
                # Clean up package name
                pkg = line.split()[0].strip()
                if pkg and pkg[0].isalpha():
                    deps.append(pkg)

        self.result.dependencies = deps[:10]  # Limit

    async def _implement_algorithm(self, algo: Algorithm) -> str:
        """Generate code implementation for an algorithm."""
        messages = [
            LLMMessage(
                role="user",
                content=f"""Implement this algorithm in {self.target_language}:

Algorithm: {algo.name}
Description: {algo.description}

Pseudocode:
{algo.pseudocode}

Inputs: {', '.join(algo.inputs)}
Outputs: {', '.join(algo.outputs)}

Requirements:
- Clean, documented code
- Type hints
- Error handling
- Docstring with algorithm description

Return only the code.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system=f"You are an expert {self.target_language} developer specializing in algorithm implementation.",
            max_tokens=2000,
        )

        # Clean up code
        code = response.content
        if "```" in code:
            # Extract from markdown
            blocks = code.split("```")
            for i, block in enumerate(blocks):
                if i % 2 == 1:
                    lines = block.split("\n")
                    if lines and lines[0].strip() in ["python", "py"]:
                        return "\n".join(lines[1:])
                    return block

        return code

    async def _generate_tests(self) -> str:
        """Generate tests for all implementations."""
        algo_names = [a.name for a in self._algorithms]

        messages = [
            LLMMessage(
                role="user",
                content=f"""Write tests for these algorithm implementations:

Algorithms: {', '.join(algo_names)}

Write pytest tests that:
1. Test basic functionality
2. Test edge cases
3. Test with sample data from the paper (if applicable)

Return only the test code.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are an expert in testing algorithm implementations.",
            max_tokens=1500,
        )

        return response.content

    async def _generate_readme(self) -> str:
        """Generate README documentation."""
        algo_list = "\n".join(f"- **{a.name}**: {a.description[:100]}" for a in self._algorithms)
        dep_list = "\n".join(f"- {d}" for d in self.result.dependencies)

        messages = [
            LLMMessage(
                role="user",
                content=f"""Generate a README.md for this paper implementation:

Paper: {self._paper_title}

Algorithms implemented:
{algo_list}

Dependencies:
{dep_list}

Include:
1. Overview
2. Installation instructions
3. Usage examples
4. Algorithm descriptions
5. Citation information

Keep it concise but informative.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are an expert technical writer.",
            max_tokens=1000,
        )

        return response.content

    def _generate_filename(self, algo_name: str) -> str:
        """Generate a filename from algorithm name."""
        # Convert to snake_case
        name = algo_name.lower()
        name = re.sub(r"[^a-z0-9]+", "_", name)
        name = re.sub(r"_+", "_", name)
        name = name.strip("_")
        return f"{name}.py"
