"""
Researcher Agent - Gathers and analyzes information.

Capabilities:
- Web search
- Document analysis
- Summarization
- Cross-referencing sources
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agents.framework.base_agent import BaseAgent, AgentResult, AgentState
from agents.personas import RESEARCHER, say, think
from adapters.base import LLMMessage
from adapters.router import get_router

logger = logging.getLogger("titan.agents.researcher")


@dataclass
class ResearchTask:
    """A research task."""

    topic: str
    questions: list[str]
    sources: list[str]
    findings: list[dict[str, Any]]
    summary: str = ""


class ResearcherAgent(BaseAgent):
    """
    Agent specialized in research and information gathering.

    Uses a systematic approach:
    1. Break down topic into questions
    2. Search for sources
    3. Analyze and cross-reference
    4. Synthesize findings
    """

    def __init__(
        self,
        topic: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Set defaults that can be overridden by kwargs
        kwargs.setdefault("name", "researcher")
        kwargs.setdefault("capabilities", ["web_search", "document_analysis", "summarization"])
        super().__init__(**kwargs)
        self.topic = topic
        self.task: ResearchTask | None = None
        self._router = get_router()

    async def initialize(self) -> None:
        """Initialize the research agent."""
        say(RESEARCHER, "Initializing research agent")

        # Initialize LLM router
        await self._router.initialize()

        # Load any previous research from Hive Mind
        if self._hive_mind and self.topic:
            previous = await self.recall(f"research on {self.topic}", k=5)
            if previous:
                say(RESEARCHER, f"Found {len(previous)} related previous research items")

    async def work(self) -> ResearchTask:
        """
        Execute research workflow.

        Returns:
            ResearchTask with findings
        """
        if not self.topic:
            say(RESEARCHER, "No topic specified, waiting for input")
            return ResearchTask(topic="", questions=[], sources=[], findings=[])

        say(RESEARCHER, f"Beginning research on: {self.topic}")
        self.task = ResearchTask(
            topic=self.topic,
            questions=[],
            sources=[],
            findings=[],
        )

        # Step 1: Generate research questions
        think(RESEARCHER, "Breaking down topic into key questions...")
        questions = await self._generate_questions()
        self.task.questions = questions
        say(RESEARCHER, f"Generated {len(questions)} research questions")

        # Step 2: For each question, gather information
        for i, question in enumerate(questions, 1):
            self.increment_turn()
            say(RESEARCHER, f"[{i}/{len(questions)}] Researching: {question[:50]}...")

            # Simulate research (in real implementation, this would use tools)
            finding = await self._research_question(question)
            self.task.findings.append(finding)

            # Log decision
            await self.log_decision(
                decision=f"Researched question: {question[:50]}",
                category="research",
                rationale=f"Part of systematic research on {self.topic}",
                tags=["research", self.topic.lower().replace(" ", "-")],
            )

        # Step 3: Synthesize findings
        think(RESEARCHER, "Synthesizing findings...")
        summary = await self._synthesize()
        self.task.summary = summary

        say(RESEARCHER, f"Research complete. Summary: {len(summary)} chars")
        return self.task

    async def shutdown(self) -> None:
        """Cleanup research agent."""
        say(RESEARCHER, "Research agent shutting down")

        # Store final research in Hive Mind
        if self._hive_mind and self.task and self.task.summary:
            await self.remember(
                content=f"Research on {self.topic}:\n\n{self.task.summary}",
                importance=0.8,
                tags=["research", "summary"],
                metadata={
                    "topic": self.topic,
                    "questions": len(self.task.questions),
                    "findings": len(self.task.findings),
                },
            )

    async def _generate_questions(self) -> list[str]:
        """Generate research questions for the topic."""
        messages = [
            LLMMessage(
                role="user",
                content=f"""Generate 3-5 key research questions to thoroughly investigate this topic:

Topic: {self.topic}

Provide questions that:
1. Cover different aspects of the topic
2. Are specific and answerable
3. Build understanding progressively

Format: One question per line, no numbering.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a research assistant. Generate focused, investigative questions.",
            max_tokens=500,
        )

        questions = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        return questions[:5]  # Limit to 5 questions

    async def _research_question(self, question: str) -> dict[str, Any]:
        """Research a specific question."""
        # In real implementation, this would use web search tools
        messages = [
            LLMMessage(
                role="user",
                content=f"""Provide a brief, factual answer to this research question:

Question: {question}

Provide:
1. A direct answer (2-3 sentences)
2. Key facts supporting the answer
3. Any caveats or uncertainties

Be concise and factual.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a research assistant. Provide factual, well-sourced answers.",
            max_tokens=300,
        )

        return {
            "question": question,
            "answer": response.content,
            "sources": [],  # Would be populated by web search
            "confidence": 0.7,  # Would be computed from source quality
        }

    async def _synthesize(self) -> str:
        """Synthesize all findings into a summary."""
        if not self.task or not self.task.findings:
            return "No findings to synthesize."

        findings_text = "\n\n".join(
            f"Q: {f['question']}\nA: {f['answer']}" for f in self.task.findings
        )

        messages = [
            LLMMessage(
                role="user",
                content=f"""Synthesize these research findings into a coherent summary:

Topic: {self.task.topic}

Findings:
{findings_text}

Create a summary that:
1. Provides an overview of the topic
2. Highlights key insights
3. Notes any contradictions or gaps
4. Suggests areas for further research

Keep it under 300 words.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a research synthesizer. Create coherent, insightful summaries.",
            max_tokens=500,
        )

        return response.content
