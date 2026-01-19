"""
Learning Agent - Agent with episodic learning capabilities.

Extends base agents with:
- Automatic episode recording
- Past experience retrieval
- Learning-informed decisions
"""

from __future__ import annotations

import logging
from typing import Any

from agents.framework.base_agent import BaseAgent, AgentResult
from agents.learning.episodic import (
    Episode,
    EpisodeOutcome,
    EpisodicMemory,
)

logger = logging.getLogger("titan.learning.learner")


class LearningAgent(BaseAgent):
    """
    Agent with episodic learning capabilities.

    Automatically:
    - Records episodes of all interactions
    - Retrieves relevant past experiences
    - Adjusts behavior based on lessons learned
    """

    def __init__(
        self,
        *,
        memory: EpisodicMemory | None = None,
        task_type: str = "general",
        learn_from_failures: bool = True,
        use_past_experience: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._memory = memory
        self._task_type = task_type
        self._learn_from_failures = learn_from_failures
        self._use_past_experience = use_past_experience

        # Current episode being recorded
        self._current_episode: Episode | None = None

        # Retrieved experience for current task
        self._retrieved_experience: list[Episode] = []

    async def initialize(self) -> None:
        """Initialize with episodic memory."""
        await super().initialize()

        # Initialize memory if not provided
        if self._memory is None:
            self._memory = EpisodicMemory()
            await self._memory.initialize()

        logger.info(f"Learning agent '{self.name}' initialized")

    async def run(self) -> AgentResult:
        """Run with episode recording."""
        # Start episode
        self._start_episode()

        try:
            # Retrieve relevant past experience
            if self._use_past_experience:
                await self._retrieve_experience()

            # Run the actual work
            result = await super().run()

            # Complete episode
            outcome = EpisodeOutcome.SUCCESS if result.success else EpisodeOutcome.FAILURE
            self._complete_episode(outcome, result.result, result.error)

            return result

        except TimeoutError:
            self._complete_episode(EpisodeOutcome.TIMEOUT, None, "Timeout")
            raise

        except Exception as e:
            self._complete_episode(EpisodeOutcome.FAILURE, None, str(e))
            raise

    def _start_episode(self) -> None:
        """Start recording a new episode."""
        self._current_episode = Episode(
            task=getattr(self, "task", "") or getattr(self, "topic", "") or "",
            task_type=self._task_type,
            agent_type=self.__class__.__name__,
            context=self._get_context_for_episode(),
        )
        logger.debug(f"Started episode {self._current_episode.id}")

    def _complete_episode(
        self,
        outcome: EpisodeOutcome,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Complete and store the current episode."""
        if not self._current_episode:
            return

        self._current_episode.complete(outcome, result, error)

        # Store episode asynchronously
        if self._memory:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule for later
                    loop.create_task(self._memory.store(self._current_episode))
                else:
                    loop.run_until_complete(self._memory.store(self._current_episode))
            except Exception as e:
                logger.warning(f"Could not store episode: {e}")

        logger.info(
            f"Completed episode {self._current_episode.id}: "
            f"{outcome.value} in {self._current_episode.duration_seconds:.1f}s"
        )

    async def _retrieve_experience(self) -> None:
        """Retrieve relevant past experiences."""
        if not self._memory:
            return

        task = getattr(self, "task", "") or getattr(self, "topic", "") or ""
        if not task:
            return

        # Get similar past episodes
        self._retrieved_experience = await self._memory.recall(
            query=task,
            task_type=self._task_type,
            limit=3,
        )

        if self._retrieved_experience:
            logger.info(
                f"Retrieved {len(self._retrieved_experience)} "
                f"relevant past episodes"
            )

    def _get_context_for_episode(self) -> dict[str, Any]:
        """Get context to record in episode."""
        return {
            "agent_name": self.name,
            "agent_id": self.id,
            "turn": self._context.turn_number if self._context else 0,
        }

    def record_action(self, action_type: str, **details: Any) -> None:
        """Record an action in the current episode."""
        if self._current_episode:
            self._current_episode.add_action(action_type, details)

    def record_tool_call(self, tool_name: str) -> None:
        """Record a tool call in the current episode."""
        if self._current_episode:
            self._current_episode.add_tool_call(tool_name)

    def record_llm_call(self) -> None:
        """Record an LLM call in the current episode."""
        if self._current_episode:
            self._current_episode.llm_calls += 1

    def get_past_lessons(self) -> list[str]:
        """Get lessons from retrieved past experience."""
        lessons: list[str] = []

        for episode in self._retrieved_experience:
            if episode.signal:
                lessons.extend(episode.signal.lessons)

        return list(set(lessons))

    def get_successful_tools(self) -> list[str]:
        """Get tools that were successful in similar tasks."""
        tools: list[str] = []

        for episode in self._retrieved_experience:
            if episode.outcome == EpisodeOutcome.SUCCESS:
                tools.extend(episode.tool_calls)

        # Return most frequent tools
        tool_counts: dict[str, int] = {}
        for tool in tools:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

        sorted_tools = sorted(
            tool_counts.keys(),
            key=lambda t: tool_counts[t],
            reverse=True,
        )

        return sorted_tools[:5]

    def build_learning_context(self) -> str:
        """
        Build context from past experience for system prompt.

        Returns a string that can be appended to system prompt
        to inform the agent of relevant past experience.
        """
        if not self._retrieved_experience:
            return ""

        parts: list[str] = []
        parts.append("\n\n## Past Experience")

        # Summarize successful approaches
        successes = [
            ep for ep in self._retrieved_experience
            if ep.outcome == EpisodeOutcome.SUCCESS
        ]
        if successes:
            parts.append("\nSuccessful approaches for similar tasks:")
            for ep in successes[:2]:
                tools = ", ".join(set(ep.tool_calls)) if ep.tool_calls else "none"
                parts.append(f"- Task: {ep.task[:100]}...")
                parts.append(f"  Tools: {tools}")
                if ep.signal and ep.signal.positive_factors:
                    parts.append(f"  Factors: {', '.join(ep.signal.positive_factors)}")

        # Summarize failures to avoid
        failures = [
            ep for ep in self._retrieved_experience
            if ep.outcome in [EpisodeOutcome.FAILURE, EpisodeOutcome.TIMEOUT]
        ]
        if failures and self._learn_from_failures:
            parts.append("\nApproaches to avoid:")
            for ep in failures[:2]:
                parts.append(f"- {ep.task[:80]}... failed: {ep.error or 'unknown'}")
                if ep.signal and ep.signal.negative_factors:
                    parts.append(f"  Issues: {', '.join(ep.signal.negative_factors)}")

        # Add lessons
        lessons = self.get_past_lessons()
        if lessons:
            parts.append(f"\nLessons learned: {', '.join(lessons[:5])}")

        return "\n".join(parts)

    async def get_memory_stats(self) -> dict[str, Any]:
        """Get episodic memory statistics."""
        if self._memory:
            return await self._memory.get_stats()
        return {"error": "No memory initialized"}


# ============================================================================
# Mixin for existing agents
# ============================================================================


class LearningMixin:
    """
    Mixin to add learning capabilities to existing agents.

    Usage:
        class MyLearningAgent(LearningMixin, MyAgent):
            pass
    """

    _memory: EpisodicMemory | None = None
    _current_episode: Episode | None = None
    _task_type: str = "general"

    def init_learning(
        self,
        memory: EpisodicMemory | None = None,
        task_type: str = "general",
    ) -> None:
        """Initialize learning capabilities."""
        self._memory = memory or EpisodicMemory()
        self._task_type = task_type

    def start_episode(self, task: str) -> None:
        """Start recording an episode."""
        self._current_episode = Episode(
            task=task,
            task_type=self._task_type,
            agent_type=self.__class__.__name__,
        )

    def end_episode(
        self,
        success: bool,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """End and store the episode."""
        if not self._current_episode:
            return

        outcome = EpisodeOutcome.SUCCESS if success else EpisodeOutcome.FAILURE
        self._current_episode.complete(outcome, result, error)

        if self._memory:
            import asyncio
            asyncio.create_task(self._memory.store(self._current_episode))

    def record_action(self, action_type: str, **details: Any) -> None:
        """Record an action."""
        if self._current_episode:
            self._current_episode.add_action(action_type, details)

    def record_tool_call(self, tool_name: str) -> None:
        """Record a tool call."""
        if self._current_episode:
            self._current_episode.add_tool_call(tool_name)
