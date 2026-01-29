"""
Titan Learning - Pipeline Coordinator

End-to-end learning pipeline that connects RLHF data collection,
reward signal extraction, and episodic learning for continuous improvement.

Components:
- RLHF sample collection from agent interactions
- Reward signal extraction from implicit/explicit feedback
- Episodic learning for topology selection
- Feedback processing from users
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from titan.learning.rlhf import (
    RLHFSample,
    FeedbackType,
    ResponseQuality,
)
from titan.learning.reward_signals import (
    RewardSignal,
    RewardEstimate,
    RewardSignalExtractor,
    SignalType,
)
from titan.metrics import get_metrics

if TYPE_CHECKING:
    from hive.learning import EpisodicLearner, Episode, EpisodeOutcome
    from titan.persistence.postgres import PostgresClient

logger = logging.getLogger("titan.learning.pipeline")


@dataclass
class FeedbackRequest:
    """Request for user feedback on a response."""

    request_id: UUID = field(default_factory=uuid4)
    sample_id: UUID | None = None
    response: str = ""
    prompt: str = ""
    agent_id: str = ""
    session_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FeedbackResponse:
    """User feedback on a response."""

    request_id: UUID | None = None
    rating: int | None = None  # 1-5
    accepted: bool | None = None
    correction: str | None = None
    thumbs_up: bool | None = None
    text_feedback: str | None = None
    source: str = "user"  # user, batch_completion, auto
    episode_id: str | None = None  # Link to episodic learning


@dataclass
class LearningMetrics:
    """Metrics for the learning pipeline."""

    samples_collected: int = 0
    feedback_received: int = 0
    rewards_calculated: int = 0
    episodes_recorded: int = 0
    avg_reward: float = 0.0
    avg_confidence: float = 0.0


class LearningPipeline:
    """
    End-to-end learning pipeline coordinator.

    Connects:
    - RLHFCollector: Captures interaction data
    - RewardSignalExtractor: Extracts reward signals
    - EpisodicLearner: Learns from topology decisions
    - FeedbackHandler: Processes user feedback

    Flow:
    1. Agent interaction recorded as RLHFSample
    2. User feedback received (explicit or implicit)
    3. Reward signals extracted
    4. Episodic learning updated
    5. Metrics published
    """

    def __init__(
        self,
        postgres_client: PostgresClient | None = None,
        episodic_learner: EpisodicLearner | None = None,
    ) -> None:
        self._postgres = postgres_client
        self._episodic_learner = episodic_learner
        self._reward_extractor = RewardSignalExtractor()

        # In-memory state for tracking
        self._pending_samples: dict[str, RLHFSample] = {}
        self._active_episodes: dict[str, str] = {}  # session_id -> episode_id
        self._metrics = LearningMetrics()

        # Callbacks
        self._on_feedback_callbacks: list[callable] = []
        self._on_reward_callbacks: list[callable] = []

    def start_response_tracking(
        self,
        prompt: str,
        agent_type: str,
        session_id: str = "",
        model: str = "",
        provider: str = "",
        system_prompt: str | None = None,
    ) -> str:
        """Start tracking a response for RLHF data collection.

        Args:
            prompt: The input prompt
            agent_type: Type of agent generating response
            session_id: Session identifier
            model: Model being used
            provider: LLM provider
            system_prompt: Optional system prompt

        Returns:
            Tracking ID for this response
        """
        sample = RLHFSample(
            prompt=prompt,
            agent_id=agent_type,
            session_id=session_id,
            model=model,
            provider=provider,
            system_prompt=system_prompt,
            task_type=agent_type,
        )

        tracking_id = str(sample.id)
        self._pending_samples[tracking_id] = sample
        self._metrics.samples_collected += 1

        logger.debug(f"Started tracking response: {tracking_id}")
        return tracking_id

    def complete_response(
        self,
        tracking_id: str,
        response: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> RLHFSample | None:
        """Complete response tracking with the generated response.

        Args:
            tracking_id: Tracking ID from start_response_tracking
            response: The generated response
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            The completed RLHFSample or None if not found
        """
        sample = self._pending_samples.get(tracking_id)
        if not sample:
            logger.warning(f"No pending sample for tracking ID: {tracking_id}")
            return None

        sample.response = response
        sample.response_length = len(response)
        sample.prompt_tokens = prompt_tokens
        sample.completion_tokens = completion_tokens

        logger.debug(f"Completed response tracking: {tracking_id}")
        return sample

    def record_error(
        self,
        tracking_id: str,
        error: str,
    ) -> None:
        """Record an error during response generation.

        Args:
            tracking_id: Tracking ID from start_response_tracking
            error: Error message
        """
        sample = self._pending_samples.get(tracking_id)
        if sample:
            sample.metadata["error"] = error
            sample.metadata["error_occurred"] = True
            logger.debug(f"Recorded error for {tracking_id}: {error}")

    async def process_feedback(self, feedback: FeedbackResponse) -> RewardEstimate | None:
        """Process user feedback and extract reward signals.

        Args:
            feedback: User feedback response

        Returns:
            Extracted reward estimate
        """
        self._metrics.feedback_received += 1

        # Find the sample to update
        sample = None
        if feedback.request_id:
            sample = self._pending_samples.get(str(feedback.request_id))

        # Build signals from feedback
        signals: list[RewardSignal] = []

        # Explicit rating
        if feedback.rating is not None:
            # Convert 1-5 rating to -1 to 1 scale
            normalized = (feedback.rating - 3) / 2
            signals.append(RewardSignal(
                signal_type=SignalType.EXPLICIT_RATING,
                value=normalized,
                confidence=1.0,
                raw_value=feedback.rating,
            ))

        # Thumbs up/down
        if feedback.thumbs_up is not None:
            signals.append(RewardSignal(
                signal_type=SignalType.THUMBS_UP_DOWN,
                value=1.0 if feedback.thumbs_up else -1.0,
                confidence=0.9,
                raw_value=feedback.thumbs_up,
            ))

        # Acceptance
        if feedback.accepted is not None:
            signals.append(RewardSignal(
                signal_type=SignalType.TASK_COMPLETION,
                value=1.0 if feedback.accepted else -0.5,
                confidence=0.85,
                raw_value=feedback.accepted,
            ))

        # Aggregate into reward estimate
        if signals:
            reward_estimate = self._reward_extractor.aggregate_signals(signals)
            self._metrics.rewards_calculated += 1

            # Update running averages
            n = self._metrics.rewards_calculated
            self._metrics.avg_reward = (
                self._metrics.avg_reward * (n - 1) + reward_estimate.reward
            ) / n
            self._metrics.avg_confidence = (
                self._metrics.avg_confidence * (n - 1) + reward_estimate.confidence
            ) / n

            # Record Prometheus metrics
            metrics = get_metrics()
            metrics.learning_signal_recorded(reward_estimate.reward, reward_estimate.confidence)

            # Update sample if found
            if sample:
                sample.human_rating = feedback.rating
                sample.accepted = feedback.accepted
                sample.correction = feedback.correction
                sample.feedback_type = (
                    FeedbackType.EXPLICIT_RATING if feedback.rating
                    else FeedbackType.ACCEPTANCE if feedback.accepted is not None
                    else FeedbackType.THUMBS if feedback.thumbs_up is not None
                    else FeedbackType.IMPLICIT_SIGNAL
                )

                # Store sample
                if self._postgres:
                    await self._store_sample(sample)

            # Update episodic learning if linked
            if feedback.episode_id and self._episodic_learner:
                self._episodic_learner.update_from_reward(
                    feedback.episode_id,
                    reward_estimate.reward,
                )
                self._metrics.episodes_recorded += 1

            # Trigger callbacks
            for callback in self._on_feedback_callbacks:
                try:
                    callback(feedback, reward_estimate)
                except Exception as e:
                    logger.warning(f"Feedback callback error: {e}")

            for callback in self._on_reward_callbacks:
                try:
                    callback(reward_estimate)
                except Exception as e:
                    logger.warning(f"Reward callback error: {e}")

            logger.info(
                f"Processed feedback: reward={reward_estimate.reward:.3f}, "
                f"confidence={reward_estimate.confidence:.3f}"
            )
            return reward_estimate

        return None

    async def process_batch_completion(
        self,
        batch_id: str,
        success_rate: float,
        total_tokens: int = 0,
        total_cost: float = 0.0,
    ) -> RewardEstimate | None:
        """Process batch completion as implicit feedback.

        Synthesizes feedback from batch metrics.

        Args:
            batch_id: Batch job ID
            success_rate: Success rate (0-100)
            total_tokens: Total tokens used
            total_cost: Total cost in USD

        Returns:
            Extracted reward estimate
        """
        # Convert success rate to rating (0-100 -> 1-5)
        rating = max(1, min(5, int(success_rate / 20) + 1))

        feedback = FeedbackResponse(
            rating=rating,
            accepted=success_rate >= 50,
            source="batch_completion",
            text_feedback=f"Batch {batch_id}: {success_rate:.1f}% success",
        )

        return await self.process_feedback(feedback)

    async def _store_sample(self, sample: RLHFSample) -> None:
        """Store RLHF sample to PostgreSQL."""
        if not self._postgres:
            return

        try:
            data = {
                "id": str(sample.id),
                "prompt": sample.prompt,
                "response": sample.response,
                "system_prompt": sample.system_prompt,
                "human_rating": sample.human_rating,
                "feedback_type": sample.feedback_type.value if sample.feedback_type else None,
                "correction": sample.correction,
                "accepted": sample.accepted,
                "session_id": sample.session_id,
                "agent_id": sample.agent_id,
                "model": sample.model,
                "provider": sample.provider,
                "task_type": sample.task_type,
                "prompt_tokens": sample.prompt_tokens,
                "completion_tokens": sample.completion_tokens,
                "metadata": sample.metadata,
                "created_at": sample.timestamp.isoformat(),
            }
            await self._postgres.insert_rlhf_sample(data)
        except Exception as e:
            logger.warning(f"Failed to store RLHF sample: {e}")

    def link_episode(self, session_id: str, episode_id: str) -> None:
        """Link a session to an episodic learning episode.

        Args:
            session_id: Session identifier
            episode_id: Episode identifier from EpisodicLearner
        """
        self._active_episodes[session_id] = episode_id

    def get_episode_id(self, session_id: str) -> str | None:
        """Get the episode ID for a session.

        Args:
            session_id: Session identifier

        Returns:
            Episode ID or None
        """
        return self._active_episodes.get(session_id)

    def on_feedback(self, callback: callable) -> None:
        """Register a callback for feedback events.

        Args:
            callback: Function taking (FeedbackResponse, RewardEstimate)
        """
        self._on_feedback_callbacks.append(callback)

    def on_reward(self, callback: callable) -> None:
        """Register a callback for reward calculation events.

        Args:
            callback: Function taking RewardEstimate
        """
        self._on_reward_callbacks.append(callback)

    def get_metrics(self) -> LearningMetrics:
        """Get current learning metrics.

        Returns:
            LearningMetrics with current statistics
        """
        return self._metrics


# Singleton pattern
_pipeline: LearningPipeline | None = None


def get_learning_pipeline() -> LearningPipeline:
    """Get the global learning pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = LearningPipeline()
    return _pipeline


def set_learning_pipeline(pipeline: LearningPipeline) -> None:
    """Set the global learning pipeline instance."""
    global _pipeline
    _pipeline = pipeline
