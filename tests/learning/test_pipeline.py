"""Tests for the learning pipeline."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from titan.learning.pipeline import (
    LearningPipeline,
    FeedbackRequest,
    FeedbackResponse,
    LearningMetrics,
    get_learning_pipeline,
    set_learning_pipeline,
)
from titan.learning.rlhf import FeedbackType


class TestLearningPipeline:
    """Tests for LearningPipeline class."""

    def test_constructor_defaults(self):
        """Test default initialization."""
        pipeline = LearningPipeline()
        assert pipeline._pending_samples == {}
        assert pipeline._active_episodes == {}
        assert isinstance(pipeline._metrics, LearningMetrics)

    def test_start_response_tracking(self):
        """Test starting response tracking."""
        pipeline = LearningPipeline()
        tracking_id = pipeline.start_response_tracking(
            prompt="Test prompt",
            agent_type="coder",
            session_id="session-1",
            model="gpt-4",
            provider="openai",
        )

        assert tracking_id is not None
        assert len(tracking_id) > 0
        assert tracking_id in pipeline._pending_samples
        assert pipeline._metrics.samples_collected == 1

    def test_complete_response(self):
        """Test completing response tracking."""
        pipeline = LearningPipeline()
        tracking_id = pipeline.start_response_tracking(
            prompt="Test prompt",
            agent_type="coder",
        )

        sample = pipeline.complete_response(
            tracking_id=tracking_id,
            response="Test response",
            prompt_tokens=100,
            completion_tokens=50,
        )

        assert sample is not None
        assert sample.response == "Test response"
        assert sample.prompt_tokens == 100
        assert sample.completion_tokens == 50
        assert sample.response_length == len("Test response")

    def test_complete_response_not_found(self):
        """Test completing unknown tracking ID."""
        pipeline = LearningPipeline()
        sample = pipeline.complete_response(
            tracking_id="unknown-id",
            response="Test response",
        )
        assert sample is None

    def test_record_error(self):
        """Test recording error during response generation."""
        pipeline = LearningPipeline()
        tracking_id = pipeline.start_response_tracking(
            prompt="Test prompt",
            agent_type="coder",
        )

        pipeline.record_error(tracking_id, "Test error")

        sample = pipeline._pending_samples[tracking_id]
        assert sample.metadata["error"] == "Test error"
        assert sample.metadata["error_occurred"] is True

    @pytest.mark.asyncio
    async def test_process_feedback_explicit_rating(self):
        """Test processing explicit rating feedback."""
        pipeline = LearningPipeline()
        tracking_id = pipeline.start_response_tracking(
            prompt="Test prompt",
            agent_type="coder",
        )
        pipeline.complete_response(tracking_id, "Test response")

        feedback = FeedbackResponse(
            request_id=pipeline._pending_samples[tracking_id].id,
            rating=4,
        )

        reward = await pipeline.process_feedback(feedback)

        assert reward is not None
        assert -1.0 <= reward.reward <= 1.0
        assert 0.0 <= reward.confidence <= 1.0
        assert pipeline._metrics.feedback_received == 1
        assert pipeline._metrics.rewards_calculated == 1

    @pytest.mark.asyncio
    async def test_process_feedback_thumbs_up(self):
        """Test processing thumbs up feedback."""
        pipeline = LearningPipeline()

        feedback = FeedbackResponse(
            thumbs_up=True,
        )

        reward = await pipeline.process_feedback(feedback)

        assert reward is not None
        assert reward.reward > 0  # Thumbs up should be positive

    @pytest.mark.asyncio
    async def test_process_feedback_thumbs_down(self):
        """Test processing thumbs down feedback."""
        pipeline = LearningPipeline()

        feedback = FeedbackResponse(
            thumbs_up=False,
        )

        reward = await pipeline.process_feedback(feedback)

        assert reward is not None
        assert reward.reward < 0  # Thumbs down should be negative

    @pytest.mark.asyncio
    async def test_process_feedback_acceptance(self):
        """Test processing acceptance feedback."""
        pipeline = LearningPipeline()

        feedback = FeedbackResponse(
            accepted=True,
        )

        reward = await pipeline.process_feedback(feedback)

        assert reward is not None
        assert reward.reward > 0  # Accepted should be positive

    @pytest.mark.asyncio
    async def test_process_feedback_no_signals(self):
        """Test processing feedback with no signals."""
        pipeline = LearningPipeline()

        feedback = FeedbackResponse(
            source="test",
        )

        reward = await pipeline.process_feedback(feedback)

        assert reward is None  # No signals to extract

    @pytest.mark.asyncio
    async def test_process_batch_completion(self):
        """Test processing batch completion as implicit feedback."""
        pipeline = LearningPipeline()

        reward = await pipeline.process_batch_completion(
            batch_id="batch-1",
            success_rate=80.0,
            total_tokens=10000,
            total_cost=1.5,
        )

        assert reward is not None
        assert pipeline._metrics.feedback_received == 1

    @pytest.mark.asyncio
    async def test_process_batch_completion_low_success(self):
        """Test processing batch with low success rate."""
        pipeline = LearningPipeline()

        reward = await pipeline.process_batch_completion(
            batch_id="batch-1",
            success_rate=20.0,
        )

        assert reward is not None
        # Low success rate should result in lower reward

    def test_link_episode(self):
        """Test linking session to episode."""
        pipeline = LearningPipeline()
        pipeline.link_episode("session-1", "episode-123")

        assert pipeline.get_episode_id("session-1") == "episode-123"

    def test_get_episode_id_not_found(self):
        """Test getting episode ID for unknown session."""
        pipeline = LearningPipeline()
        assert pipeline.get_episode_id("unknown") is None

    def test_on_feedback_callback(self):
        """Test feedback callback registration."""
        pipeline = LearningPipeline()
        callback = MagicMock()
        pipeline.on_feedback(callback)

        assert callback in pipeline._on_feedback_callbacks

    def test_on_reward_callback(self):
        """Test reward callback registration."""
        pipeline = LearningPipeline()
        callback = MagicMock()
        pipeline.on_reward(callback)

        assert callback in pipeline._on_reward_callbacks

    @pytest.mark.asyncio
    async def test_callbacks_invoked(self):
        """Test that callbacks are invoked on feedback."""
        pipeline = LearningPipeline()

        feedback_callback = MagicMock()
        reward_callback = MagicMock()
        pipeline.on_feedback(feedback_callback)
        pipeline.on_reward(reward_callback)

        feedback = FeedbackResponse(rating=5)
        await pipeline.process_feedback(feedback)

        feedback_callback.assert_called_once()
        reward_callback.assert_called_once()

    def test_get_metrics(self):
        """Test getting metrics."""
        pipeline = LearningPipeline()
        pipeline.start_response_tracking(prompt="test", agent_type="coder")

        metrics = pipeline.get_metrics()

        assert isinstance(metrics, LearningMetrics)
        assert metrics.samples_collected == 1


class TestLearningMetrics:
    """Tests for LearningMetrics dataclass."""

    def test_defaults(self):
        """Test default values."""
        metrics = LearningMetrics()
        assert metrics.samples_collected == 0
        assert metrics.feedback_received == 0
        assert metrics.rewards_calculated == 0
        assert metrics.episodes_recorded == 0
        assert metrics.avg_reward == 0.0
        assert metrics.avg_confidence == 0.0


class TestFeedbackRequest:
    """Tests for FeedbackRequest dataclass."""

    def test_defaults(self):
        """Test default values."""
        request = FeedbackRequest()
        assert request.request_id is not None
        assert request.response == ""
        assert request.prompt == ""
        assert request.agent_id == ""
        assert request.session_id == ""


class TestFeedbackResponse:
    """Tests for FeedbackResponse dataclass."""

    def test_defaults(self):
        """Test default values."""
        response = FeedbackResponse()
        assert response.request_id is None
        assert response.rating is None
        assert response.accepted is None
        assert response.correction is None
        assert response.thumbs_up is None
        assert response.source == "user"

    def test_with_rating(self):
        """Test with rating."""
        response = FeedbackResponse(rating=4)
        assert response.rating == 4

    def test_with_thumbs(self):
        """Test with thumbs up."""
        response = FeedbackResponse(thumbs_up=True)
        assert response.thumbs_up is True


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def test_get_learning_pipeline_singleton(self):
        """Test that get_learning_pipeline returns singleton."""
        # Reset first
        set_learning_pipeline(LearningPipeline())

        pipeline1 = get_learning_pipeline()
        pipeline2 = get_learning_pipeline()
        assert pipeline1 is pipeline2

    def test_set_learning_pipeline(self):
        """Test setting custom pipeline."""
        custom_pipeline = LearningPipeline()
        set_learning_pipeline(custom_pipeline)

        pipeline = get_learning_pipeline()
        assert pipeline is custom_pipeline

        # Reset for other tests
        set_learning_pipeline(LearningPipeline())
