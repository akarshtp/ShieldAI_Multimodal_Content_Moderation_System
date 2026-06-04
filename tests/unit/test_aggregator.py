"""Unit tests for ResultAggregator — multi-modal verdict aggregation."""

from __future__ import annotations

import pytest

from shieldai.config import ThresholdConfig
from shieldai.models import (
    AggregatedResult,
    CategoryScore,
    ContentCategory,
    ModerationResult,
    ModerationVerdict,
)
from shieldai.pipeline.aggregator import ResultAggregator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def aggregator() -> ResultAggregator:
    """An aggregator configured with default thresholds."""
    return ResultAggregator(thresholds=ThresholdConfig())


@pytest.fixture
def safe_image_result() -> ModerationResult:
    """A safe image moderation result."""
    return ModerationResult(
        scores=[
            CategoryScore(category=ContentCategory.SAFE, confidence=0.90),
            CategoryScore(category=ContentCategory.VIOLENCE, confidence=0.05),
            CategoryScore(category=ContentCategory.NSFW, confidence=0.03),
            CategoryScore(category=ContentCategory.HATE_SPEECH, confidence=0.02),
        ],
        verdict=ModerationVerdict.APPROVED,
        input_type="image",
        processing_time_ms=45.0,
        model_name="test-image-model",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAggregate:
    """Tests for ResultAggregator.aggregate()."""

    def test_aggregate_text_only(
        self,
        aggregator: ResultAggregator,
        mock_safe_result: ModerationResult,
    ) -> None:
        """A single text result should pass through with an APPROVED verdict."""
        result = aggregator.aggregate(text_result=mock_safe_result, image_result=None)
        assert isinstance(result, AggregatedResult)
        assert result.final_verdict == ModerationVerdict.APPROVED
        assert result.text_result is mock_safe_result
        assert result.image_result is None

    def test_aggregate_image_only(
        self,
        aggregator: ResultAggregator,
        safe_image_result: ModerationResult,
    ) -> None:
        """A single image result should pass through with an APPROVED verdict."""
        result = aggregator.aggregate(text_result=None, image_result=safe_image_result)
        assert isinstance(result, AggregatedResult)
        assert result.final_verdict == ModerationVerdict.APPROVED
        assert result.image_result is safe_image_result
        assert result.text_result is None

    def test_aggregate_both_safe(
        self,
        aggregator: ResultAggregator,
        mock_safe_result: ModerationResult,
        safe_image_result: ModerationResult,
    ) -> None:
        """Two safe results should combine into an APPROVED verdict."""
        result = aggregator.aggregate(
            text_result=mock_safe_result, image_result=safe_image_result
        )
        assert result.final_verdict == ModerationVerdict.APPROVED
        assert result.text_result is not None
        assert result.image_result is not None

    def test_aggregate_text_toxic_image_safe(
        self,
        aggregator: ResultAggregator,
        mock_toxic_result: ModerationResult,
        safe_image_result: ModerationResult,
    ) -> None:
        """Toxic text + safe image should produce a REJECTED verdict."""
        result = aggregator.aggregate(
            text_result=mock_toxic_result, image_result=safe_image_result
        )
        assert result.final_verdict == ModerationVerdict.REJECTED

    def test_aggregate_takes_max_confidence(
        self,
        aggregator: ResultAggregator,
    ) -> None:
        """The MAX aggregation strategy should pick the higher confidence per category."""
        text_result = ModerationResult(
            scores=[
                CategoryScore(category=ContentCategory.TOXIC, confidence=0.30),
                CategoryScore(category=ContentCategory.NSFW, confidence=0.10),
            ],
            verdict=ModerationVerdict.APPROVED,
            input_type="text",
            processing_time_ms=10.0,
            model_name="test",
        )
        image_result = ModerationResult(
            scores=[
                CategoryScore(category=ContentCategory.TOXIC, confidence=0.20),
                CategoryScore(category=ContentCategory.NSFW, confidence=0.50),
            ],
            verdict=ModerationVerdict.APPROVED,
            input_type="image",
            processing_time_ms=20.0,
            model_name="test",
        )

        result = aggregator.aggregate(
            text_result=text_result, image_result=image_result
        )

        # Build a lookup from the aggregated scores
        score_map = {s.category: s.confidence for s in result.final_scores}

        # TOXIC should be max(0.30, 0.20) = 0.30
        assert score_map[ContentCategory.TOXIC] == pytest.approx(0.30)
        # NSFW should be max(0.10, 0.50) = 0.50
        assert score_map[ContentCategory.NSFW] == pytest.approx(0.50)
