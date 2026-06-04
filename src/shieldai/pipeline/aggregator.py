"""Result aggregation for multimodal content moderation.

Combines text and image moderation results using a conservative (max-confidence)
strategy and applies configurable thresholds to derive a final verdict.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from shieldai.config import ThresholdConfig
from shieldai.logging_config import get_logger
from shieldai.models import (
    AggregatedResult,
    CategoryScore,
    ContentCategory,
    ModerationResult,
    ModerationVerdict,
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Map each ContentCategory to its corresponding ThresholdConfig attribute name.
_CATEGORY_THRESHOLD_MAP: dict[ContentCategory, str] = {
    ContentCategory.TOXIC: "toxic",
    ContentCategory.HATE_SPEECH: "hate_speech",
    ContentCategory.SPAM: "spam",
    ContentCategory.NSFW: "nsfw",
    ContentCategory.VIOLENCE: "toxic",  # reuse toxic threshold for violence
}


class ResultAggregator:
    """Combines text and image moderation results into a single verdict.

    For each content category the aggregator takes the **maximum** confidence
    across modalities (conservative approach) and then applies the configured
    reject / needs-review thresholds to produce a final verdict.
    """

    def __init__(self, thresholds: ThresholdConfig) -> None:
        """Initialise the aggregator with threshold configuration.

        Args:
            thresholds: Confidence thresholds that control when content is
                rejected or flagged for review.
        """
        self._thresholds = thresholds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        text_result: ModerationResult | None,
        image_result: ModerationResult | None,
    ) -> AggregatedResult:
        """Aggregate text and image moderation results.

        Args:
            text_result: Result from text moderation, or ``None`` if no text
                was provided.
            image_result: Result from image moderation, or ``None`` if no
                image was provided.

        Returns:
            An ``AggregatedResult`` containing the combined scores, final
            verdict, and total processing time.

        Raises:
            ValueError: If both *text_result* and *image_result* are ``None``.
        """
        start_ns = time.perf_counter_ns()

        if text_result is None and image_result is None:
            raise ValueError(
                "At least one of text_result or image_result must be provided."
            )

        # ----- compute combined scores -----
        if text_result is not None and image_result is not None:
            final_scores = self._merge_scores(text_result, image_result)
            total_processing_ms = (
                text_result.processing_time_ms + image_result.processing_time_ms
            )
        elif text_result is not None:
            final_scores = list(text_result.scores)
            total_processing_ms = text_result.processing_time_ms
        else:
            assert image_result is not None  # guarded above
            final_scores = list(image_result.scores)
            total_processing_ms = image_result.processing_time_ms

        # ----- determine verdict -----
        final_verdict = self._determine_verdict(final_scores)

        aggregation_overhead_ms = (time.perf_counter_ns() - start_ns) / 1e6
        total_processing_ms += aggregation_overhead_ms

        logger.info(
            "aggregation_complete",
            final_verdict=final_verdict.value,
            categories={s.category.value: s.confidence for s in final_scores},
            text_present=text_result is not None,
            image_present=image_result is not None,
            total_processing_time_ms=round(total_processing_ms, 2),
        )

        return AggregatedResult(
            text_result=text_result,
            image_result=image_result,
            final_verdict=final_verdict,
            final_scores=final_scores,
            processing_time_ms=round(total_processing_ms, 2),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_scores(
        text_result: ModerationResult,
        image_result: ModerationResult,
    ) -> list[CategoryScore]:
        """Merge scores from two results, taking the MAX per category."""
        score_map: dict[ContentCategory, float] = {}

        for score in text_result.scores:
            score_map[score.category] = score.confidence

        for score in image_result.scores:
            existing = score_map.get(score.category, 0.0)
            score_map[score.category] = max(existing, score.confidence)

        return [
            CategoryScore(category=cat, confidence=conf)
            for cat, conf in score_map.items()
        ]

    def _determine_verdict(
        self,
        scores: list[CategoryScore],
    ) -> ModerationVerdict:
        """Apply thresholds to scores and return the appropriate verdict.

        Logic:
        1. If **any** category exceeds its per-category reject threshold →
           ``REJECTED``.
        2. Else if **any** category exceeds the global ``needs_review``
           threshold → ``NEEDS_REVIEW``.
        3. Otherwise → ``APPROVED``.
        """
        needs_review = False

        for score in scores:
            if score.category == ContentCategory.SAFE:
                continue

            reject_threshold = self._get_reject_threshold(score.category)

            if score.confidence >= reject_threshold:
                logger.debug(
                    "threshold_exceeded",
                    category=score.category.value,
                    confidence=score.confidence,
                    reject_threshold=reject_threshold,
                    decision="REJECTED",
                )
                return ModerationVerdict.REJECTED

            if score.confidence >= self._thresholds.needs_review:
                needs_review = True
                logger.debug(
                    "threshold_exceeded",
                    category=score.category.value,
                    confidence=score.confidence,
                    needs_review_threshold=self._thresholds.needs_review,
                    decision="NEEDS_REVIEW",
                )

        return ModerationVerdict.NEEDS_REVIEW if needs_review else ModerationVerdict.APPROVED

    def _get_reject_threshold(self, category: ContentCategory) -> float:
        """Return the reject threshold for a given category."""
        attr_name = _CATEGORY_THRESHOLD_MAP.get(category)
        if attr_name is None:
            # Fall back to the lowest configured threshold for unknown categories.
            return min(
                self._thresholds.toxic,
                self._thresholds.hate_speech,
                self._thresholds.spam,
                self._thresholds.nsfw,
            )
        return getattr(self._thresholds, attr_name)
