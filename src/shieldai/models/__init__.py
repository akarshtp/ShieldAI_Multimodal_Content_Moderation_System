"""Abstract base classifier and shared data types for moderation models."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ContentCategory(str, enum.Enum):
    """Categories of content that can be detected by moderation models."""

    SAFE = "safe"
    TOXIC = "toxic"
    HATE_SPEECH = "hate_speech"
    SPAM = "spam"
    NSFW = "nsfw"
    VIOLENCE = "violence"


class ModerationVerdict(str, enum.Enum):
    """Final verdict for a moderation request."""

    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True)
class CategoryScore:
    """Confidence score for a single content category."""

    category: ContentCategory
    confidence: float  # 0.0 to 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )


@dataclass
class ModerationResult:
    """Result from a single moderation model (text or image).

    Attributes:
        scores: Per-category confidence scores from the model.
        verdict: The final moderation verdict.
        input_type: Whether the input was ``text`` or ``image``.
        processing_time_ms: Time taken for inference in milliseconds.
        model_name: Name/ID of the model that produced this result.
        metadata: Additional metadata (e.g., detected language, image dimensions).
    """

    scores: list[CategoryScore]
    verdict: ModerationVerdict
    input_type: str  # "text" or "image"
    processing_time_ms: float
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def highest_risk_category(self) -> CategoryScore | None:
        """Return the category with the highest non-safe confidence score."""
        non_safe = [s for s in self.scores if s.category != ContentCategory.SAFE]
        if not non_safe:
            return None
        return max(non_safe, key=lambda s: s.confidence)

    @property
    def is_safe(self) -> bool:
        """Return ``True`` if the verdict is APPROVED."""
        return self.verdict == ModerationVerdict.APPROVED


@dataclass
class AggregatedResult:
    """Combined moderation result from multiple modalities.

    Attributes:
        text_result: Result from text moderation (if text was provided).
        image_result: Result from image moderation (if image was provided).
        final_verdict: The overall verdict combining all signals.
        final_scores: Aggregated scores across all modalities.
        processing_time_ms: Total processing time in milliseconds.
    """

    text_result: ModerationResult | None
    image_result: ModerationResult | None
    final_verdict: ModerationVerdict
    final_scores: list[CategoryScore]
    processing_time_ms: float

    @property
    def is_safe(self) -> bool:
        """Return ``True`` if the final verdict is APPROVED."""
        return self.final_verdict == ModerationVerdict.APPROVED


class BaseClassifier(ABC):
    """Abstract interface that all moderation classifiers must implement."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the model weights and tokenizer/processor into memory.

        This method is called once during application startup.
        """

    @abstractmethod
    def predict(self, input_data: Any) -> ModerationResult:
        """Run inference on a single input and return moderation scores.

        Args:
            input_data: The input to classify (text string or image).

        Returns:
            A ``ModerationResult`` with per-category scores and a verdict.
        """

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return ``True`` if the model is loaded and ready for inference."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the human-readable name of this model."""
