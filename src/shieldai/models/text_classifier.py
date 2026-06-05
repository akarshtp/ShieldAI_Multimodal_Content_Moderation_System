"""Text toxicity classifier using HuggingFace ``unitary/toxic-bert``.

Provides multi-label toxicity classification and maps raw model labels to
the unified :class:`~shieldai.models.ContentCategory` enum used across the
moderation pipeline.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shieldai.config import get_settings
from shieldai.logging_config import get_logger
from shieldai.models import (
    BaseClassifier,
    CategoryScore,
    ContentCategory,
    ModerationResult,
    ModerationVerdict,
)

logger = get_logger(__name__)

# Maps each toxic-bert output label to the corresponding ContentCategory.
# When multiple toxic-bert labels share a category the highest confidence wins.
_LABEL_TO_CATEGORY: dict[str, ContentCategory] = {
    "toxic": ContentCategory.TOXIC,
    "severe_toxic": ContentCategory.TOXIC,
    "obscene": ContentCategory.NSFW,
    "threat": ContentCategory.VIOLENCE,
    "insult": ContentCategory.HATE_SPEECH,
    "identity_hate": ContentCategory.HATE_SPEECH,
}

# Ordered list matching the model's output head positions.
_LABEL_ORDER: list[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


class TextClassifier(BaseClassifier):
    """Multi-label text toxicity classifier built on ``unitary/toxic-bert``.

    The model produces independent sigmoid probabilities for six toxicity
    labels which are then collapsed into the project's four unsafe
    :class:`ContentCategory` values plus a synthetic ``SAFE`` score.

    Example::

        classifier = TextClassifier()
        classifier.load_model()
        result = classifier.predict("some user comment")
        print(result.verdict, result.scores)
    """

    def __init__(self) -> None:
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: str = get_settings().model.device

    # ------------------------------------------------------------------
    # BaseClassifier interface
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Download (if necessary) and load the toxic-bert model.

        Uses the model cache directory and device configured in
        :func:`shieldai.config.get_settings`.
        """
        settings = get_settings()
        model_id = settings.model.text_model_name
        cache_dir = settings.model.model_cache_dir

        logger.info(
            "loading_text_model",
            model_id=model_id,
            device=self._device,
            cache_dir=str(cache_dir),
        )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=cache_dir,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                cache_dir=cache_dir,
            )
            self._model.to(self._device)  # type: ignore[union-attr]
            self._model.eval()  # type: ignore[union-attr]
            logger.info("text_model_loaded", model_id=model_id)
        except Exception:
            logger.exception("text_model_load_failed", model_id=model_id)
            raise

    def predict(self, input_data: Any) -> ModerationResult:
        """Classify a text string and return a :class:`ModerationResult`.

        Args:
            input_data: The text string to classify.

        Returns:
            A :class:`ModerationResult` containing per-category confidence
            scores and the overall moderation verdict.

        Raises:
            RuntimeError: If the model has not been loaded yet.
            TypeError: If *input_data* is not a string.
        """
        if not self.is_loaded():
            raise RuntimeError("TextClassifier model is not loaded. Call load_model() first.")

        if not isinstance(input_data, str):
            raise TypeError(f"TextClassifier expects a str, got {type(input_data).__name__}")

        text: str = input_data
        settings = get_settings()
        start = time.perf_counter()

        try:
            scores = self._run_inference(text)
            verdict = self._determine_verdict(scores, settings)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            result = ModerationResult(
                scores=scores,
                verdict=verdict,
                input_type="text",
                processing_time_ms=round(elapsed_ms, 2),
                model_name=self.model_name,
                metadata={"text_length": len(text)},
            )

            logger.info(
                "text_prediction_complete",
                verdict=verdict.value,
                processing_time_ms=result.processing_time_ms,
                text_length=len(text),
            )
            return result

        except Exception:
            logger.exception("text_prediction_failed", text_length=len(text))
            raise

    def is_loaded(self) -> bool:
        """Return ``True`` if both tokenizer and model are ready."""
        return self._tokenizer is not None and self._model is not None

    @property
    def model_name(self) -> str:
        """Return the HuggingFace model identifier."""
        return get_settings().model.text_model_name

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_inference(self, text: str) -> list[CategoryScore]:
        """Tokenize *text*, run a forward pass, and return category scores.

        The six sigmoid probabilities from toxic-bert are collapsed into
        project-level categories by keeping the maximum confidence when
        multiple labels map to the same :class:`ContentCategory`.  A
        synthetic ``SAFE`` score is computed as ``1 - max(unsafe scores)``.
        """
        assert self._tokenizer is not None
        assert self._model is not None

        settings = get_settings()
        encoding = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.model.max_text_length,
            padding=True,
        )
        encoding = {k: v.to(self._device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = self._model(**encoding).logits  # shape: (1, 6)

        probabilities = torch.sigmoid(logits).squeeze(0).cpu().tolist()

        # Collapse raw labels into ContentCategory (max confidence wins).
        category_confidence: dict[ContentCategory, float] = {}
        for label, prob in zip(_LABEL_ORDER, probabilities, strict=True):
            category = _LABEL_TO_CATEGORY[label]
            if category not in category_confidence or prob > category_confidence[category]:
                category_confidence[category] = prob

        # Derive a synthetic SAFE score.
        max_unsafe = max(category_confidence.values()) if category_confidence else 0.0
        category_confidence[ContentCategory.SAFE] = round(1.0 - max_unsafe, 6)

        return [
            CategoryScore(category=cat, confidence=round(conf, 6))
            for cat, conf in category_confidence.items()
        ]

    @staticmethod
    def _determine_verdict(
        scores: list[CategoryScore],
        settings: Any,
    ) -> ModerationVerdict:
        """Derive a verdict from *scores* using configured thresholds.

        Decision logic (evaluated in order):

        1. **REJECTED** - any unsafe category score ≥ its rejection threshold.
        2. **NEEDS_REVIEW** - any unsafe category score ≥ the review threshold.
        3. **APPROVED** - otherwise.
        """
        thresholds = settings.thresholds
        category_threshold_map: dict[ContentCategory, float] = {
            ContentCategory.TOXIC: thresholds.toxic,
            ContentCategory.HATE_SPEECH: thresholds.hate_speech,
            ContentCategory.NSFW: thresholds.nsfw,
            ContentCategory.VIOLENCE: thresholds.toxic,  # no dedicated threshold; reuse toxic
        }

        for score in scores:
            if score.category == ContentCategory.SAFE:
                continue
            reject_threshold = category_threshold_map.get(score.category, thresholds.toxic)
            if score.confidence >= reject_threshold:
                return ModerationVerdict.REJECTED

        for score in scores:
            if score.category == ContentCategory.SAFE:
                continue
            if score.confidence >= thresholds.needs_review:
                return ModerationVerdict.NEEDS_REVIEW

        return ModerationVerdict.APPROVED
