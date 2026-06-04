"""Text moderation pipeline — validates, preprocesses, and classifies text."""

from __future__ import annotations

import re
import time
import unicodedata

from shieldai.logging_config import get_logger
from shieldai.models import ModerationResult, ModerationVerdict
from shieldai.models.text_classifier import TextClassifier

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_MIN_TEXT_LENGTH = 3
_MAX_TEXT_LENGTH = 10_000
_MAX_PREPROCESSED_LENGTH = 5_000

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")


class TextPipeline:
    """Orchestrates the full text-moderation flow.

    The pipeline performs three sequential steps:
    1. **Validate** — reject inputs that are empty, too short, or too long.
    2. **Preprocess** — strip HTML, remove URLs, normalise unicode, collapse
       whitespace, and truncate to a safe model-input length.
    3. **Classify** — delegate to the underlying ``TextClassifier`` for
       inference.

    Args:
        classifier: A loaded ``TextClassifier`` instance used for prediction.
    """

    def __init__(self, classifier: TextClassifier) -> None:
        self._classifier = classifier
        logger.info(
            "text_pipeline_initialized",
            classifier=classifier.model_name,
        )

    @property
    def model_name(self) -> str:
        """Return the name of the underlying classifier model."""
        return self._classifier.model_name

    def is_loaded(self) -> bool:
        """Return ``True`` if the underlying classifier model is loaded."""
        return self._classifier.is_loaded()

    # ── Public API ───────────────────────────────────────────────────────

    def preprocess(self, text: str) -> str:
        """Clean and normalise raw text for downstream classification.

        Processing steps (in order):
        1. Strip HTML tags.
        2. Remove URLs.
        3. Normalise unicode to NFKD form.
        4. Collapse consecutive whitespace into a single space and strip
           leading / trailing whitespace.
        5. Truncate to ``_MAX_PREPROCESSED_LENGTH`` characters.

        Args:
            text: The raw input text.

        Returns:
            The cleaned text string ready for the classifier.
        """
        cleaned = _HTML_TAG_RE.sub("", text)
        cleaned = _URL_RE.sub("", cleaned)
        cleaned = unicodedata.normalize("NFKD", cleaned)
        cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
        cleaned = cleaned[: _MAX_PREPROCESSED_LENGTH]

        logger.debug(
            "text_preprocessed",
            original_length=len(text),
            cleaned_length=len(cleaned),
        )
        return cleaned

    def validate(self, text: str) -> tuple[bool, str]:
        """Check whether *text* is acceptable for moderation.

        Validation rules:
        * The raw text must not exceed ``_MAX_TEXT_LENGTH`` characters.
        * After stripping leading/trailing whitespace the text must not be
          empty.
        * The stripped text must be at least ``_MIN_TEXT_LENGTH`` characters.

        Args:
            text: The raw input text (before preprocessing).

        Returns:
            A ``(is_valid, error_message)`` tuple.  When the text passes
            validation ``error_message`` is an empty string.
        """
        if len(text) > _MAX_TEXT_LENGTH:
            msg = (
                f"Text exceeds maximum length of {_MAX_TEXT_LENGTH} characters "
                f"(got {len(text)})"
            )
            logger.warning("text_validation_failed", reason="too_long", length=len(text))
            return False, msg

        stripped = text.strip()

        if not stripped:
            logger.warning("text_validation_failed", reason="empty")
            return False, "Text is empty after stripping whitespace"

        if len(stripped) < _MIN_TEXT_LENGTH:
            msg = (
                f"Text is too short (minimum {_MIN_TEXT_LENGTH} characters, "
                f"got {len(stripped)})"
            )
            logger.warning(
                "text_validation_failed",
                reason="too_short",
                length=len(stripped),
            )
            return False, msg

        return True, ""

    def moderate(self, text: str) -> ModerationResult:
        """Run the full moderation pipeline on *text*.

        Steps:
        1. Validate the raw input.
        2. Preprocess the text.
        3. Classify via the underlying ``TextClassifier``.

        If any step raises an unexpected exception the error is logged and a
        ``ModerationResult`` with a ``NEEDS_REVIEW`` verdict is returned so
        that human moderators can inspect the content.

        Args:
            text: The raw user-supplied text.

        Returns:
            A ``ModerationResult`` containing per-category scores and a
            verdict.
        """
        start = time.perf_counter()

        try:
            # Step 1 — validate
            is_valid, error_message = self.validate(text)
            if not is_valid:
                elapsed_ms = (time.perf_counter() - start) * 1_000
                logger.info(
                    "text_moderation_rejected",
                    reason=error_message,
                    processing_time_ms=round(elapsed_ms, 2),
                )
                return ModerationResult(
                    scores=[],
                    verdict=ModerationVerdict.NEEDS_REVIEW,
                    input_type="text",
                    processing_time_ms=round(elapsed_ms, 2),
                    model_name=self._classifier.model_name,
                    metadata={"error": error_message},
                )

            # Step 2 — preprocess
            cleaned = self.preprocess(text)

            # Step 3 — classify
            logger.debug("text_classification_start", text_length=len(cleaned))
            result = self._classifier.predict(cleaned)
            elapsed_ms = (time.perf_counter() - start) * 1_000

            logger.info(
                "text_moderation_complete",
                verdict=result.verdict.value,
                processing_time_ms=round(elapsed_ms, 2),
            )
            return result

        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1_000
            logger.exception(
                "text_moderation_error",
                processing_time_ms=round(elapsed_ms, 2),
            )
            return ModerationResult(
                scores=[],
                verdict=ModerationVerdict.NEEDS_REVIEW,
                input_type="text",
                processing_time_ms=round(elapsed_ms, 2),
                model_name=self._classifier.model_name,
                metadata={"error": "Internal processing error"},
            )
