"""Unit tests for the TextPipeline preprocessing, validation, and moderation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shieldai.models import ModerationResult, ModerationVerdict

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from shieldai.pipeline.text_pipeline import TextPipeline

# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


class TestPreprocess:
    """Tests for TextPipeline.preprocess()."""

    def test_preprocess_strips_html(self, text_pipeline: TextPipeline) -> None:
        """HTML tags should be completely removed."""
        raw = "<p>Hello <b>world</b></p>"
        result = text_pipeline.preprocess(raw)
        assert "<" not in result
        assert ">" not in result
        assert "Hello" in result
        assert "world" in result

    def test_preprocess_removes_urls(self, text_pipeline: TextPipeline) -> None:
        """HTTP(S) URLs should be stripped from the text."""
        raw = "Visit https://example.com or http://foo.bar/baz for more."
        result = text_pipeline.preprocess(raw)
        assert "https://" not in result
        assert "http://" not in result
        assert "example.com" not in result

    def test_preprocess_normalizes_whitespace(
        self, text_pipeline: TextPipeline
    ) -> None:
        """Multiple consecutive whitespace characters should collapse to one space."""
        raw = "  hello   world\t\tfoo  "
        result = text_pipeline.preprocess(raw)
        assert "  " not in result
        assert result == "hello world foo"

    def test_preprocess_handles_unicode(self, text_pipeline: TextPipeline) -> None:
        """Unicode text should pass through without being stripped."""
        raw = "Ünïcödé chàracters — works great™"
        result = text_pipeline.preprocess(raw)
        # The core content should survive (NFKD may decompose some chars)
        assert "great" in result
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidate:
    """Tests for TextPipeline.validate()."""

    def test_validate_empty_text(self, text_pipeline: TextPipeline) -> None:
        """Empty string should fail validation."""
        is_valid, msg = text_pipeline.validate("")
        assert is_valid is False
        assert "empty" in msg.lower()

    def test_validate_too_short(self, text_pipeline: TextPipeline) -> None:
        """Text shorter than 3 characters should fail validation."""
        is_valid, msg = text_pipeline.validate("ab")
        assert is_valid is False
        assert "short" in msg.lower()

    def test_validate_valid_text(self, text_pipeline: TextPipeline) -> None:
        """Normal text should pass validation."""
        is_valid, msg = text_pipeline.validate("Hello, world!")
        assert is_valid is True
        assert msg == ""


# ---------------------------------------------------------------------------
# Full moderation flow tests
# ---------------------------------------------------------------------------


class TestModerate:
    """Tests for TextPipeline.moderate()."""

    def test_moderate_safe_text(
        self, text_pipeline: TextPipeline, mock_safe_result: ModerationResult
    ) -> None:
        """Full pipeline should return a result with APPROVED verdict for safe text."""
        result = text_pipeline.moderate("This is perfectly safe text.")
        assert isinstance(result, ModerationResult)
        assert result.verdict == ModerationVerdict.APPROVED

    def test_moderate_calls_classifier(
        self, text_pipeline: TextPipeline, mock_text_classifier: MagicMock
    ) -> None:
        """The classifier's predict() should be called with the preprocessed text."""
        text_pipeline.moderate("Some valid input text here.")
        mock_text_classifier.predict.assert_called_once()

        # The argument should be the *preprocessed* text (no HTML, no URLs, etc.)
        call_arg = mock_text_classifier.predict.call_args[0][0]
        assert isinstance(call_arg, str)
        assert len(call_arg) > 0
